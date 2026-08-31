"""Phase 1 - train baseline models on synthetic, index-based, or RGBE-Gaze H5 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from neurolive.data.dataset import (
    RGBELivenessDataset,
    SyntheticEventLivenessDataset,
    collate_liveness,
)
from neurolive.data.rgbe_h5_dataset import RGBEGazeH5Dataset
from neurolive.eval.metrics import acer, top1_accuracy
from neurolive.models.scnn import LivenessSCNN
from neurolive.models.tcn import OcularTCN


def build_dataset(args):
    if args.smoke_test:
        return SyntheticEventLivenessDataset(
            size=args.smoke_size,
            num_bins=args.num_bins,
            height=args.height,
            width=args.width,
        )

    if args.h5:
        return RGBEGazeH5Dataset(args.data_root)

    return RGBELivenessDataset(
        args.data_root,
        num_bins=args.num_bins,
        height=args.height,
        width=args.width,
    )


def h5_collate(batch):
    """Collate RGBE-Gaze H5 samples.

    The H5 adapter provides voxel and liveness information.
    It does not provide ocular segmentation labels.
    """

    return {
        "voxel": torch.stack([x["voxel"] for x in batch]),
        "liveness_label": torch.stack(
            [x["liveness_label"] for x in batch]
        ),
        "is_synthetic": [x.get("is_synthetic", False) for x in batch],
        "user": [x.get("user") for x in batch],
        "experiment": [x.get("experiment") for x in batch],
        "sample_index": [x.get("sample_index") for x in batch],
    }


def train_scnn_one_epoch(model, loader, optimizer, device):
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0

    for batch in loader:
        voxel = batch["voxel"].to(device)
        labels = batch["liveness_label"].to(device)

        optimizer.zero_grad()

        logits = model(voxel)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_scnn(model, loader, device):
    model.eval()

    predictions = []
    labels = []

    for batch in loader:
        voxel = batch["voxel"].to(device)

        logits = model(voxel)
        preds = logits.argmax(dim=-1).cpu().numpy()

        predictions.append(preds)
        labels.append(batch["liveness_label"].numpy())

    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)

    return {
        "scnn_liveness_accuracy": top1_accuracy(predictions, labels),
        "scnn_liveness_acer": acer(predictions, labels),
        "num_evaluated_samples": int(len(labels)),
        "note": (
            "This dataset currently contains genuine RGBE-Gaze samples only. "
            "Therefore liveness accuracy/ACER is NOT a valid genuine-vs-replay "
            "benchmark until attack/replay samples are added."
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--h5", action="store_true")

    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-size", type=int, default=64)

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)

    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=96)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument(
        "--out-dir",
        type=str,
        default="runs/phase1_baseline",
    )

    args = parser.parse_args()

    if not args.smoke_test and args.data_root is None:
        raise SystemExit(
            "Pass --data-root for a real run, or --smoke-test "
            "to sanity-check the pipeline."
        )

    dataset = build_dataset(args)

    if args.h5:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=h5_collate,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_liveness,
        )

    device = torch.device(args.device)

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Dataset size:", len(dataset))

    # ---------------------------------------------------------
    # H5 REAL-DATA MODE
    # ---------------------------------------------------------

    if args.h5:
        scnn = LivenessSCNN().to(device)

        optimizer = torch.optim.Adam(
            scnn.parameters(),
            lr=args.lr,
        )

        for epoch in range(args.epochs):
            loss = train_scnn_one_epoch(
                scnn,
                loader,
                optimizer,
                device,
            )

            print(
                f"epoch {epoch + 1}/{args.epochs} "
                f"live_loss={loss:.4f}"
            )

        metrics = evaluate_scnn(
            scnn,
            loader,
            device,
        )

        metrics["is_synthetic"] = False
        metrics["segmentation_available"] = False

        print(json.dumps(metrics, indent=2))

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            scnn.state_dict(),
            out_dir / "scnn.pt",
        )

        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

        print(
            f"Saved SCNN checkpoint + metrics to {out_dir}/"
        )

        return

    # ---------------------------------------------------------
    # ORIGINAL SYNTHETIC / INDEX-BASED MODE
    # ---------------------------------------------------------

    tcn = OcularTCN(
        num_bins=args.num_bins
    ).to(device)

    scnn = LivenessSCNN().to(device)

    tcn_opt = torch.optim.Adam(
        tcn.parameters(),
        lr=args.lr,
    )

    scnn_opt = torch.optim.Adam(
        scnn.parameters(),
        lr=args.lr,
    )

    seg_ce = torch.nn.CrossEntropyLoss()
    live_ce = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):

        tcn.train()
        scnn.train()

        total_seg_loss = 0.0
        total_live_loss = 0.0

        for batch in loader:

            voxel = batch["voxel"].to(device)
            seg_labels = batch["seg_labels"].to(device)
            live_labels = batch["liveness_label"].to(device)

            # TCN
            tcn_opt.zero_grad()

            seg_logits = tcn(voxel)

            seg_loss = seg_ce(
                seg_logits.reshape(
                    -1,
                    seg_logits.shape[-1],
                ),
                seg_labels.reshape(-1),
            )

            seg_loss.backward()
            tcn_opt.step()

            # SCNN
            scnn_opt.zero_grad()

            live_logits = scnn(voxel)

            live_loss = live_ce(
                live_logits,
                live_labels,
            )

            live_loss.backward()
            scnn_opt.step()

            total_seg_loss += seg_loss.item()
            total_live_loss += live_loss.item()

        print(
            f"epoch {epoch + 1}/{args.epochs} "
            f"seg_loss={total_seg_loss / len(loader):.4f} "
            f"live_loss={total_live_loss / len(loader):.4f}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(tcn.state_dict(), out_dir / "tcn.pt")
    torch.save(scnn.state_dict(), out_dir / "scnn.pt")

    print(
        f"Saved checkpoints to {out_dir}/"
    )


if __name__ == "__main__":
    main()