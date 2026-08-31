"""Phase 2 - train the unified joint model on RGBE-Gaze H5 data.

The RGBE-Gaze H5 dataset currently provides voxel grids and genuine labels,
but does not provide ocular segmentation labels. Therefore H5 mode trains
only the liveness head while preserving the complete JointLivenessModel
architecture.

This is a real-data pipeline training run, NOT a genuine-vs-replay
liveness benchmark because the current dataset contains genuine samples only.
"""

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
from neurolive.models.joint_mamba import JointLivenessModel


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
    """Collate samples returned by RGBEGazeH5Dataset."""

    return {
        "voxel": torch.stack(
            [sample["voxel"] for sample in batch]
        ),
        "liveness_label": torch.stack(
            [sample["liveness_label"] for sample in batch]
        ),
        "is_synthetic": [
            sample.get("is_synthetic", False)
            for sample in batch
        ],
        "user": [
            sample.get("user")
            for sample in batch
        ],
        "experiment": [
            sample.get("experiment")
            for sample in batch
        ],
        "sample_index": [
            sample.get("sample_index")
            for sample in batch
        ],
    }


def train_h5_epoch(
    model,
    loader,
    optimizer,
    device,
):
    """Train only the liveness head using real H5 labels."""

    model.train()

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    num_batches = 0

    for batch in loader:

        voxel = batch["voxel"].to(device)
        labels = batch["liveness_label"].to(device)

        optimizer.zero_grad()

        _, liveness_logits = model(voxel)

        loss = criterion(
            liveness_logits,
            labels,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_h5(
    model,
    loader,
    device,
):
    """Evaluate the H5 liveness head.

    NOTE:
    The current H5 dataset contains genuine samples only.
    Therefore accuracy is not a genuine-vs-replay benchmark.
    """

    model.eval()

    predictions = []
    labels = []

    for batch in loader:

        voxel = batch["voxel"].to(device)

        _, liveness_logits = model(voxel)

        preds = (
            liveness_logits
            .argmax(dim=-1)
            .cpu()
            .numpy()
        )

        predictions.append(preds)

        labels.append(
            batch["liveness_label"]
            .cpu()
            .numpy()
        )

    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)

    accuracy = float(
        (predictions == labels).mean()
    )

    return {
        "liveness_accuracy": accuracy,
        "n_samples": int(len(labels)),
        "is_synthetic": False,
        "segmentation_available": False,
        "note": (
            "RGBE-Gaze H5 currently contains genuine samples only. "
            "This accuracy is NOT a genuine-vs-replay liveness benchmark. "
            "Replay/attack samples are required for valid APCER, BPCER and ACER."
        ),
    }


def train_original_dataset(args):
    """Preserve the original index.jsonl/synthetic training path."""

    dataset = build_dataset(args)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_liveness,
    )

    model = JointLivenessModel(
        d_model=args.d_model,
        force_fallback=args.force_fallback_backbone,
    ).to(args.device)

    print(
        f"Backbone in use: {model.backbone_name}"
    )

    from neurolive.models.losses import (
        MultiTaskLivenessLoss,
    )

    loss_fn = MultiTaskLivenessLoss(
        args.seg_weight,
        args.liveness_weight,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    model.train()

    for epoch in range(args.epochs):

        total = 0.0

        for batch in loader:

            voxel = batch["voxel"].to(args.device)

            seg_labels = batch[
                "seg_labels"
            ].to(args.device)

            live_labels = batch[
                "liveness_label"
            ].to(args.device)

            optimizer.zero_grad()

            seg_logits, live_logits = model(
                voxel
            )

            losses = loss_fn(
                seg_logits,
                live_logits,
                seg_labels,
                live_labels,
            )

            losses["total"].backward()

            optimizer.step()

            total += losses["total"].item()

        print(
            f"epoch {epoch + 1}/{args.epochs} "
            f"total_loss={total / len(loader):.4f}"
        )

    return model, loader


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--h5",
        action="store_true",
        help="Use the RGBE-Gaze H5 adapter.",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    parser.add_argument(
        "--smoke-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num-bins",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=96,
    )

    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--seg-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--liveness-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--force-fallback-backbone",
        action="store_true",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="runs/phase2_joint",
    )

    args = parser.parse_args()

    if (
        not args.smoke_test
        and args.data_root is None
    ):
        raise SystemExit(
            "Pass --data-root for a real run, "
            "or --smoke-test."
        )

    device = torch.device(args.device)

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # =========================================================
    # REAL RGBE-GAZE H5 MODE
    # =========================================================

    if args.h5:

        dataset = RGBEGazeH5Dataset(
            args.data_root
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=h5_collate,
        )

        print(
            "Dataset size:",
            len(dataset),
        )

        model = JointLivenessModel(
            d_model=args.d_model,
            force_fallback=args.force_fallback_backbone,
        ).to(device)

        print(
            "Backbone in use:",
            model.backbone_name,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
        )

        for epoch in range(args.epochs):

            loss = train_h5_epoch(
                model,
                loader,
                optimizer,
                device,
            )

            print(
                f"epoch {epoch + 1}/{args.epochs} "
                f"live_loss={loss:.4f}"
            )

        metrics = evaluate_h5(
            model,
            loader,
            device,
        )

        metrics["backbone"] = (
            model.backbone_name
        )

        metrics[
            "segmentation_available"
        ] = False

        out_dir = Path(args.out_dir)

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            model.state_dict(),
            out_dir / "joint_model.pt",
        )

        (
            out_dir / "metrics.json"
        ).write_text(
            json.dumps(
                metrics,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            json.dumps(
                metrics,
                indent=2,
            )
        )

        print(
            f"Saved Joint model checkpoint + "
            f"metrics to {out_dir}/"
        )

        return

    # =========================================================
    # ORIGINAL MODE
    # =========================================================

    model, loader = train_original_dataset(
        args
    )

    out_dir = Path(args.out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        out_dir / "joint_model.pt",
    )

    print(
        f"Saved checkpoint to {out_dir}/"
    )


if __name__ == "__main__":
    main()