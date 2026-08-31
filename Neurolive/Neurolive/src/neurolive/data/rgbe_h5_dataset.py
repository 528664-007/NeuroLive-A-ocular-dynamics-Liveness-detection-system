"""
RGBE-Gaze H5 dataset adapter.

Loads the already-processed RGBE-Gaze event voxel grids.

Each voxel H5 file contains:
    (N, 5, 64, 96)

where each sample is:
    (5, 64, 96)

This adapter is for genuine RGBE-Gaze data validation/training.
It does NOT invent replay/print/AI-injection labels.
"""

from pathlib import Path
from typing import Optional

import h5py
import torch
from torch.utils.data import Dataset


class RGBEGazeH5Dataset(Dataset):
    """
    Dataset over RGBE-Gaze processed voxel-grid H5 files.

    By default, both left and right eye voxel grids are loaded and
    averaged into one (5, 64, 96) tensor.

    If use_right=False, only the left-eye voxel grid is returned.

    Each item returns:
        {
            "voxel": Tensor (5, 64, 96),
            "gaze_left": Tensor (2,),
            "gaze_right": Tensor (2,),
            "headpose_left": Tensor (2,),
            "headpose_right": Tensor (2,),
            "user": int,
            "experiment": int,
            "sample_index": int,
            "is_synthetic": False,
            "liveness_label": 1
        }

    IMPORTANT:
        liveness_label=1 means genuine only.
        This dataset contains no attack recordings.
    """

    def __init__(
        self,
        data_root: str,
        use_right: bool = True,
        combine_eyes: bool = True,
    ):
        self.data_root = Path(data_root)
        self.use_right = use_right
        self.combine_eyes = combine_eyes

        if not self.data_root.exists():
            raise FileNotFoundError(
                f"RGBE-Gaze directory does not exist: {self.data_root}"
            )

        self.records = []

        left_files = sorted(
            self.data_root.rglob(
                "64_96_voxel_grid_left_eye_normalized_user*_exp*.h5"
            )
        )

        if not left_files:
            raise FileNotFoundError(
                "No RGBE-Gaze left-eye voxel H5 files found under "
                f"{self.data_root}"
            )

        for left_file in left_files:
            name = left_file.name

            # Example:
            # 64_96_voxel_grid_left_eye_normalized_user_10_exp1.h5
            stem = left_file.stem

            try:
                user_part = stem.split("_user_")[1]
                user_str, exp_part = user_part.split("_exp")
                user = int(user_str)
                experiment = int(exp_part)
            except (IndexError, ValueError):
                raise ValueError(
                    f"Could not parse user/experiment from filename: {name}"
                )

            right_file = left_file.with_name(
  			  name.replace(
        		"voxel_grid_left_eye_normalized",
        		"voxel_grid_right_eye_normalized",
    			)
			)

            # Replace left-eye with right-eye while preserving directory.
            right_file = left_file.with_name(
                name.replace(
                    "voxel_grid_left_eye_normalized",
                    "voxel_grid_right_eye_normalized",
                )
            )

            if self.use_right and not right_file.exists():
                raise FileNotFoundError(
                    f"Matching right-eye file not found for:\n{left_file}\n"
                    f"Expected:\n{right_file}"
                )

            with h5py.File(left_file, "r") as h5:
                n_samples = int(h5["data"].shape[0])

            if self.use_right:
                with h5py.File(right_file, "r") as h5:
                    right_n = int(h5["data"].shape[0])

                if right_n != n_samples:
                    raise ValueError(
                        f"Left/right sample count mismatch for "
                        f"user {user}, experiment {experiment}: "
                        f"{n_samples} vs {right_n}"
                    )

            for sample_index in range(n_samples):
                self.records.append(
                    {
                        "left_path": left_file,
                        "right_path": right_file if self.use_right else None,
                        "sample_index": sample_index,
                        "user": user,
                        "experiment": experiment,
                    }
                )

        self._handles = {}

    def __len__(self):
        return len(self.records)

    def _get_file(self, path: Path):
        """
        Open H5 lazily and keep one handle per worker/process.
        """
        key = str(path)

        if key not in self._handles:
            self._handles[key] = h5py.File(path, "r")

        return self._handles[key]

    def __getitem__(self, idx: int):
        record = self.records[idx]

        left_h5 = self._get_file(record["left_path"])
        left = left_h5["data"][record["sample_index"]]

        left = torch.from_numpy(left).float()

        if self.use_right:
            right_h5 = self._get_file(record["right_path"])
            right = right_h5["data"][record["sample_index"]]
            right = torch.from_numpy(right).float()

            if self.combine_eyes:
                voxel = (left + right) / 2.0
            else:
                voxel = torch.cat([left, right], dim=0)
        else:
            voxel = left

        result = {
            "voxel": voxel,
            "user": record["user"],
            "experiment": record["experiment"],
            "sample_index": record["sample_index"],
            "liveness_label": torch.tensor(1, dtype=torch.long),
            "is_synthetic": False,
        }

        return result

    def close(self):
        """Close all open H5 files."""
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass

        self._handles.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass