import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from neurolive.data.rgbe_h5_dataset import RGBEGazeH5Dataset
from neurolive.models.scnn import LivenessSCNN


DATA_ROOT = r"data\RGBE_Gaze_dataset\processed_data\event_training"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ---------------------------------------------------------
# Load real RGBE-Gaze data
# ---------------------------------------------------------

dataset = RGBEGazeH5Dataset(DATA_ROOT)

print("Total real samples:", len(dataset))


# Use only 128 samples for the sanity test.
# This is NOT the real training run.
subset = Subset(dataset, range(min(128, len(dataset))))

loader = DataLoader(
    subset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = LivenessSCNN().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
)


# ---------------------------------------------------------
# Training sanity test
# ---------------------------------------------------------

model.train()

for epoch in range(2):

    total_loss = 0.0

    for batch_idx, batch in enumerate(loader):

        x = batch["voxel"].to(device)

        # IMPORTANT:
        # These are genuine-only samples.
        #
        # This label is ONLY being used to test that
        # forward/backward works.
        y = torch.ones(
            x.size(0),
            dtype=torch.long,
            device=device,
        )

        optimizer.zero_grad()

        logits = model(x)

        loss = F.cross_entropy(logits, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if batch_idx == 0:
            print(
                "Epoch:",
                epoch + 1,
                "Input:",
                tuple(x.shape),
                "Output:",
                tuple(logits.shape),
                "Loss:",
                loss.item(),
            )

    print(
        f"Epoch {epoch + 1} average loss:",
        total_loss / len(loader),
    )


print()
print("REAL DATA TRAINING SANITY TEST: PASSED")
print("Samples processed:", len(subset))
print("This is NOT a liveness accuracy result.")