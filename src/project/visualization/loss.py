from pathlib import Path

import matplotlib.pyplot as plt

from project.config import PROJECT_ROOT


def plot_loss_over_epoch(data):
    # Parse data and group losses by epoch
    epoch_losses = {}

    for entry in data:
        parts = entry.split(',')
        epoch = int(parts[0].split()[1])
        loss = float(parts[2].split()[1])

        if epoch not in epoch_losses:
            epoch_losses[epoch] = []
        epoch_losses[epoch].append(loss)

    # Calculate average loss per epoch
    epochs = sorted(epoch_losses.keys())
    avg_losses = [sum(losses) / len(losses) for losses in epoch_losses.values()]

    # Plot the loss over epoch
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, avg_losses, marker='o', clip_on=False)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    # y ranges from 0 to 100
    plt.ylim(0, 100)
    plt.xlim(0, len(epochs))
    plt.title("Average Loss over Epochs")
    plt.savefig(Path(PROJECT_ROOT) / "artifacts" / "UNet ClinicDB Loss.svg")
    plt.show()

with open(Path(PROJECT_ROOT) / "artifacts" / "UNet ClinicDB Loss.txt", "r") as f:
    data = f.readlines()

plot_loss_over_epoch(data)
