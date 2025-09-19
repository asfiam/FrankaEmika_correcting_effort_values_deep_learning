#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "neural_net_franka"))

import torch
from torch.utils.data import DataLoader, random_split
from neural_net_franka.dataset import RosbagDataset
from neural_net_franka.model import FusionNet
from neural_net_franka.utils import train_one_epoch, evaluate
import matplotlib.pyplot as plt

def main():
    csv_file = "data_from_all_rosbags.csv"  
    dataset = RosbagDataset(csv_file)

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size   = int(0.15 * total_size)
    test_size  = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    numeric_dim = len(dataset.numeric_cols)
    model = FusionNet(numeric_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    val_losses = []

    # --- Training Loop ---
    for epoch in range(5):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss   = evaluate(model, val_loader, device)  

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.8f}, val_loss={val_loss:.8f}")

    # --- Plot training & validation losses ---
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation Loss (Residual Learning)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Final evaluation on test set ---
    print("\n🔹 Running final evaluation on TEST set...")
    test_loss = evaluate(model, test_loader, device, plot=True)
    print(f"Final Test Loss: {test_loss:.8f}")

if __name__ == "__main__":
    main()
