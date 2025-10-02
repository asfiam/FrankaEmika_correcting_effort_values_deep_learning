#!/usr/bin/env python3
# train_seq.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "neural_net_franka_sequential"))

import torch
from torch.utils.data import DataLoader, random_split
from neural_net_franka_sequential.model_seq import CNNLSTMNet
from neural_net_franka_sequential.dataset_seq import RosbagSequentialDataset
from neural_net_franka_sequential.utils_seq import train_one_epoch, evaluate

CSV_FILE = "data_from_all_rosbags.csv"
SEQ_LEN = 4
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_PATH = "cnn_lstm_franka_residual.pth"

def main():
    dataset = RosbagSequentialDataset(CSV_FILE, seq_len=SEQ_LEN)

    # split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # initialize model
    model = CNNLSTMNet(numeric_dim=len(dataset.numeric_cols), output_dim=len(dataset.target_cols)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # training loop
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_loss, _ = evaluate(model, val_loader, DEVICE, plot=False)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # final evaluation on test set
    test_loss, avg_inference_time = evaluate(model, test_loader, DEVICE, plot=True)
    print(f"\n✅ Final Test Loss (residuals): {test_loss:.6f}")
    print(f"⚡ Avg inference time per sample: {avg_inference_time*1000:.3f} ms")

    # --- save model parameters ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model parameters saved to {MODEL_SAVE_PATH}")

    # optional: print first few parameters to verify
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}, mean={param.data.mean():.6f}")
        break  # remove break to see all

if __name__ == "__main__":
    main()
