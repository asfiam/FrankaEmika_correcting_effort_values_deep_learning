# utils_seq.py
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

loss_fn = nn.MSELoss()

def train_one_epoch(model, loader, optimizer, device,
                    numeric_force_idx=0, numeric_torque_idx=1):
    model.train()
    total_loss = 0
    for numeric, images, y in loader:
        numeric, images, y = numeric.to(device), images.to(device), y.to(device)

        # --- compute residual target ---
        sim_effort = numeric[:, -1, [numeric_force_idx, numeric_torque_idx]]
        residual_target = y - sim_effort

        optimizer.zero_grad()
        residual_pred = model(numeric, images)

        loss = loss_fn(residual_pred, residual_target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * numeric.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, dataloader, device, plot=False,
             numeric_force_idx=0, numeric_torque_idx=1):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_time = 0.0
    total_samples = 0

    all_target, all_pred, all_sim = [], [], []
    all_resid_true, all_resid_pred = [], []

    with torch.no_grad():
        for numeric, images, target in dataloader:
            numeric, images, target = numeric.to(device), images.to(device), target.to(device)

            # inference
            start = time.time()
            residual_pred = model(numeric, images)   # predict residual
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()

            batch_time = end - start
            total_time += batch_time
            total_samples += numeric.size(0)

            # baseline simulated effort
            sim_effort = numeric[:, -1, [numeric_force_idx, numeric_torque_idx]]
            residual_target = target - sim_effort

            # loss on residuals
            loss = criterion(residual_pred, residual_target)
            total_loss += loss.item() * numeric.size(0)

            # collect for plotting
            all_target.append(target.cpu())
            all_pred.append((sim_effort + residual_pred).cpu())  # corrected effort
            all_sim.append(sim_effort.cpu())
            all_resid_true.append(residual_target.cpu())
            all_resid_pred.append(residual_pred.cpu())

    # concat
    all_target     = torch.cat(all_target, dim=0)
    all_pred       = torch.cat(all_pred, dim=0)
    all_sim        = torch.cat(all_sim, dim=0)
    all_resid_true = torch.cat(all_resid_true, dim=0)
    all_resid_pred = torch.cat(all_resid_pred, dim=0)

    avg_loss = total_loss / total_samples
    avg_time_per_sample = total_time / total_samples

    if plot:
        start, end = 100, 200
        fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

        axs[0].plot(range(start, end), all_target[start:end, 0], label="Real Force")
        axs[0].plot(range(start, end), all_pred[start:end, 0], label="Predicted Force (Sim+Residual)")
        axs[0].plot(range(start, end), all_sim[start:end, 0], label="Simulated Force", linestyle="--")
        axs[0].set_ylabel("Force")
        axs[0].legend()
        axs[0].set_title("Simulated vs Predicted vs Real Force")

        axs[1].plot(range(start, end), all_target[start:end, 1], label="Real Torque")
        axs[1].plot(range(start, end), all_pred[start:end, 1], label="Predicted Torque (Sim+Residual)")
        axs[1].plot(range(start, end), all_sim[start:end, 1], label="Simulated Torque", linestyle="--")
        axs[1].set_ylabel("Torque")
        axs[1].legend()
        axs[1].set_title("Simulated vs Predicted vs Real Torque")

        axs[2].plot(range(start, end), all_resid_true[start:end, 0], label="True Residual (Force)")
        axs[2].plot(range(start, end), all_resid_pred[start:end, 0], label="Predicted Residual (Force)")
        axs[2].set_ylabel("Residual Force")
        axs[2].legend()
        axs[2].set_title("Residual: True vs Predicted Force")

        axs[3].plot(range(start, end), all_resid_true[start:end, 1], label="True Residual (Torque)")
        axs[3].plot(range(start, end), all_resid_pred[start:end, 1], label="Predicted Residual (Torque)")
        axs[3].set_ylabel("Residual Torque")
        axs[3].set_xlabel("Sample Index")
        axs[3].legend()
        axs[3].set_title("Residual: True vs Predicted Torque")

        plt.tight_layout()
        plt.show()

    return avg_loss, avg_time_per_sample

