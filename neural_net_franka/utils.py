import torch
import torch.nn as nn

def train_one_epoch(model, dataloader, optimizer, device,
                    numeric_force_idx=0, numeric_torque_idx=1):
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0

    for numeric, images, target in dataloader:
        numeric, images, target = numeric.to(device), images.to(device), target.to(device)

        # simulated effort (input baseline)
        sim_effort = numeric[:, [numeric_force_idx, numeric_torque_idx]]

        # ground-truth residual = real - sim
        residual_target = target - sim_effort

        optimizer.zero_grad()
        residual_pred = model(numeric, images)
        loss = criterion(residual_pred, residual_target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, plot=False, numeric_force_idx=0, numeric_torque_idx=1):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0

    all_target, all_pred, all_sim = [], [], []
    all_resid_true, all_resid_pred = [], []  # <-- new

    with torch.no_grad():
        for numeric, images, target in dataloader:
            numeric, images, target = numeric.to(device), images.to(device), target.to(device)

            # model predicts residual
            residual_pred = model(numeric, images)

            # true residual = real - simulated
            sim_effort = numeric[:, [numeric_force_idx, numeric_torque_idx]]
            residual_target = target - sim_effort

            loss = criterion(residual_pred, residual_target)
            total_loss += loss.item()

            all_target.append(target.cpu())
            all_pred.append((sim_effort + residual_pred).cpu())  # final predicted effort
            all_sim.append(sim_effort.cpu())

            # save residuals
            all_resid_true.append(residual_target.cpu())
            all_resid_pred.append(residual_pred.cpu())

    # concat everything
    all_target     = torch.cat(all_target, dim=0)
    all_pred       = torch.cat(all_pred, dim=0)
    all_sim        = torch.cat(all_sim, dim=0)
    all_resid_true = torch.cat(all_resid_true, dim=0)
    all_resid_pred = torch.cat(all_resid_pred, dim=0)

    if plot:  # Only for TEST set
        import matplotlib.pyplot as plt
        start, end = 100, 200  # sample range to visualize

        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # --- Force subplot ---
        axs[0].plot(range(start, end), all_target[start:end, 0], label="Real Force")
        axs[0].plot(range(start, end), all_pred[start:end, 0], label="Predicted Force")
        axs[0].plot(range(start, end), all_sim[start:end, 0], label="Simulated Force", linestyle="--")
        axs[0].set_ylabel("Force")
        axs[0].legend()
        axs[0].set_title("Test Set: Simulated vs Predicted vs Real Force")

        # --- Torque subplot ---
        axs[1].plot(range(start, end), all_target[start:end, 1], label="Real Torque")
        axs[1].plot(range(start, end), all_pred[start:end, 1], label="Predicted Torque")
        axs[1].plot(range(start, end), all_sim[start:end, 1], label="Simulated Torque", linestyle="--")
        axs[1].set_ylabel("Torque")
        axs[1].legend()

        # --- Residual subplot ---
        axs[2].plot(range(start, end), all_resid_true[start:end, 0], label="True Residual (Force)")
        axs[2].plot(range(start, end), all_resid_pred[start:end, 0], label="Predicted Residual (Force)")
        axs[2].plot(range(start, end), all_resid_true[start:end, 1], label="True Residual (Torque)", linestyle="--")
        axs[2].plot(range(start, end), all_resid_pred[start:end, 1], label="Predicted Residual (Torque)", linestyle="--")
        axs[2].set_xlabel("Sample index")
        axs[2].set_ylabel("Residual")
        axs[2].legend()

        plt.tight_layout()
        plt.show()

    return total_loss / len(dataloader)

