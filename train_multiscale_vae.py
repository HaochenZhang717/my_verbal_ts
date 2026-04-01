import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb

from multi_scale_vae_models.multiscale_vae import DualVAE   # ⚠️ 根据你的路径改


# =========================
# Args
# =========================

def get_args():
    parser = argparse.ArgumentParser()

    # ===== data =====
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)

    # ===== training =====
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)

    # ===== model =====
    parser.add_argument("--z_channels", type=int, default=32)
    parser.add_argument("--latent_channels", type=int, default=None)
    parser.add_argument("--ch", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)

    # ⚠️ tuple/list 类型（重要）
    parser.add_argument("--ch_mult", type=int, nargs="+", default=[1, 1, 2])

    # ===== decomposition =====
    parser.add_argument("--decomposition_width", type=int, default=64)
    parser.add_argument("--frequency_groups", type=int, default=4)

    # ⚠️ 这个是 list，必须用 nargs="+"
    parser.add_argument(
        "--moving_avg_kernel_sizes",
        type=int,
        nargs="+",
        default=[5, 7]
    )

    # ===== loss weights =====
    parser.add_argument("--lambda_recon_low", type=float, default=1.0)
    parser.add_argument("--lambda_recon_mid", type=float, default=1.0)
    parser.add_argument("--lambda_recon_high", type=float, default=1.0)
    parser.add_argument("--lambda_recon_total", type=float, default=1.0)

    parser.add_argument("--lambda_kl_low", type=float, default=1e-3)
    parser.add_argument("--lambda_kl_mid", type=float, default=1e-3)
    parser.add_argument("--lambda_kl_high", type=float, default=1e-3)

    # ===== misc =====
    parser.add_argument("--save_dir", type=str, default="./ckpts_multiscale")
    parser.add_argument("--device", type=str, default="cuda")

    # ===== wandb =====
    parser.add_argument("--wandb_project", type=str, default="multiscale_vae")
    parser.add_argument("--wandb_name", type=str, default="debug")

    return parser.parse_args()

# =========================
# Dataset
# =========================
def load_dataset(npy_path):
    data = np.load(npy_path)
    # data = torch.tensor(data, dtype=torch.float32).permute(0, 2, 1)
    data = torch.tensor(data, dtype=torch.float32)

    print(f"Loaded {npy_path}: {data.shape}")

    return TensorDataset(data)


# =========================
# Loss
# =========================
def compute_loss(out, args):

    loss = 0.0

    # ===== reconstruction =====
    loss += args.lambda_recon_low * out["recon_loss_low_freq"]
    loss += args.lambda_recon_mid * out["recon_loss_mid_freq"]
    loss += args.lambda_recon_high * out["recon_loss_high_freq"]
    loss += args.lambda_recon_total * out["recon_loss_overall"]

    # ===== KL =====
    loss += args.lambda_kl_low * out["kl_loss_low_freq"]
    loss += args.lambda_kl_mid * out["kl_loss_mid_freq"]
    loss += args.lambda_kl_high * out["kl_loss_high_freq"]

    return loss


# =========================
# Train One Epoch
# =========================
def train_one_epoch(model, dataloader, optimizer, device, args, global_step):

    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Train")

    for batch in pbar:
        x = batch[0].to(device)

        out = model(x)
        loss = compute_loss(out, args)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        # ===== wandb log =====
        wandb.log({
            "train/loss": loss.item(),

            "train/recon_low": out["recon_loss_low_freq"].item(),
            "train/recon_mid": out["recon_loss_mid_freq"].item(),
            "train/recon_high": out["recon_loss_high_freq"].item(),
            "train/recon_total": out["recon_loss_overall"].item(),

            "train/kl_low": out["kl_loss_low_freq"].item(),
            "train/kl_mid": out["kl_loss_mid_freq"].item(),
            "train/kl_high": out["kl_loss_high_freq"].item(),

        }, step=global_step)

        global_step += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader), global_step


# =========================
# Validation
# =========================
@torch.no_grad()
def validate(model, dataloader, device, args):

    model.eval()

    total = {
        "loss": 0.0,
        "recon_low": 0.0,
        "recon_mid": 0.0,
        "recon_high": 0.0,
        "recon_total": 0.0,
        "kl_low": 0.0,
        "kl_mid": 0.0,
        "kl_high": 0.0,
    }

    for batch in dataloader:
        x = batch[0].to(device)

        out = model(x)
        loss = compute_loss(out, args)

        total["loss"] += loss.item()

        total["recon_low"] += out["recon_loss_low_freq"].item()
        total["recon_mid"] += out["recon_loss_mid_freq"].item()
        total["recon_high"] += out["recon_loss_high_freq"].item()
        total["recon_total"] += out["recon_loss_overall"].item()

        total["kl_low"] += out["kl_loss_low_freq"].item()
        total["kl_mid"] += out["kl_loss_mid_freq"].item()
        total["kl_high"] += out["kl_loss_high_freq"].item()

    n = len(dataloader)
    for k in total:
        total[k] /= n

    return total


# =========================
# Train
# =========================
def train(args):

    device = args.device if torch.cuda.is_available() else "cpu"

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args)
    )

    # ===== dataset =====
    train_dataset = load_dataset(args.train_path)
    val_dataset = load_dataset(args.val_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    sample = train_dataset[0][0]
    T, C = sample.shape

    # ===== model =====
    model = DualVAE(
        in_channels=C,
        z_channels=args.z_channels,
        latent_channels=args.latent_channels,
        ch=args.ch,
        dropout=args.dropout,
        ch_mult=args.ch_mult,
        decomposition_width=args.decomposition_width,
        frequency_groups=args.frequency_groups,
        moving_avg_kernel_sizes=args.moving_avg_kernel_sizes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")

    global_step = 0

    # =========================
    # Loop
    # =========================
    for epoch in range(args.epochs):

        print(f"\n===== Epoch {epoch} =====")

        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, device, args, global_step
        )

        val_dict = validate(model, val_loader, device, args)

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val   Loss: {val_dict['loss']:.6f}")

        # ===== wandb (val) =====
        wandb.log({
            "epoch": epoch,

            "val/loss": val_dict["loss"],

            "val/recon_low": val_dict["recon_low"],
            "val/recon_mid": val_dict["recon_mid"],
            "val/recon_high": val_dict["recon_high"],
            "val/recon_total": val_dict["recon_total"],

            "val/kl_low": val_dict["kl_low"],
            "val/kl_mid": val_dict["kl_mid"],
            "val/kl_high": val_dict["kl_high"],

        }, step=global_step)

        # ===== save =====
        torch.save(model.state_dict(), os.path.join(args.save_dir, "last.pt"))

        if val_dict["loss"] < best_val_loss:
            best_val_loss = val_dict["loss"]
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pt"))
            print("Saved BEST")


# =========================
if __name__ == "__main__":
    args = get_args()
    train(args)