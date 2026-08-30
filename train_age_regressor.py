import os
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T

from datasets import collect_utkface, SingleAgeDataset
from age_regression_model import AgeResNet18
from utils import create_writer, save_checkpoint, load_checkpoint


def make_dataloaders(
    utk_root: str,
    batch_size: int = 64,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader]:
    samples = collect_utkface(utk_root)
    n_total = len(samples)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    train_samples, val_samples = random_split(samples, [n_train, n_val])

    # Aggiunta Data Augmentation per il training per evitare l'overfitting
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = SingleAgeDataset(list(train_samples), transform=train_transform)
    val_ds = SingleAgeDataset(list(val_samples), transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader


def train_age_regressor(
    utk_root: str,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    save_dir: str,
    resume_path: str = None
) -> None:
    """
    Train AgeResNet18 to predict scalar age (regression).
    """
    train_loader, val_loader = make_dataloaders(utk_root, batch_size=batch_size)

    # Usa weights=... invece di pretrained=True per evitare il warning di PyTorch
    model = AgeResNet18(pretrained=True).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # SETUP RESUME
    start_epoch = 1
    best_val_loss = float('inf')
    if resume_path:
        start_epoch, best_val_loss = load_checkpoint(resume_path, model, optimizer, device)

    # SETUP TENSORBOARD
    config_name = f"lr{lr}_bs{batch_size}"
    writer = create_writer("age_regressor", config_name)

    for epoch in range(start_epoch, epochs + 1):
        # -------- TRAIN --------
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        n_train = 0
        
        for imgs, ages in train_loader:
            imgs, ages = imgs.to(device), ages.to(device).float()

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, ages)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * ages.size(0)
            train_mae += torch.abs(preds - ages).sum().item() # Errore assoluto in anni
            n_train += ages.size(0)

        train_loss /= n_train
        train_mae /= n_train

        # -------- VAL --------
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        n_val = 0
        
        with torch.no_grad():
            for imgs, ages in val_loader:
                imgs, ages = imgs.to(device), ages.to(device).float()

                preds = model(imgs)
                loss = criterion(preds, ages)

                val_loss += loss.item() * ages.size(0)
                val_mae += torch.abs(preds - ages).sum().item()
                n_val += ages.size(0)

        val_loss /= n_val
        val_mae /= n_val

        print(
            f"[AgeReg] Epoch {epoch}/{epochs} "
            f"| train_mse={train_loss:.4f}, train_mae={train_mae:.2f}anni "
            f"| val_mse={val_loss:.4f}, val_mae={val_mae:.2f}anni"
        )

        # LOG TENSORBOARD
        writer.add_scalar('Loss/Train_MSE', train_loss, epoch)
        writer.add_scalar('Loss/Validation_MSE', val_loss, epoch)
        writer.add_scalar('Error/Train_MAE_Years', train_mae, epoch)
        writer.add_scalar('Error/Validation_MAE_Years', val_mae, epoch)

        # CHECKPOINT
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss
        }
        save_checkpoint(checkpoint_state, is_best, save_dir, filename="last_age_reg.pth", best_filename="best_age_reg.pth")

    writer.close()
    print("Addestramento Fase 1 (Regressore) completato!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Addestramento Age Regressor (Fase 1)")
    parser.add_argument("--data_dir", type=str, default=os.path.join("data", "UTKFace"), help="Path dataset UTKFace")
    parser.add_argument("--epochs", type=int, default=10, help="Numero di epoche")
    parser.add_argument("--batch_size", type=int, default=64, help="Grandezza batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="models", help="Cartella dei modelli")
    parser.add_argument("--resume", type=str, default=None, help="Path checkpoint resume")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_age_regressor(args.data_dir, args.epochs, args.lr, args.batch_size, device, args.save_dir, args.resume)