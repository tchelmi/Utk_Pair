import os
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from datasets import collect_utkface, PairwiseAgeDataset
from age_regression_model import AgeResNet18, AgeFeatureExtractor
from pair_mlp_model import PairwiseAgeMLP
from utils import create_writer, save_checkpoint, load_checkpoint


def make_dataloaders(
    utk_root: str, 
    batch_size: int = 64,
    val_split: float = 0.1
) -> Tuple[DataLoader, DataLoader]:
    
    samples = collect_utkface(utk_root)
    n_val = max(1, int(len(samples) * val_split))
    n_train = len(samples) - n_val

    train_samples, val_samples = random_split(samples, [n_train, n_val])
    
    # Non passiamo transform complessi qui perché le feature congelate 
    # richiedono un input standard, non aumentato in modo aggressivo.
    train_ds = PairwiseAgeDataset(list(train_samples), min_age_gap=10, seed=123)
    val_ds = PairwiseAgeDataset(list(val_samples), min_age_gap=10, seed=124)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader


def train_pairwise_mlp(
    utk_root: str,
    age_regressor_ckpt: str,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    save_dir: str,
    resume_path: str = None
) -> None:
    """
    Train solo l'MLP head; il CNN feature extractor rimane congelato.
    """
    # 1. Caricamento della CNN pre-addestrata (Fase 1)
    if not os.path.exists(age_regressor_ckpt):
        raise FileNotFoundError(f"Impossibile trovare il modello pre-addestrato: {age_regressor_ckpt}. Esegui prima train_age_regressor.py")
        
    age_reg = AgeResNet18(pretrained=False)
    # Supporta sia il formato vecchio che quello a dizionario
    checkpoint = torch.load(age_regressor_ckpt, map_location="cpu")
    if 'model_state_dict' in checkpoint:
        age_reg.load_state_dict(checkpoint['model_state_dict'])
    else:
        age_reg.load_state_dict(checkpoint)

    feature_extractor = AgeFeatureExtractor(age_reg).to(device)
    feature_extractor.eval()

    for p in feature_extractor.parameters():
        p.requires_grad = False

    # 2. Setup dell'MLP
    feature_dim = feature_extractor.out_dim
    mlp = PairwiseAgeMLP(feature_dim=feature_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=lr, weight_decay=1e-4)

    # 3. Resume & TensorBoard
    start_epoch = 1
    best_val_loss = float('inf')
    if resume_path:
        start_epoch, best_val_loss = load_checkpoint(resume_path, mlp, optimizer, device)

    config_name = f"lr{lr}_bs{batch_size}"
    writer = create_writer("pairwise_mlp", config_name)
    train_loader, val_loader = make_dataloaders(utk_root, batch_size=batch_size)

    # 4. Loop di addestramento
    for epoch in range(start_epoch, epochs + 1):
        # -------- TRAIN --------
        mlp.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for img1, img2, labels in train_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            with torch.no_grad():
                f1 = feature_extractor(img1)
                f2 = feature_extractor(img2)

            optimizer.zero_grad()
            logits = mlp(f1, f2)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss /= total
        train_acc = correct / total

        # -------- VAL --------
        mlp.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                
                f1 = feature_extractor(img1)
                f2 = feature_extractor(img2)
                
                logits = mlp(f1, f2)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"[MLP] Epoch {epoch}/{epochs} | train_loss={train_loss:.4f}, acc={train_acc:.4f} | val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        # LOG TENSORBOARD
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)

        # CHECKPOINT
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': mlp.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss
        }
        save_checkpoint(checkpoint_state, is_best, save_dir, filename="last_mlp.pth", best_filename="best_mlp.pth")

    writer.close()
    print("Addestramento Fase 2 (MLP) completato!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Addestramento Modello Pairwise (Fase 2 - MLP su CNN congelata)")
    parser.add_argument("--data_dir", type=str, default=os.path.join("data", "UTKFace"), help="Path dataset UTKFace")
    # Usa il checkpoint del regressore che salverai in fase 1
    parser.add_argument("--age_reg_ckpt", type=str, default=os.path.join("models", "best_age_reg.pth"), help="Path ai pesi del regressore Fase 1")
    parser.add_argument("--epochs", type=int, default=5, help="Numero di epoche")
    parser.add_argument("--batch_size", type=int, default=64, help="Grandezza del batch")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="models", help="Cartella dei modelli")
    parser.add_argument("--resume", type=str, default=None, help="Path checkpoint resume (MLP)")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_pairwise_mlp(
        args.data_dir, args.age_reg_ckpt, args.epochs, args.lr, 
        args.batch_size, device, args.save_dir, args.resume
    )