import os
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T

from datasets import collect_utkface, PairwiseAgeDataset
from model import ResNetPairwiseAge
from utils import create_writer, save_checkpoint, load_checkpoint

def make_dataloaders(
    utk_root: str,
    batch_size: int = 64,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Crea i DataLoader di train e validation per il dataset pairwise.
    """
    samples = collect_utkface(utk_root)
    n_total = len(samples)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    train_samples, val_samples = random_split(samples, [n_train, n_val])

    train_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_ds = PairwiseAgeDataset(
        list(train_samples), transform=train_transform, min_age_gap=10, seed=42
    )
    val_ds = PairwiseAgeDataset(
        list(val_samples), transform=eval_transform, min_age_gap=10, seed=43
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )
    return train_loader, val_loader


def train(
    utk_root: str,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    save_dir: str,
    resume_path: str = None
) -> None:
    """
    Loop di training per il modello ResNetPairwiseAge.
    """
    train_loader, val_loader = make_dataloaders(utk_root, batch_size=batch_size)
    model = ResNetPairwiseAge(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # ----------------------------------------------------
    # 1. SETUP RESUME (se viene passato un checkpoint)
    # ----------------------------------------------------
    start_epoch = 1
    best_val_loss = float('inf')
    if resume_path:
        start_epoch, best_val_loss = load_checkpoint(resume_path, model, optimizer, device)

    # ----------------------------------------------------
    # 2. SETUP TENSORBOARD
    # ----------------------------------------------------
    config_name = f"lr{lr}_bs{batch_size}"
    writer = create_writer("pairwise_resnet", config_name)

    for epoch in range(start_epoch, epochs + 1):
        # -------------------- TRAIN --------------------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for img1, img2, labels in train_loader:
            img1, img2, labels = (
                img1.to(device),
                img2.to(device),
                labels.to(device),
            )

            optimizer.zero_grad()
            logits = model(img1, img2)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # -------------------- VALIDATION --------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = (
                    img1.to(device),
                    img2.to(device),
                    labels.to(device),
                )
                logits = model(img1, img2)
                loss = criterion(logits, labels)

                val_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch}/{epochs} "
            f"| train_loss={train_loss:.4f}, train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        # ----------------------------------------------------
        # 3. LOG SU TENSORBOARD
        # ----------------------------------------------------
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)

        # ----------------------------------------------------
        # 4. SALVATAGGIO CHECKPOINT
        # ----------------------------------------------------
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
        
        save_checkpoint(
            checkpoint_state, 
            is_best, 
            save_dir, 
            filename="last_pairwise.pth", 
            best_filename="best_pairwise.pth"
        )

    writer.close()
    print("Addestramento completato!")


if __name__ == "__main__":
    # ----------------------------------------------------
    # SETUP ARGPARSE (per lanciare da terminale)
    # ----------------------------------------------------
    parser = argparse.ArgumentParser(description="Addestramento Modello Pairwise (ResNet 6-canali)")
    parser.add_argument("--data_dir", type=str, default=os.path.join("data", "UTKFace"), help="Path al dataset UTKFace")
    parser.add_argument("--epochs", type=int, default=30, help="Numero di epoche")
    parser.add_argument("--batch_size", type=int, default=64, help="Grandezza del batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="models", help="Cartella dove salvare i pesi")
    parser.add_argument("--resume", type=str, default=None, help="Path al checkpoint per riprendere il training")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Avvio del training con i parametri da terminale
    train(
        utk_root=args.data_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        save_dir=args.save_dir,
        resume_path=args.resume
    )