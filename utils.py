# utils.py
import os
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter

def create_writer(experiment_name: str, config_name: str) -> SummaryWriter:
    """
    Crea un SummaryWriter per TensorBoard con un timestamp univoco.
    Es: runs/pairwise_resnet/lr1e-4_bs64_20231027_153022/
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join("runs", experiment_name, f"{config_name}_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[INFO] TensorBoard logger creato in: {log_dir}")
    return writer

def save_checkpoint(state: dict, is_best: bool, save_dir: str, filename: str = "last_checkpoint.pth", best_filename: str = "best_model.pth"):
    """
    Salva lo stato del modello. Se is_best è True, lo salva anche come best_model.pth.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Salva sempre l'ultimo checkpoint (utile per il resume)
    last_path = os.path.join(save_dir, filename)
    torch.save(state, last_path)
    
    # Se è il migliore finora, salva una copia separata
    if is_best:
        best_path = os.path.join(save_dir, best_filename)
        torch.save(state, best_path)
        print(f"  ✅ New best model saved to: {best_path}")

def load_checkpoint(resume_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, device: str = "cpu"):
    """
    Carica i pesi, l'epoca e lo stato dell'ottimizzatore da un file .pth.
    """
    if os.path.isfile(resume_path):
        print(f"=> Caricamento checkpoint da '{resume_path}'")
        checkpoint = torch.load(resume_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        start_epoch = 1
        best_val_loss = float('inf')
        
        # Ripristina l'ottimizzatore e altre variabili se esistono nel dict
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
            
        print(f"=> Ripresa addestramento dall'epoca {start_epoch}")
        return start_epoch, best_val_loss
    else:
        print(f"=> NESSUN CHECKPOINT TROVATO IN '{resume_path}'. Partenza da zero.")
        return 1, float('inf')