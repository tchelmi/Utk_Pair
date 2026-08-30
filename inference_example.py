import os
import torch
from PIL import Image
import torchvision.transforms as T
from model import ResNetPairwiseAge
from age_regression_model import AgeResNet18, AgeFeatureExtractor
from pair_mlp_model import PairwiseAgeMLP

def get_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def load_model(ckpt_path, device):
    """Carica il modello in base al nome del file gestendo correttamente i dizionari dei checkpoint."""
    
    def safe_load(model, path):
        checkpoint = torch.load(path, map_location=device)
        # Se il file è un dizionario (come i nostri), estraiamo solo 'model_state_dict'
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        return model

    if "mlp" in ckpt_path:
        # Carichiamo la pipeline Siamese completa
        regressor = AgeResNet18(pretrained=False)
        reg_ckpt = os.path.join("models", "best_age_reg.pth")
        regressor = safe_load(regressor, reg_ckpt)
        
        feature_extractor = AgeFeatureExtractor(regressor).to(device)
        model = PairwiseAgeMLP(feature_dim=feature_extractor.out_dim).to(device)
        model = safe_load(model, ckpt_path)
        
        return model, "siamese", feature_extractor
    else:
        # Carichiamo la ResNet 6-canali standard
        model = ResNetPairwiseAge(pretrained=False).to(device)
        model = safe_load(model, ckpt_path)
        return model, "early_fusion", None

def infer(img_path1, img_path2, ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mode, feature_extractor = load_model(ckpt_path, device)
    model.eval()
    transform = get_transform()

    img1 = transform(Image.open(img_path1).convert("RGB")).unsqueeze(0).to(device)
    img2 = transform(Image.open(img_path2).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        if mode == "siamese":
            f1 = feature_extractor(img1)
            f2 = feature_extractor(img2)
            logits = model(f1, f2)
        else:
            logits = model(img1, img2)
        
        pred = logits.argmax(dim=1).item()

    result = "PIÙ GIOVANE" if pred == 0 else "PIÙ VECCHIA"
    print(f"Modello usato: {mode} | Risultato: La prima foto è {result}")

if __name__ == "__main__":
    # ESEMPIO: Per testare il Siamese (MLP)
    #infer("data/UTKFace/chayouba.jpg", "data/UTKFace/foto_arbia.jpg", "models/best_mlp.pth")
    # ESEMPIO: Per testare l'Early Fusion (6 canali)
     infer("data/UTKFace/chayouba.jpg", "data/UTKFace/foto_arbia.jpg", "models/best_pairwise.pth")