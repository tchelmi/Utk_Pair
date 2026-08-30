import os
import random
from typing import List, Tuple, Optional

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


def parse_age_from_filename(filename: str) -> Optional[int]:
    """
    Estrae l'età da un nome file UTKFace del tipo:
    age_gender_race_time.jpg (es. 25_0_0_20170116174525125.jpg).
    Ritorna None se il formato non è valido.
    """
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    parts = name.split("_")
    if len(parts) < 4:
        return None
    try:
        return int(parts[0])
    except ValueError:
        return None


def collect_utkface(root_dir: str) -> List[Tuple[str, int]]:
    """
    Scansiona la cartella UTKFace e ritorna una lista di tuple:
    (percorso_assoluto_immagine, eta).
    """
    imgs: List[Tuple[str, int]] = []
    for fname in os.listdir(root_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        age = parse_age_from_filename(fname)
        if age is None:
            continue
        imgs.append((os.path.join(root_dir, fname), age))

    if len(imgs) < 2:
        raise RuntimeError("Troppo poche immagini valide trovate in UTKFace.")

    return imgs


class SingleAgeDataset(Dataset):
    """
    Dataset UTKFace per stima dell'età (age regression).

    Ogni elemento è (img, age).
    """

    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform or T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, age = self.samples[index]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, float(age)


class PairwiseAgeDataset(Dataset):
    """
    Dataset che genera coppie di facce per il confronto di età.

    Ogni elemento è (img1, img2, label) dove:
      - label = 0 se img1 è più giovane di img2 (age1 < age2)
      - label = 1 altrimenti
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform=None,
        min_age_gap: int = 10,
        seed: Optional[int] = None,
    ):
        self.samples = samples
        self.min_age_gap = int(min_age_gap)
        self.transform = transform or T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.rng = random.Random(seed)

        if len(self.samples) < 2:
            raise ValueError("Servono almeno 2 immagini per creare coppie.")

    def __len__(self) -> int:
        # Numero virtuale di coppie; può essere aumentato a piacere
        return len(self.samples) * 2

    def _sample_two_indices(self) -> Tuple[int, int]:
        n = len(self.samples)
        max_tries = 1000
        for _ in range(max_tries):
            i1 = self.rng.randrange(n)
            i2 = self.rng.randrange(n)
            if i2 == i1:
                continue
            age1 = self.samples[i1][1]
            age2 = self.samples[i2][1]
            if abs(age1 - age2) >= self.min_age_gap:
                return i1, i2

        raise RuntimeError(
            "Impossibile campionare una coppia con differenza di età "
            f">= {self.min_age_gap} dopo {max_tries} tentativi. "
            "Prova a ridurre min_age_gap o verifica che il dataset contenga "
            "abbastanza età diverse."
        )

    def __getitem__(self, index: int):
        # Ignoriamo l'indice e generiamo una coppia casuale
        i1, i2 = self._sample_two_indices()
        path1, age1 = self.samples[i1]
        path2, age2 = self.samples[i2]

        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        label = 0 if age1 < age2 else 1
        return img1, img2, label


# Alias più descrittivo (come richiesto nel progetto)
AgeComparisonDataset = PairwiseAgeDataset

