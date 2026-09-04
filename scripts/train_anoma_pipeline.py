import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from ultralytics import YOLO
from models.autoencoder import SonarConvAutoencoder

class AnomaDataset(Dataset):
    def __init__(self, img_dir, img_size=128, augment=False):
        self.img_paths = list(Path(img_dir).glob('*.jpg')) + list(Path(img_dir).glob('*.png'))
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        p = str(self.img_paths[idx])
        im = cv2.imread(p)
        if im is None:
            im = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        if self.augment:
            if np.random.rand() > 0.5:
                im = cv2.flip(im, 1)
            if np.random.rand() > 0.5:
                im = cv2.flip(im, 0)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (self.img_size, self.img_size))
        tensor = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
        return tensor


def main():
    print('=' * 70)
    print('STARTING COMPREHENSIVE TRAINING ON ANOMA DATASET (535 IMAGES)')
    print('=' * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f'Using Compute Device: {device} ({dev_name})')

    # 1. Train Autoencoder
    train_ds = AnomaDataset('samples/anoma/train/images', augment=True)
    valid_ds = AnomaDataset('samples/anoma/valid/images', augment=False)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False, num_workers=0)

    print(f'\n[Phase 1/2] Training Autoencoder (40 Epochs, Train: {len(train_ds)}, Val: {len(valid_ds)})...')
    ae_model = SonarConvAutoencoder(in_channels=3, latent_dim=128).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(ae_model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

    best_val_loss = float('inf')
    ae_weights_path = Path('weights/autoencoder_best.pt')

    t0 = time.time()
    for epoch in range(1, 41):
        ae_model.train()
        t_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = ae_model(batch)
            loss = criterion(recon, batch) + 0.1 * torch.mean(torch.abs(recon - batch))
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(batch)
        scheduler.step()
        t_loss /= len(train_ds)

        ae_model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                recon = ae_model(batch)
                loss = criterion(recon, batch)
                v_loss += loss.item() * len(batch)
        v_loss /= len(valid_ds)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(ae_model.state_dict(), str(ae_weights_path))

        if epoch % 10 == 0 or epoch == 1 or epoch == 40:
            print(f'  Epoch {epoch:02d}/40 -> Train Loss: {t_loss:.5f} | Val Loss: {v_loss:.5f} | Best Val: {best_val_loss:.5f}')

    print(f'Autoencoder Training Complete ({time.time()-t0:.1f}s)! Weights saved to {ae_weights_path}')

    # 2. Train YOLO on Anoma
    print('\n[Phase 2/2] Training YOLO11s on Anoma 4-Class Sonar Dataset (25 Epochs)...')
    yolo = YOLO('weights/yolo11s_sih_27class_best.pt')
    results = yolo.train(
        data='samples/anoma/data.yaml',
        epochs=25,
        imgsz=640,
        batch=16,
        workers=0,
        device=0 if torch.cuda.is_available() else 'cpu',
        project='weights',
        name='yolo_anoma_run',
        exist_ok=True,
        amp=False,
        verbose=False
    )

    best_pt = Path('weights/yolo_anoma_run/weights/best.pt')
    final_target = Path('weights/yolo11s_anoma_best.pt')
    if best_pt.exists():
        import shutil
        shutil.copy(str(best_pt), str(final_target))
        print(f'YOLO Training Complete! Saved best model to {final_target} ({final_target.stat().st_size/1024/1024:.2f} MB)')


if __name__ == '__main__':
    main()
