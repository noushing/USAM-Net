

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

print(f"PyTorch version: {torch.__version__}")
print(f"GPU available:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name:        {torch.cuda.get_device_name(0)}")

# ── Parameters ───────────────────────────────────────────────
# Dataset root — must have train/images/ and train/masks/
DATASET_ROOT     = '/content/raw_images/'

# Synthetic output folder — separate from real data
SYNTHETIC_FOLDER = '/content/synthetic_data'
OUTPUT_IMAGES    = os.path.join(SYNTHETIC_FOLDER, 'images')
OUTPUT_MASKS     = os.path.join(SYNTHETIC_FOLDER, 'masks')

os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_MASKS,  exist_ok=True)

# Training settings
IMAGE_SIZE   = 256    # must match U-SAMNet input size
BATCH_SIZE   = 4      # reduce to 2 if out of memory
N_EPOCHS     = 150    # 150 is good for ~1,147 training images
LR           = 2e-4   # learning rate
LAMBDA_L1    = 100    # L1 loss weight
N_SYNTHETIC  = 6000   # synthetic pairs to generate

# GAN trained ONLY on train split
# Val and test are never touched
TRAIN_IMAGES = os.path.join(DATASET_ROOT, 'train', 'images')
TRAIN_MASKS  = os.path.join(DATASET_ROOT, 'train', 'masks')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice:          {DEVICE}")
print(f"Image size:      {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Epochs:          {N_EPOCHS}")
print(f"Synthetic pairs: {N_SYNTHETIC}")
print(f"Train images:    {TRAIN_IMAGES}")
print(f"Output:          {SYNTHETIC_FOLDER}")

# ============================================================
# LOAD TRAIN PAIRS ONLY
# Val and test are strictly excluded from GAN training
# ============================================================

def load_train_pairs():
    """
    Loads image-mask pairs from train split ONLY.
    Val and test sets are never included.
    Pairs matched by identical filename.
    Example: train/images/L17.png <-> train/masks/L17.png
    """
    pairs   = []
    missing = []

    if not os.path.exists(TRAIN_IMAGES):
        raise FileNotFoundError(
            f"Train images folder not found: {TRAIN_IMAGES}")

    for fname in sorted(os.listdir(TRAIN_IMAGES)):
        if not fname.lower().endswith(
                ('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            continue

        img_path  = os.path.join(TRAIN_IMAGES, fname)
        mask_path = os.path.join(TRAIN_MASKS,  fname)

        if not os.path.exists(mask_path):
            missing.append(fname)
            continue

        pairs.append((img_path, mask_path))

    print(f"\nTrain pairs loaded: {len(pairs)}")
    if missing:
        print(f"Missing masks ({len(missing)}): {missing[:5]}...")
    print("Val and test sets excluded from GAN training.")
    return pairs


# ============================================================
# DATASET CLASS
# ============================================================

class PoreDataset(Dataset):
    """
    Loads image-mask pairs for pix2pix training.

    Generator learns:
      INPUT  -> mask  (binary, shows pore locations)
      TARGET -> image (real grayscale BSE plate image)

    Both normalized to [-1, 1] as required by GAN training.
    """
    def __init__(self, pairs, augment=True):
        self.pairs   = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img  = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img  = cv2.resize(img,  (IMAGE_SIZE, IMAGE_SIZE))
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))

        # Keep mask strictly binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # To float tensors [1, H, W] normalized to [-1, 1]
        img  = torch.tensor(img,  dtype=torch.float32
                            ).unsqueeze(0) / 127.5 - 1.0
        mask = torch.tensor(mask, dtype=torch.float32
                            ).unsqueeze(0) / 127.5 - 1.0

        # Same augmentation applied to both image and mask
        if self.augment:
            if torch.rand(1) > 0.5:
                img  = TF.hflip(img)
                mask = TF.hflip(mask)
            if torch.rand(1) > 0.5:
                img  = TF.vflip(img)
                mask = TF.vflip(mask)
            k    = torch.randint(0, 4, (1,)).item()
            img  = torch.rot90(img,  k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[1, 2])

        # Return: (input to generator, target output)
        return mask, img


# ============================================================
# GENERATOR — U-Net
# ============================================================

class UNetDown(nn.Module):
    """Encoder block: Conv -> BN -> LeakyReLU"""
    def __init__(self, in_ch, out_ch, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetUp(nn.Module):
    """Decoder block: ConvTranspose -> BN -> Dropout? -> ReLU"""
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.block(x)
        return torch.cat([x, skip], dim=1)


class Generator(nn.Module):
    """
    U-Net Generator.
    Input:  mask  [1, 256, 256]
    Output: image [1, 256, 256] (Tanh -> range [-1, 1])
    """
    def __init__(self):
        super().__init__()
        self.e1 = UNetDown(1,   64,  use_bn=False)
        self.e2 = UNetDown(64,  128)
        self.e3 = UNetDown(128, 256)
        self.e4 = UNetDown(256, 512)
        self.e5 = UNetDown(512, 512)
        self.e6 = UNetDown(512, 512)
        self.e7 = UNetDown(512, 512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.ReLU(inplace=True)
        )

        self.d1 = UNetUp(512,     512, dropout=True)
        self.d2 = UNetUp(512+512, 512, dropout=True)
        self.d3 = UNetUp(512+512, 512, dropout=True)
        self.d4 = UNetUp(512+512, 512)
        self.d5 = UNetUp(512+512, 256)
        self.d6 = UNetUp(256+256, 128)
        self.d7 = UNetUp(128+128,  64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(64+64, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        b  = self.bottleneck(e7)
        d1 = self.d1(b,  e7)
        d2 = self.d2(d1, e6)
        d3 = self.d3(d2, e5)
        d4 = self.d4(d3, e4)
        d5 = self.d5(d4, e3)
        d6 = self.d6(d5, e2)
        d7 = self.d7(d6, e1)
        return self.final(d7)


# ============================================================
# DISCRIMINATOR — PatchGAN
# ============================================================

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator.
    Input:  mask + image concatenated [2, 256, 256]
    Output: patch score map (real=1, fake=0)
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2,   64,  4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,  128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1,   4, 1, 1),
        )

    def forward(self, mask, image):
        return self.model(torch.cat([mask, image], dim=1))


# ============================================================
# PREVIEW REAL DATA
# ============================================================

def preview_real_data(pairs, n=8):
    """Show n real image+mask pairs to confirm loading is correct."""
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 5))
    fig.suptitle('Real training data — image and mask pairs',
                 fontsize=11)

    indices = np.random.choice(len(pairs), size=n, replace=False)

    for col, idx in enumerate(indices):
        img_path, mask_path = pairs[idx]
        img  = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img  = cv2.resize(img,  (IMAGE_SIZE, IMAGE_SIZE))
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        axes[0, col].imshow(img,  cmap='gray', vmin=0, vmax=255)
        axes[0, col].set_title(
            os.path.basename(img_path)[:10], fontsize=7)
        axes[0, col].axis('off')
        axes[1, col].imshow(mask, cmap='gray', vmin=0, vmax=255)
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel('Image', fontsize=9)
    axes[1, 0].set_ylabel('Mask',  fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(SYNTHETIC_FOLDER,
                             'real_data_preview.png'),
                dpi=120, bbox_inches='tight')
    plt.show()
    print("Real data preview saved.")


# ============================================================
# TRAINING PREVIEW
# ============================================================

def quick_preview(G, pairs, epoch, n=5):
    """Show a 5-image preview during training to monitor quality."""
    G.eval()
    fig, axes = plt.subplots(3, n, figsize=(n * 2.5, 7))
    fig.suptitle(f'Training preview — Epoch {epoch}', fontsize=11)

    dataset = PoreDataset(pairs[:n], augment=False)

    with torch.no_grad():
        for col in range(n):
            mask_t, img_t = dataset[col]
            fake = G(mask_t.unsqueeze(0).to(DEVICE)
                     ).squeeze().cpu().numpy()

            real_img  = ((img_t.squeeze().numpy()
                          + 1) * 127.5).clip(0, 255).astype(np.uint8)
            real_mask = ((mask_t.squeeze().numpy()
                          + 1) * 127.5).clip(0, 255).astype(np.uint8)
            fake_img  = ((fake
                          + 1) * 127.5).clip(0, 255).astype(np.uint8)

            axes[0, col].imshow(real_mask, cmap='gray')
            axes[0, col].axis('off')
            axes[1, col].imshow(real_img,  cmap='gray')
            axes[1, col].axis('off')
            axes[2, col].imshow(fake_img,  cmap='gray')
            axes[2, col].axis('off')

    axes[0, 0].set_ylabel('Input mask',  fontsize=8)
    axes[1, 0].set_ylabel('Real image',  fontsize=8)
    axes[2, 0].set_ylabel('GAN output',  fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(SYNTHETIC_FOLDER,
                             f'preview_epoch_{epoch}.png'),
                dpi=100, bbox_inches='tight')
    plt.show()
    G.train()


# ============================================================
# TRAINING
# ============================================================

def train_pix2pix(pairs):
    """
    Trains pix2pix GAN on train split only.
    Val and test sets are never used here.
    """
    dataset    = PoreDataset(pairs, augment=True)
    dataloader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = 2,
        pin_memory  = True
    )

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)

    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1  = nn.L1Loss()

    loss_G_log = []
    loss_D_log = []

    print("="*55)
    print(f"  Training pix2pix on TRAIN SPLIT ONLY")
    print(f"  Train pairs : {len(pairs)}")
    print(f"  Epochs      : {N_EPOCHS}")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  Device      : {DEVICE}")
    print(f"  Val/test    : excluded")
    print("="*55)

    for epoch in range(1, N_EPOCHS + 1):
        G.train()
        D.train()
        running_G = 0.0
        running_D = 0.0

        for mask, real_img in dataloader:
            mask     = mask.to(DEVICE)
            real_img = real_img.to(DEVICE)

            # Train Discriminator
            opt_D.zero_grad()
            fake_img  = G(mask).detach()
            pred_real = D(mask, real_img)
            pred_fake = D(mask, fake_img)
            loss_D = (
                criterion_GAN(pred_real,
                              torch.ones_like(pred_real)) +
                criterion_GAN(pred_fake,
                              torch.zeros_like(pred_fake))
            ) * 0.5
            loss_D.backward()
            opt_D.step()

            # Train Generator
            opt_G.zero_grad()
            fake_img  = G(mask)
            pred_fake = D(mask, fake_img)
            loss_G_adv = criterion_GAN(
                pred_fake, torch.ones_like(pred_fake))
            loss_G_l1  = criterion_L1(
                fake_img, real_img) * LAMBDA_L1
            loss_G     = loss_G_adv + loss_G_l1
            loss_G.backward()
            opt_G.step()

            running_G += loss_G.item()
            running_D += loss_D.item()

        avg_G = running_G / len(dataloader)
        avg_D = running_D / len(dataloader)
        loss_G_log.append(avg_G)
        loss_D_log.append(avg_D)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{N_EPOCHS}]  "
                  f"G_loss: {avg_G:.4f}   D_loss: {avg_D:.4f}")

        if epoch % 50 == 0:
            quick_preview(G, pairs, epoch)

    print("\nTraining complete.")

    # Save trained generator
    torch.save(G.state_dict(),
               os.path.join(SYNTHETIC_FOLDER, 'generator.pth'))
    print(f"Generator saved to {SYNTHETIC_FOLDER}/generator.pth")

    return G, loss_G_log, loss_D_log


# ============================================================
# PLOT TRAINING LOSS
# ============================================================

def plot_loss(loss_G_log, loss_D_log):
    plt.figure(figsize=(10, 4))
    plt.plot(loss_G_log, label='Generator loss',
             color='steelblue', linewidth=1.5)
    plt.plot(loss_D_log, label='Discriminator loss',
             color='tomato', linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Pix2Pix Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SYNTHETIC_FOLDER, 'training_loss.png'),
                dpi=120, bbox_inches='tight')
    plt.show()
    print("Loss curve saved.")


# ============================================================
# GENERATE SYNTHETIC PAIRS
# ============================================================

def generate_synthetic_pairs(G, pairs, n=6000):
    """
    Generates n synthetic image-mask pairs.
    Uses real masks from train set as input to generator.
    Generator produces synthetic images matching those masks.
    Masks are guaranteed correct — only the image is synthetic.

    Synthetic pairs saved to SYNTHETIC_FOLDER (separate from
    real train/val/test data).
    """
    G.eval()
    dataset = PoreDataset(pairs, augment=False)

    # Sample with replacement if n > dataset size
    replace  = n > len(dataset)
    indices  = np.random.choice(len(dataset), size=n,
                                replace=replace)

    gen_images = []
    gen_masks  = []

    print(f"\nGenerating {n} synthetic pairs...")
    print(f"Source: train split only")
    print(f"Output: {SYNTHETIC_FOLDER}")

    with torch.no_grad():
        for i, idx in enumerate(indices):
            mask_tensor, _ = dataset[idx]

            fake = G(mask_tensor.unsqueeze(0).to(DEVICE))
            fake = fake.squeeze().cpu().numpy()

            syn_img  = ((fake + 1.0) * 127.5
                        ).clip(0, 255).astype(np.uint8)
            syn_mask = ((mask_tensor.squeeze().numpy() + 1.0
                         ) * 127.5).clip(0, 255).astype(np.uint8)
            _, syn_mask = cv2.threshold(
                syn_mask, 127, 255, cv2.THRESH_BINARY)

            fname = f"syn_{i+1:05d}.png"
            cv2.imwrite(os.path.join(OUTPUT_IMAGES, fname), syn_img)
            cv2.imwrite(os.path.join(OUTPUT_MASKS,  fname), syn_mask)

            gen_images.append(syn_img)
            gen_masks.append(syn_mask)

            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{n} done")

    print(f"\n{n} synthetic pairs saved.")
    print(f"  Images -> {OUTPUT_IMAGES}")
    print(f"  Masks  -> {OUTPUT_MASKS}")
    return gen_images, gen_masks


# ============================================================
# FINAL VISUALIZATION
# ============================================================

def visualize_final(pairs, gen_images, gen_masks, n_show=10):
    dataset = PoreDataset(pairs[:n_show], augment=False)

    fig = plt.figure(figsize=(n_show * 2.2, 10))
    fig.suptitle(
        'Pix2Pix Results — '
        'Real image | Real mask | Synthetic image | Synthetic mask',
        fontsize=10, y=1.02
    )

    gs = gridspec.GridSpec(4, n_show, hspace=0.04, wspace=0.04)
    row_labels = ['Real\nimage', 'Real\nmask',
                  'Synthetic\nimage', 'Synthetic\nmask']

    for col in range(n_show):
        mask_t, img_t = dataset[col]
        real_img  = ((img_t.squeeze().numpy()
                      + 1) * 127.5).clip(0, 255).astype(np.uint8)
        real_mask = ((mask_t.squeeze().numpy()
                      + 1) * 127.5).clip(0, 255).astype(np.uint8)
        rows_data = [real_img, real_mask,
                     gen_images[col], gen_masks[col]]

        for row in range(4):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(rows_data[row], cmap='gray', vmin=0, vmax=255)
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=8,
                              rotation=0, labelpad=55, va='center')
            if row == 0:
                ax.set_title(f'#{col+1}', fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(SYNTHETIC_FOLDER, 'final_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Final visualization saved: {save_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Step 1 — Load train pairs only
    train_pairs = load_train_pairs()

    # Step 2 — Preview real data
    preview_real_data(train_pairs)

    # Step 3 — Train GAN on train split only
    G, loss_G_log, loss_D_log = train_pix2pix(train_pairs)

    # Step 4 — Plot training loss
    plot_loss(loss_G_log, loss_D_log)

    # Step 5 — Generate 6,000 synthetic pairs
    gen_images, gen_masks = generate_synthetic_pairs(
        G, train_pairs, n=N_SYNTHETIC)

    # Step 6 — Final visualization
    visualize_final(train_pairs, gen_images, gen_masks, n_show=10)

    print("\nDone. Next step: add synthetic data to train folder")
    print(f"  Copy {SYNTHETIC_FOLDER}/images/ -> train/images/")
    print(f"  Copy {SYNTHETIC_FOLDER}/masks/  -> train/masks/")