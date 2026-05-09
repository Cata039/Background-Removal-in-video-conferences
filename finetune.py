import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import argparse
import os
import glob
import re
import csv
import cv2

SAVE_EPOCHS = {1, 2, 3, 4, 5, 10, 20, 30, 40, 50}


class VideoMattingDataset(Dataset):
    def __init__(self, clips_dir, mattes_dir):
        self.pairs = []
        self.transform = transforms.ToTensor()

        for clip_name in sorted(os.listdir(mattes_dir)):
            matte_folder = os.path.join(mattes_dir, clip_name)
            clip_path = os.path.join(clips_dir, clip_name + '.mp4')
            if not os.path.isdir(matte_folder) or not os.path.isfile(clip_path):
                continue
            mattes = sorted(glob.glob(os.path.join(matte_folder, '*.png')))

            cap = cv2.VideoCapture(clip_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            for i, m in enumerate(mattes):
                if i < len(frames):
                    self.pairs.append((frames[i], m))

        print(f"Dataset: {len(self.pairs)} perechi (frame, matte)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        frame, matte_path = self.pairs[idx]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_t = self.transform(Image.fromarray(frame_rgb))
        matte_t = self.transform(Image.open(matte_path).convert('L'))
        return frame_t, matte_t


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune RVM and save fixed epochs for comparison.")
    parser.add_argument("--run-name", type=str, default="attempt2", help="Prefix for saved checkpoints/logs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for trainable parameters.")
    parser.add_argument("--epochs", type=int, default=50, help="Total epochs to run (will auto-resume).")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--downsample-ratio", type=float, default=0.25, help="RVM downsample ratio for training.")
    return parser.parse_args()


def finetune():
    from model import MattingNetwork

    args = parse_args()
    run_name = args.run_name

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")
    print(f"Run: {run_name} | LR: {args.lr} | Epochs: {args.epochs} | Batch: {args.batch_size}")

    model = MattingNetwork('mobilenetv3').to(device)
    model.load_state_dict(torch.load('rvm_mobilenetv3.pth', map_location=device), strict=True)

    # Freeze all
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze decoder + matting head (more adaptation capacity than last 2 layers)
    trainable_layers = [model.decoder, model.project_mat]
    for layer in trainable_layers:
        for param in layer.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} parametri")

    dataset = VideoMattingDataset(
        clips_dir='data/clips',
        mattes_dir='data/mattes'
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset gol. Verifica folderele data/clips si data/mattes.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    l1_loss = nn.L1Loss()
    bce_loss = nn.BCELoss()

    # CSV pentru grafic
    csv_path = f"training_log_{run_name}.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'avg_loss', 'min_loss', 'max_loss'])
    else:
        # Dedupe in case training was resumed and epochs were appended twice.
        # Keep the last occurrence per epoch (usually the most recent run).
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader]
        by_epoch = {}
        for row in rows:
            try:
                e = int(row["epoch"])
            except Exception:
                continue
            by_epoch[e] = row
        if by_epoch:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "avg_loss", "min_loss", "max_loss"])
                for e in sorted(by_epoch.keys()):
                    r = by_epoch[e]
                    writer.writerow([e, r.get("avg_loss", ""), r.get("min_loss", ""), r.get("max_loss", "")])

    # Auto-resume: if attempt2_epochXX.pth exists, continue from the latest.
    latest_epoch = 0
    epoch_re = re.compile(rf"^{re.escape(run_name)}_epoch(\d+)\.pth$")
    for path in glob.glob(f"{run_name}_epoch*.pth"):
        name = os.path.basename(path)
        match = epoch_re.match(name)
        if match:
            latest_epoch = max(latest_epoch, int(match.group(1)))

    if latest_epoch > 0:
        ckpt_path = f"{run_name}_epoch{latest_epoch}.pth"
        if os.path.exists(ckpt_path):
            print(f"Resuming from: {ckpt_path} (epoch {latest_epoch})")
            model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
        else:
            print(f"Found epoch {latest_epoch} but missing file: {ckpt_path}. Starting from scratch.")
            latest_epoch = 0

    # Read already-logged epochs to prevent duplicates when resuming.
    logged_epochs = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    logged_epochs.add(int(row["epoch"]))
                except Exception:
                    pass

    num_epochs = args.epochs
    for epoch in range(latest_epoch, num_epochs):
        model.train()
        losses = []

        for frames, mattes in loader:
            frames = frames.to(device)
            mattes = mattes.to(device)

            optimizer.zero_grad()

            # Model expects [B, C, H, W] for independent images.
            _, pha = model(frames, downsample_ratio=args.downsample_ratio)[:2]

            # Match prediction/target shapes robustly.
            if pha.shape != mattes.shape:
                mattes = TF.resize(
                    mattes,
                    size=[pha.shape[-2], pha.shape[-1]],
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                )

            mattes = mattes.clamp(0.0, 1.0)
            pha = pha.clamp(0.0, 1.0)

            # Combined objective gives sharper mattes than pure L1.
            loss = l1_loss(pha, mattes) + 0.5 * bce_loss(pha, mattes)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        avg = sum(losses) / len(losses)
        mn  = min(losses)
        mx  = max(losses)

        print(f"Epoch {epoch+1}/{num_epochs} — avg: {avg:.4f}  min: {mn:.4f}  max: {mx:.4f}")

        # Salvează în CSV
        current_epoch = epoch + 1
        if current_epoch not in logged_epochs:
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([current_epoch, round(avg,4), round(mn,4), round(mx,4)])
            logged_epochs.add(current_epoch)

        # Salveaza doar epocile experimentului cerute.
        if current_epoch in SAVE_EPOCHS:
            checkpoint_name = f"{run_name}_epoch{current_epoch}.pth"
            torch.save(model.state_dict(), checkpoint_name)
            print(f"  → salvat: {checkpoint_name}")

    final_name = f"{run_name}_finetuned.pth"
    torch.save(model.state_dict(), final_name)
    print(f"Done! Model final: {final_name}")
    print(f"Log complet: {csv_path}")

if __name__ == '__main__':
    finetune()
