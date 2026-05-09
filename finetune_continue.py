import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os, glob, csv, cv2
from model import MattingNetwork

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
                if not ret: break
                frames.append(frame)
            cap.release()
            for i, m in enumerate(mattes):
                if i < len(frames):
                    self.pairs.append((frames[i], m))
        print(f"Dataset: {len(self.pairs)} perechi")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        frame, matte_path = self.pairs[idx]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_t = self.transform(Image.fromarray(frame_rgb))
        matte_t = self.transform(Image.open(matte_path).convert('L'))
        return frame_t, matte_t


def finetune_continue():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    # Incarca modelul de la epoch 5
    model = MattingNetwork('mobilenetv3').to(device)
    model.load_state_dict(torch.load('rvm_finetuned.pth', map_location=device))

    # Freeze all, unfreeze last 2
    for param in model.parameters():
        param.requires_grad = False
    for layer in list(model.decoder.children())[-2:]:
        for param in layer.parameters():
            param.requires_grad = True

    dataset = VideoMattingDataset('data/clips', 'data/mattes')
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5
    )
    loss_fn = nn.L1Loss()

    # Continua din epoch 6 pana la 100
    START_EPOCH = 6
    END_EPOCH   = 100

    csv_path = 'training_log.csv'

    for epoch in range(START_EPOCH, END_EPOCH + 1):
        model.train()
        losses = []
        rec = [None] * 4

        for frames, mattes in loader:
            frames = frames.to(device)
            mattes = mattes.to(device)
            optimizer.zero_grad()
            src = frames.unsqueeze(0)
            fgr, pha, *rec = model(src, *rec, downsample_ratio=0.25)
            rec = [r.detach() if r is not None else None for r in rec]
            loss = loss_fn(pha[0], mattes.unsqueeze(1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg = sum(losses) / len(losses)
        mn  = min(losses)
        mx  = max(losses)

        print(f"Epoch {epoch}/{END_EPOCH} — avg: {avg:.4f}  min: {mn:.4f}  max: {mx:.4f}")

        # Salveaza in CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, round(avg,4), round(mn,4), round(mx,4)])

        # Salveaza modelul la fiecare 10 epoci
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'rvm_epoch{epoch}.pth')
            print(f"  → salvat: rvm_epoch{epoch}.pth")

    torch.save(model.state_dict(), 'rvm_epoch100.pth')
    print("Done! rvm_epoch100.pth salvat")

if __name__ == '__main__':
    finetune_continue()
