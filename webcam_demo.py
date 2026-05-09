import torch
import cv2
from model import MattingNetwork

model = MattingNetwork('mobilenetv3').eval()
model.load_state_dict(torch.load('rvm_epoch100.pth', map_location='cpu'))

cap = cv2.VideoCapture(1)
rec = [None] * 4

print("Apasă Q ca să oprești!")

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        src = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        fgr, pha, *rec = model(src, *rec, downsample_ratio=0.25)

        # Alpha matte
        alpha = pha[0, 0].numpy()
        alpha_vis = (alpha * 255).astype('uint8')
        alpha_color = cv2.cvtColor(alpha_vis, cv2.COLOR_GRAY2BGR)

        # Fond verde
        green_bg = frame.copy()
        green_bg[:] = [0, 255, 0]
        mask = alpha[:, :, None]
        composite = (frame * mask + green_bg * (1 - mask)).astype('uint8')

        combined = cv2.hconcat([frame, alpha_color, composite])
        cv2.imshow('Original | Alpha | Composite', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
