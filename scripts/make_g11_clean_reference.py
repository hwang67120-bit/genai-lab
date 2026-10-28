"""G11 참조 정리본: 옷 픽셀은 그대로, 모자·마네킹 머리(위쪽 잘라냄)·회색 배경(흰색)·받침대(흰색)만 뺀다. 배경 분리는 GrabCut."""
import numpy as np, cv2
src = "outputs/generalization-20261012/G11-r17/N1/qwen-inputs/garment.jpg"
im = cv2.imread(src)
TOP, BOTTOM, PLATE = 215, 1245, 1175
crop = im[TOP:BOTTOM].copy()
h, w = crop.shape[:2]
bg = np.median(np.concatenate([crop[:4].reshape(-1,3), crop[-4:].reshape(-1,3), crop[:, :8].reshape(-1,3), crop[:, -8:].reshape(-1,3)]), 0)
coarse = (np.abs(crop.astype(int) - bg.astype(int)).max(2) > 45)
ys, xs = np.where(coarse); rect = (int(xs.min()) - 10, 0, int(xs.max() - xs.min()) + 20, h)
mask = np.zeros((h, w), np.uint8)
bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
cv2.grabCut(crop, mask, rect, bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)
background = np.isin(mask, (cv2.GC_BGD, cv2.GC_PR_BGD)).astype(np.uint8)
# 옷 쪽으로 2px 물러서서(가장자리 보호) 흰색으로
background = cv2.erode(background, np.ones((5, 5), np.uint8)).astype(bool)
g = crop.mean(2)
plate = np.zeros_like(background); plate[PLATE - TOP:] = g[PLATE - TOP:] > 70
out = crop.copy(); out[background | plate] = 255
cv2.imwrite("outputs/generalization-20261012/garments/G11-clean.jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 96])
changed = background | plate
print("crop", crop.shape, "changed px", int(changed.sum()), "of", h * w)
side = np.concatenate([cv2.resize(im, (int(im.shape[1]*h/im.shape[0]), h)), crop, out], 1)
cv2.imwrite("outputs/generalization-20261012/rules17-cpu/G11-clean-preview.png", side)
