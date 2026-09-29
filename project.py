import cv2
import numpy as np
from pathlib import Path

from calibrate import IMG_DIR, OUT_DIR, PATTERN, SQUARE_SIZE, load_calibration

W, H = PATTERN

# pontos 3D conhecidos no referencial do tabuleiro (z<0 sai do tabuleiro)
AXES = np.float32([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, -3]]) * SQUARE_SIZE
OUTER = np.float32([[-1, -1, 0], [W, -1, 0], [W, H, 0], [-1, H, 0]]) * SQUARE_SIZE


def draw(img, pts):
    out = img.copy()
    o, x, y, z = [tuple(p.astype(int)) for p in pts[:4]]
    cv2.line(out, o, x, (0, 0, 255), 3)  # eixo x
    cv2.line(out, o, y, (0, 255, 0), 3)  # eixo y
    cv2.line(out, o, z, (255, 0, 0), 3)  # eixo z
    for p in pts[4:]:
        cv2.circle(out, tuple(p.astype(int)), 9, (255, 0, 255), 3)
    return out


def main():
    OUT_DIR.mkdir(exist_ok=True)
    calib = load_calibration()
    K, dist = calib["K"], calib["dist"]
    obj = np.vstack([AXES, OUTER])

    for name, rvec, tvec in zip(calib["names"], calib["rvecs"], calib["tvecs"]):
        pts, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        pts = pts.reshape(-1, 2)
        print(name, pts.round(1).tolist())

        img = cv2.imread(str(IMG_DIR / str(name)))
        cv2.imwrite(str(OUT_DIR / f"pontos_{Path(name).stem}.png"), draw(img, pts))


if __name__ == "__main__":
    main()
