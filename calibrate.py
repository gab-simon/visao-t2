import cv2
import numpy as np
from pathlib import Path

IMG_DIR = Path("img")
OUT_DIR = Path("resultados")
CALIB_FILE = Path("calibracao.npz")

PATTERN = (9, 4)  # cantos internos (tabuleiro de 10x5)
SQUARE_SIZE = 1.0  # to usando o lado do quadrado como unidade
FLAGS = cv2.CALIB_FIX_K3  # com pouca foto o k3 da uns valores nada a ver


def image_paths():
    return sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def board_points():
    # cantos do tabuleiro no plano z=0, em unidades de quadrado
    pts = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    pts[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2)
    return pts * SQUARE_SIZE


def find_corners(gray):
    ok, corners = cv2.findChessboardCornersSB(gray, PATTERN, flags=cv2.CALIB_CB_ACCURACY)
    return corners if ok else None


def load_calibration():
    data = np.load(CALIB_FILE)
    return {k: data[k] for k in data.files}


def main():
    obj_pts, img_pts, names = [], [], []
    size = None

    for path in image_paths():
        gray = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        corners = find_corners(gray)
        if corners is None:
            print(f"{path.name}: tabuleiro nao encontrado")
            continue
        obj_pts.append(board_points())
        img_pts.append(corners)
        names.append(path.name)

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None, flags=FLAGS)

    np.set_printoptions(precision=4, suppress=True)
    print(f"imagens usadas: {len(names)}")
    print(f"erro RMS: {rms:.4f} px")
    print("K =\n", K)
    print("dist (k1 k2 p1 p2 k3) =", dist.ravel())

    np.savez(CALIB_FILE, K=K, dist=dist, rvecs=np.array(rvecs), tvecs=np.array(tvecs),
             names=np.array(names), rms=rms)


if __name__ == "__main__":
    main()
