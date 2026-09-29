import cv2

from calibrate import OUT_DIR, image_paths, load_calibration


def main():
    OUT_DIR.mkdir(exist_ok=True)
    calib = load_calibration()
    K, dist = calib["K"], calib["dist"]

    for path in image_paths():
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        # alpha=0 pra cortar a borda preta que sobra depois da correcao
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)
        und = cv2.undistort(img, K, dist, None, new_K)
        cv2.imwrite(str(OUT_DIR / f"sem_distorcao_{path.stem}.png"), und)
        print(path.name)


if __name__ == "__main__":
    main()
