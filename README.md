# visao-t2

TA02 — calibração de câmera com OpenCV, usando o tabuleiro do VRI (10×5 quadrados, 9×4 cantos internos) e 7 fotos de celular em `img/`.

Rodar com `uv run <script>.py`, na ordem abaixo.

## Arquivos

- **`calibrate.py`** — detecta os cantos do tabuleiro nas fotos, roda `calibrateCamera` e salva a matriz intrínseca `K`, os coeficientes de distorção e as poses em `calibracao.npz`.

- **`undistort.py`** — remove a distorção das fotos (`resultados/sem_distorcao_*.png`).

- **`project.py`** — projeta pontos 3D conhecidos (eixos e cantos externos do quadriculado) em cada foto com `cv2.projectPoints` e desenha por cima, pra conferir se caem no lugar certo.
