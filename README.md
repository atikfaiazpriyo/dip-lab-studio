# DIP Lab Studio (CSE 4246)

Upload one image, then open any of the 56 tools. Every tool shows pictures, graphs,
numbers and matrices, with the lab formula and lab code underneath.

## Run

    pip install -r requirements.txt
    pip install -r requirements-extra.txt     (only for YOLO counting and OCR)
    python app.py

The browser opens at http://127.0.0.1:5000

`yolo11n.pt` is already in this folder (next to app.py). YOLO tools need `ultralytics`;
if the page says it is missing, install it in the same Python you use to run `python app.py`.

## Tools

Basics, Digital image (pixel grid, sampling, quantization, interpolation, indexed image),
Noise and color (noise models, color spaces, HSI both ways, pixel calculator),
Enhancement (contrast, gamma, histogram, stretching, equalization, CLAHE),
Histogram lab (equalization step by step, histogram matching), Transform and filters,
Edge detection (Sobel, Prewitt, Roberts, Laplacian, Canny), Segmentation (thresholding,
watershed, region growing, K-means), Morphology, Frequency and compression (FFT, ideal /
Butterworth / Gaussian filters, JPEG, DCT), Feature extraction (Harris, HOG, LBP, GLCM,
SIFT / ORB / FAST, zoom into points), Detection (YOLO people count, contours, Hough,
template matching, faces, OCR), Pixel connectivity graph, Full image report.

## Add your own tool

Create a function in any file inside `dip/`, decorate it with `@op("id", ...)`, add the id
to ROADMAP in `dip/registry.py`, and import the file in `dip/__init__.py`.

## Phone

Open the site on a phone and tap "Tools" for the menu. "Take photo" opens the camera.
Touch an image or a chart to read a pixel or a value. In Chrome or Safari use
"Add to Home Screen" to install it like an app.
On the same Wi-Fi you can also open `http://<pc-ip>:5000` while `python app.py` runs.

## Deploy on Render

1. Push this folder to a GitHub repository (yolo11n.pt is only 5.6 MB, so it can go in).
2. On render.com: New > Blueprint > pick the repo. `render.yaml` sets everything up.
   (Or New > Web Service with Build `pip install -r requirements.txt` and
   Start `gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`.)
3. Wait for the build, then open the `https://dip-lab-studio.onrender.com` style link.

Notes:
- Keep `--workers 1`. Uploaded images are kept in memory, so more workers would lose them.
- Free plan sleeps after 15 minutes idle. The first visit afterwards takes about a minute.
- MAX_SIDE=1600 (set in render.yaml) shrinks big phone photos so tools stay fast.
- YOLO counting and OCR need `requirements-extra.txt` (PyTorch, over 1 GB). They will
  show a "needs package" message on the free plan. Everything else works.
