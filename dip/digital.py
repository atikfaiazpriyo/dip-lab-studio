"""Digital image: pixel grid, sampling, quantization, interpolation, indexed image."""
import math

import cv2
import numpy as np

from .registry import op
from .utils import Result, to_gray, hist256, num, fit, mse, psnr, ssim, entropy

RGB = {"R": "#ff4d4d", "G": "#3ddc84", "B": "#4d8dff"}


@op("pixel_grid",
    desc="A digital image is just a matrix of numbers. See the real pixel values of any N x N part of your image.",
    formula="f(x, y) = intensity at row y, column x\nimage = M x N matrix, each value 0 to 255",
    params=[
        {"name": "mode", "label": "Source", "type": "select", "default": "crop",
         "options": [["crop", "Real pixels of a small window"], ["resize", "Whole image shrunk to N x N"]]},
        {"name": "n", "label": "N", "type": "slider", "min": 4, "max": 16, "step": 1, "default": 8},
        {"name": "x", "label": "Window x %", "type": "slider", "min": 0, "max": 95, "step": 1, "default": 40},
        {"name": "y", "label": "Window y %", "type": "slider", "min": 0, "max": 95, "step": 1, "default": 40},
    ])
def pixel_grid(img, p, ctx):
    n = int(p["n"])
    g = to_gray(img)
    H, W = g.shape
    r = Result()
    if p["mode"] == "resize":
        small = cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA)
        cap = f"Whole image averaged down to {n} x {n} pixels"
        x0 = y0 = 0
        box = img.copy()
    else:
        x0 = min(int((W - n) * p["x"] / 100), W - n) if W > n else 0
        y0 = min(int((H - n) * p["y"] / 100), H - n) if H > n else 0
        n_ = min(n, W, H)
        small = img[y0:y0 + n_, x0:x0 + n_]
        cap = f"Real pixels, columns {x0} to {x0 + n_ - 1}, rows {y0} to {y0 + n_ - 1}"
        box = img.copy()
        t = max(2, int(max(H, W) / 250))
        cv2.rectangle(box, (x0, y0), (x0 + n_, y0 + n_), (51, 201, 255), t)
    sg = to_gray(small)
    r.image("Original with window", box, cap if p["mode"] == "crop" else "", "window")
    r.image(f"The {sg.shape[1]} x {sg.shape[0]} digital image", sg, cap, "digital_image")
    r.stat("Matrix size", f"{sg.shape[0]} x {sg.shape[1]}", "rows x columns")
    r.stat("Values in matrix", sg.size)
    r.stat("Smallest value", int(sg.min()))
    r.stat("Largest value", int(sg.max()))
    r.stat("Bits per pixel", 8, "256 gray levels = 2^8")
    r.stat("Storage of this matrix", f"{sg.size} bytes", "1 byte per pixel")
    r.table("Pixel values (gray)", sg.tolist(), heat=(0, 255),
            caption="Each cell is one pixel. Dark cell = small number, bright cell = large number.")
    if sg.shape[0] <= 10:
        for k, name, idx in (("R", "Red", 2), ("G", "Green", 1), ("B", "Blue", 0)):
            r.table(f"{name} channel", small[:, :, idx].tolist(), heat=(0, 255))
    r.note("A color pixel needs three numbers (R, G, B). A gray pixel needs one.")
    r.source('''
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
patch = img[y0:y0+8, x0:x0+8]
print(patch)           # the digital image as a matrix''')
    return r


@op("sampling",
    desc="Sampling decides how many pixels we keep. Fewer samples = smaller file but less detail.",
    formula="g(x, y) = f(k x, k y)   keep every k-th pixel",
    params=[{"name": "k", "label": "Keep every k-th pixel", "type": "slider", "min": 1, "max": 32, "step": 1, "default": 4}])
def sampling(img, p, ctx):
    k = int(p["k"])
    H, W = img.shape[:2]
    s = img[::k, ::k]
    back = cv2.resize(s, (W, H), interpolation=cv2.INTER_NEAREST)
    r = Result()
    r.image("Original", img, f"{W} x {H}", "original")
    r.image("Sampled", s, f"{s.shape[1]} x {s.shape[0]} pixels", "sampled")
    r.image("Sampled, shown at original size", back, "Each kept pixel is stretched to a k x k block", "sampled_big")
    r.stat("Original pixels", f"{W * H:,}")
    r.stat("Kept pixels", f"{s.shape[0] * s.shape[1]:,}")
    r.stat("Data kept", f"{100 * s.shape[0] * s.shape[1] / (W * H):.2f} %")
    r.stat("PSNR vs original", f"{psnr(img, back):.2f} dB", "higher = closer")
    ks = [1, 2, 4, 8, 16, 32]
    ps = []
    for kk in ks:
        b = cv2.resize(img[::kk, ::kk], (W, H), interpolation=cv2.INTER_NEAREST)
        ps.append(round(min(psnr(img, b), 60), 2))
    r.chart("PSNR for different k", [{"name": "PSNR dB", "color": "#ffc933", "values": ps}],
            labels=[f"k={x}" for x in ks], x_label="Sampling step", y_label="dB", kind="bars")
    g6 = cv2.resize(to_gray(img), (6, 6), interpolation=cv2.INTER_AREA)
    r.table("Worked example: 6 x 6 image", g6.tolist(), heat=(0, 255), caption="Your image shrunk to 6 x 6.")
    r.table("Keep every 2nd pixel, now 3 x 3", g6[::2, ::2].tolist(), heat=(0, 255),
            caption="Rows 0, 2, 4 and columns 0, 2, 4 survive.")
    r.source('''
sampled = img[::k, ::k]          # keep every k-th pixel
back = cv2.resize(sampled, (w, h), interpolation=cv2.INTER_NEAREST)''')
    return r


@op("quantization",
    desc="Quantization decides how many gray levels (brightness steps) we keep. Fewer levels = visible bands.",
    formula="levels L = 2^bits\nq(x, y) = floor(f(x, y) / (256 / L)) * (256 / L)",
    params=[
        {"name": "bits", "label": "Bits per pixel", "type": "slider", "min": 1, "max": 8, "step": 1, "default": 3},
        {"name": "mode", "label": "Image", "type": "select", "default": "gray", "options": [["gray", "Grayscale"], ["color", "Color"]]},
    ])
def quantization(img, p, ctx):
    def q(a, bits):
        step = 256 // (2 ** bits)
        return ((a // step) * step).astype(np.uint8)
    src = to_gray(img) if p["mode"] == "gray" else img
    bits = int(p["bits"])
    qi = q(src, bits)
    r = Result()
    r.image("Original", src, "256 levels (8 bits)", "original")
    r.image(f"{bits}-bit image", qi, f"{2 ** bits} levels", "quantized")
    r.stat("Gray levels", 2 ** bits, f"2^{bits}")
    r.stat("Step size", 256 // (2 ** bits))
    r.stat("Unique values found", len(np.unique(qi)))
    r.stat("Size at this depth", f"{src.size * bits / 8 / 1024:,.1f} KB", f"{bits} bits x pixels")
    r.stat("PSNR vs original", f"{psnr(src, qi):.2f} dB")
    g = to_gray(src)
    r.chart("Histogram after quantization", [{"name": "Quantized", "color": "#ffc933", "values": hist256(to_gray(qi))}])
    ps = [round(min(psnr(src, q(src, b)), 60), 2) for b in range(1, 9)]
    r.chart("PSNR for 1 to 8 bits", [{"name": "PSNR dB", "color": "#4d8dff", "values": ps}],
            labels=[f"{b}" for b in range(1, 9)], x_label="Bits", y_label="dB", kind="bars")
    r.table("Same idea with a tiny ramp", [[0, 32, 64, 96, 128, 160, 192, 224, 255],
                                            [int(v) for v in q(np.array([0, 32, 64, 96, 128, 160, 192, 224, 255], np.uint8), bits)]],
            columns=[str(i) for i in range(9)], caption="Row 1 = original values. Row 2 = after quantization.")
    r.note("With very few bits you see false contours: flat bands where smooth shading used to be.")
    r.source('''
levels = 2 ** bits
step = 256 // levels
quantized = (img // step) * step''')
    return r


def _knn_resize(small, W, H, K):
    from scipy.spatial import cKDTree
    h, w = small.shape[:2]
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    tree = cKDTree(np.c_[gx.ravel(), gy.ravel()])
    tx = (np.arange(W) + .5) * w / W - .5
    ty = (np.arange(H) + .5) * h / H - .5
    TX, TY = np.meshgrid(tx, ty)
    K = max(1, min(K, w * h))
    d, idx = tree.query(np.c_[TX.ravel(), TY.ravel()], k=K)
    if K == 1:
        d, idx = d[:, None], idx[:, None]
    wgt = 1.0 / (d + 1e-6)
    wgt /= wgt.sum(1, keepdims=True)
    flat = small.reshape(h * w, -1).astype(np.float64)
    out = (flat[idx] * wgt[..., None]).sum(1)
    return np.clip(out, 0, 255).astype(np.uint8).reshape(H, W, -1).squeeze()


@op("interpolation",
    desc="Shrink the image, then enlarge it again with different methods and see which one rebuilds the original best.",
    formula="nearest: copy the closest pixel\nbilinear: weighted mean of 4 neighbours\nbicubic: cubic fit over 16 neighbours\nKNN: inverse-distance mean of K neighbours",
    params=[
        {"name": "f", "label": "Shrink factor", "type": "slider", "min": 2, "max": 8, "step": 1, "default": 4},
        {"name": "k", "label": "K for KNN", "type": "slider", "min": 1, "max": 25, "step": 1, "default": 4},
    ], heavy=True)
def interpolation(img, p, ctx):
    base = fit(to_gray(img), 480)
    H, W = base.shape
    f = int(p["f"])
    small = cv2.resize(base, (max(2, W // f), max(2, H // f)), interpolation=cv2.INTER_AREA)
    ups = {
        "Nearest": cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST),
        "Bilinear": cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR),
        "Bicubic": cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC),
        "Lanczos": cv2.resize(small, (W, H), interpolation=cv2.INTER_LANCZOS4),
        f"KNN (K={int(p['k'])})": _knn_resize(small, W, H, int(p["k"])),
    }
    r = Result().gallery()
    r.image("Original", base, f"{W} x {H}", "original")
    r.image("Shrunk", small, f"{small.shape[1]} x {small.shape[0]}", "small")
    rows = []
    for name, u in ups.items():
        r.image(name, u, f"PSNR {psnr(base, u):.2f} dB", name.split()[0].lower())
        rows.append([name, round(mse(base, u), 2), round(psnr(base, u), 2), round(ssim(base, u), 4)])
    best = max(rows, key=lambda x: x[2])
    r.stat("Best method", best[0], f"PSNR {best[2]} dB")
    r.table("How close is each result to the original?", rows, columns=["Method", "MSE", "PSNR dB", "SSIM"],
            caption="Lower MSE and higher PSNR / SSIM mean a better rebuild.", highlight=rows.index(best))
    r.chart("PSNR by method", [{"name": "PSNR dB", "color": "#ffc933", "values": [x[2] for x in rows]}],
            labels=[x[0].split()[0] for x in rows], x_label="Method", y_label="dB", kind="bars")
    ex = np.array([[10, 20], [30, 40]], np.uint8)
    r.table("Worked example: 2 x 2 image", ex.tolist(), heat=(0, 60))
    for nm, flag in (("Nearest", cv2.INTER_NEAREST), ("Bilinear", cv2.INTER_LINEAR)):
        r.table(f"{nm}, enlarged to 4 x 4", cv2.resize(ex, (4, 4), interpolation=flag).tolist(), heat=(0, 60))
    r.source('''
small = cv2.resize(img, (w//f, h//f), interpolation=cv2.INTER_AREA)
nearest  = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
bilinear = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
bicubic  = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)''')
    return r


@op("index_image",
    desc="An indexed image stores a small color palette plus one index per pixel instead of three numbers per pixel.",
    formula="pixel = palette[index(x, y)]\nbits per pixel = ceil(log2(colors))",
    params=[{"name": "k", "label": "Palette size", "type": "slider", "min": 2, "max": 64, "step": 1, "default": 8}],
    heavy=True)
def index_image(img, p, ctx):
    k = int(p["k"])
    small = fit(img, 320)
    data = small.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    _, lab, cen = cv2.kmeans(data, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    cen = np.clip(cen, 0, 255).astype(np.uint8)
    lab = lab.reshape(small.shape[:2])
    rec = cen[lab]
    order = np.argsort(-np.bincount(lab.ravel(), minlength=k))
    counts = np.bincount(lab.ravel(), minlength=k)
    bits = max(1, math.ceil(math.log2(k)))
    r = Result()
    r.image("Original", small, "", "original")
    r.image("Index map (gray = index)", (lab * (255 // max(1, k - 1))).astype(np.uint8), "Each pixel stores only a palette number", "index_map")
    r.image(f"Rebuilt from {k} colors", rec, "", "indexed")
    r.stat("Palette colors", k)
    r.stat("Bits per pixel", bits, "indexed")
    r.stat("Full color bits", 24)
    r.stat("Storage saving", f"{100 * (1 - (bits * lab.size + k * 24) / (24 * lab.size)):.1f} %")
    r.stat("PSNR", f"{psnr(small, rec):.2f} dB")
    r.palette("Color map (palette)", [cen[i][::-1] for i in order],
              caption="Sorted by how many pixels use each color.",
              values=[f"#{i}  {100 * counts[i] / lab.size:.1f}%" for i in order])
    h = min(8, lab.shape[0]); w = min(8, lab.shape[1])
    r.table("Index matrix, top-left 8 x 8", lab[:h, :w].tolist(), heat=(0, max(1, k - 1)),
            caption="These numbers are what an indexed image really stores.")
    rows = [[int(i), int(cen[i][2]), int(cen[i][1]), int(cen[i][0])] for i in order[:12]]
    r.table("Palette table (first 12)", rows, columns=["Index", "R", "G", "B"])
    r.source('''
data = img.reshape(-1, 3).astype(np.float32)
_, labels, centers = cv2.kmeans(data, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
index_map = labels.reshape(h, w)       # one number per pixel
palette = centers.astype(np.uint8)     # K x 3 color map''')
    return r
