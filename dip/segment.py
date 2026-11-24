"""Segmentation: thresholding, watershed, region growing, k-means."""
import cv2
import numpy as np

from .registry import op
from .utils import Result, to_gray, hist256, num, fit, psnr

AMBER, GRAY, BLUE = "#ffc933", "#d8d8d8", "#4d8dff"


@op("thresholding",
    desc="Five ways to split an image into foreground and background, side by side.",
    formula="basic: 255 if f > T\nadaptive: T is the mean (or Gaussian-weighted mean) of the neighbourhood minus C\nOtsu: T that maximizes the between-class variance",
    params=[
        {"name": "t", "label": "T (basic)", "type": "slider", "min": 0, "max": 255, "step": 1, "default": 128},
        {"name": "t2", "label": "T (global)", "type": "slider", "min": 0, "max": 255, "step": 1, "default": 100},
        {"name": "block", "label": "Block size", "type": "slider", "min": 3, "max": 99, "step": 2, "default": 11},
        {"name": "C", "label": "C", "type": "slider", "min": -20, "max": 30, "step": 1, "default": 2},
    ])
def thresholding(img, p, ctx):
    g = to_gray(fit(img, 800))
    gb = cv2.GaussianBlur(g, (5, 5), 0)
    blk = int(p["block"]) | 1
    otsu_t, ot = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    res = {
        f"Basic T = {p['t']}": cv2.threshold(g, p["t"], 255, cv2.THRESH_BINARY)[1],
        f"Global T = {p['t2']}": cv2.threshold(g, p["t2"], 255, cv2.THRESH_BINARY)[1],
        "Adaptive mean": cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blk, p["C"]),
        "Adaptive Gaussian": cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk, p["C"]),
        f"Otsu T = {int(otsu_t)}": ot,
    }
    r = Result().gallery()
    r.image("Gray input", g, "", "gray")
    rows = []
    for n, m in res.items():
        r.image(n, m, "", n.split()[0].lower())
        nc = cv2.connectedComponents((m > 0).astype(np.uint8), connectivity=8)[0] - 1
        rows.append([n, round(100 * float((m > 0).mean()), 2), nc])
    r.table("Foreground share", rows, columns=["Method", "White %", "White pieces"])
    r.stat("Otsu threshold", int(otsu_t), "found automatically")
    r.chart("Gray histogram with thresholds", [{"name": "Gray", "color": GRAY, "values": hist256(g)}],
            marker=[[p["t"], f"basic {p['t']}"], [p["t2"], f"global {p['t2']}"], [int(otsu_t), f"Otsu {int(otsu_t)}"]])
    r.note("Adaptive methods look at a small window around each pixel, so they cope with uneven lighting.")
    r.source('''
_, basic = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)''')
    return r


@op("watershed",
    desc="Split touching objects. Distance transform finds the cores, then the watershed floods outward until regions meet.",
    formula="sure foreground = distance > k max(distance)\nunknown = sure background - sure foreground",
    params=[
        {"name": "fg", "label": "Foreground k", "type": "slider", "min": 0.1, "max": 0.9, "step": 0.05, "default": 0.5},
        {"name": "inv", "label": "Objects are dark", "type": "checkbox", "default": False},
    ], heavy=True)
def watershed(img, p, ctx):
    c = fit(img, 640)
    g = to_gray(c)
    mode = cv2.THRESH_BINARY if p["inv"] else cv2.THRESH_BINARY_INV
    _, th = cv2.threshold(g, 0, 255, mode + cv2.THRESH_OTSU)
    k = np.ones((3, 3), np.uint8)
    op_ = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=2)
    sure_bg = cv2.dilate(op_, k, iterations=3)
    dist = cv2.distanceTransform(op_, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, p["fg"] * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)
    n, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    mk = cv2.watershed(c.copy(), markers.astype(np.int32))
    out = c.copy()
    out[mk == -1] = (0, 0, 255)
    colors = np.random.default_rng(3).integers(60, 255, (mk.max() + 2, 3)).astype(np.uint8)
    seg = colors[np.clip(mk, 0, None)]
    seg[mk <= 1] = 0
    regions = int(len(np.unique(mk[mk > 1])))
    r = Result().gallery()
    r.big(regions, "regions separated")
    r.image("Original", c, "", "original")
    r.image("1. Threshold (Otsu)", th, "", "threshold")
    r.image("2. After opening", op_, "removes small specks", "opening")
    r.image("3. Sure background", sure_bg, "dilated", "sure_bg")
    r.image("4. Distance transform", cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), "bright = far from background", "distance")
    r.image("5. Sure foreground", sure_fg, "cores of the objects", "sure_fg")
    r.image("6. Unknown zone", unknown, "to be decided by flooding", "unknown")
    r.image("Boundaries in red", out, "", "boundaries")
    r.image("Regions", seg, "each color is one region", "regions")
    r.stat("Regions", regions)
    r.stat("Boundary pixels", f"{int((mk == -1).sum()):,}")
    r.stat("Max distance", num(dist.max(), 2), "pixels")
    r.note("If objects are dark on a light background, tick 'Objects are dark'.")
    r.source('''
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
markers = cv2.connectedComponents(sure_fg)[1] + 1
markers[unknown == 255] = 0
markers = cv2.watershed(img, markers)''')
    return r


@op("region_growing",
    desc="Start from a seed pixel and keep adding neighbours whose brightness is close enough.",
    formula="add neighbour q to the region if |f(q) - f(seed)| <= T",
    params=[
        {"name": "sx", "label": "Seed x %", "type": "slider", "min": 0, "max": 99, "step": 1, "default": 50},
        {"name": "sy", "label": "Seed y %", "type": "slider", "min": 0, "max": 99, "step": 1, "default": 50},
        {"name": "t", "label": "Threshold T", "type": "slider", "min": 1, "max": 100, "step": 1, "default": 20},
        {"name": "conn", "label": "Neighbours", "type": "select", "default": "8", "options": [["4", "4-connected"], ["8", "8-connected"]]},
    ])
def region_growing(img, p, ctx):
    c = fit(img, 640)
    g = to_gray(c)
    H, W = g.shape
    x, y = min(W - 1, int(W * p["sx"] / 100)), min(H - 1, int(H * p["sy"] / 100))
    mask = np.zeros((H + 2, W + 2), np.uint8)
    t = int(p["t"])
    flags = int(p["conn"]) | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
    cv2.floodFill(g.copy(), mask, (x, y), 255, (t,), (t,), flags)
    reg = mask[1:-1, 1:-1] > 0
    over = c.copy()
    over[reg] = (0.55 * over[reg] + 0.45 * np.array([255, 201, 51])).astype(np.uint8)
    cv2.drawMarker(over, (x, y), (0, 0, 255), cv2.MARKER_CROSS, max(16, W // 25), 2)
    r = Result()
    r.image("Seed and grown region", over, f"Seed at ({x}, {y})", "overlay")
    r.image("Region mask", reg, "", "mask")
    r.image("Region cut out", np.where(reg[..., None], c, 0).astype(np.uint8), "", "cutout")
    r.stat("Seed value", int(g[y, x]), f"gray at ({x}, {y})")
    r.stat("Accepted range", f"{max(0, int(g[y, x]) - t)} to {min(255, int(g[y, x]) + t)}")
    r.stat("Region pixels", f"{int(reg.sum()):,}")
    r.stat("Region share", f"{100 * reg.mean():.2f} %")
    r.chart("Gray histogram, region in amber", [{"name": "Whole image", "color": GRAY, "values": hist256(g)},
                                                {"name": "Region", "color": AMBER, "values": hist256(g[reg]) if reg.any() else [0] * 256}],
            marker=[[int(g[y, x]), "seed"]])
    r.note("Move the seed with the sliders. A bigger T grows the region faster and can leak into other objects.")
    r.source('''
mask = np.zeros((h + 2, w + 2), np.uint8)
flags = 8 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
cv2.floodFill(gray, mask, (seed_x, seed_y), 255, (T,), (T,), flags)''')
    return r


@op("kmeans",
    desc="Group pixels into K color clusters. Each cluster becomes one flat region of the segmented picture.",
    formula="minimize sum of ||pixel - cluster centre||^2 over all pixels",
    params=[{"name": "k", "label": "K clusters", "type": "slider", "min": 2, "max": 10, "step": 1, "default": 4}],
    heavy=True)
def kmeans(img, p, ctx):
    c = fit(img, 360)
    k = int(p["k"])
    data = c.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    comp, lab, cen = cv2.kmeans(data, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    cen8 = np.clip(cen, 0, 255).astype(np.uint8)
    lab = lab.reshape(c.shape[:2])
    seg = cen8[lab]
    counts = np.bincount(lab.ravel(), minlength=k)
    order = np.argsort(-counts)
    r = Result().gallery()
    r.image("Original", c, "", "original")
    r.image(f"K-means, K = {k}", seg, "", "kmeans")
    for i in order[:min(k, 4)]:
        r.image(f"Cluster {i}", lab == i, f"{100 * counts[i] / lab.size:.1f} % of pixels", f"cluster_{i}")
    r.palette("Cluster colors", [cen8[i][::-1] for i in order], values=[f"{100 * counts[i] / lab.size:.1f}%" for i in order])
    r.stat("Compactness", f"{comp:,.0f}", "sum of squared distances")
    r.stat("PSNR vs original", f"{psnr(c, seg):.2f} dB")
    sample = data[np.random.default_rng(0).choice(len(data), min(4000, len(data)), replace=False)]
    ks = list(range(1, 9))
    el = [float(cv2.kmeans(sample, kk, None, crit, 2, cv2.KMEANS_PP_CENTERS)[0]) for kk in ks]
    r.chart("Elbow curve (choose K where it bends)", [{"name": "Compactness", "color": AMBER, "values": [round(v) for v in el]}],
            labels=[str(x) for x in ks], x_label="K", y_label="Compactness", kind="line", marker=[[k - 1, f"K = {k}"]])
    r.table("Cluster centres", [[int(i), int(cen8[i][2]), int(cen8[i][1]), int(cen8[i][0]), int(counts[i]), round(100 * counts[i] / lab.size, 2)] for i in order],
            columns=["Cluster", "R", "G", "B", "Pixels", "%"])
    r.source('''
data = img.reshape(-1, 3).astype(np.float32)
_, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
segmented = centers.astype(np.uint8)[labels.flatten()].reshape(img.shape)''')
    return r
