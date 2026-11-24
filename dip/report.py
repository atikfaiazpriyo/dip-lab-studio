"""Pixel connectivity graph and the full image report."""
import itertools

import cv2
import numpy as np

from .registry import op
from .utils import Result, to_gray, hist256, num, fit, entropy, norm8
from .utils import psnr

AMBER, GRAY, BLUE, RED, GREEN = "#ffc933", "#d8d8d8", "#4d8dff", "#ff4d4d", "#3ddc84"
SLIDE = np.array([[10, 20, 30], [15, 25, 35], [20, 30, 40]])


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


@op("pixel_graph",
    desc="Draw an image as a graph: each pixel is a node, and neighbours are joined by edges. Compare 4, 8 and m-adjacency.",
    formula="N4(p): up, down, left, right\nND(p): the 4 diagonals\nm-adjacency: q in N4(p), or q in ND(p) and N4(p) and N4(q) share no pixel of V",
    params=[
        {"name": "src", "label": "Pixels", "type": "select", "default": "slide",
         "options": [["slide", "Lab example 3 x 3"], ["window", "Real window of my image"], ["binary", "My image as 0 and 1"]]},
        {"name": "n", "label": "N", "type": "slider", "min": 3, "max": 10, "step": 1, "default": 3},
        {"name": "adj", "label": "Adjacency", "type": "select", "default": "4", "options": [["4", "4-adjacency"], ["8", "8-adjacency"], ["m", "m-adjacency"]]},
        {"name": "v", "label": "V: values at least", "type": "slider", "min": 0, "max": 255, "step": 1, "default": 0},
        {"name": "x", "label": "Window x %", "type": "slider", "min": 0, "max": 95, "step": 1, "default": 40},
        {"name": "y", "label": "Window y %", "type": "slider", "min": 0, "max": 95, "step": 1, "default": 40},
    ])
def pixel_graph(img, p, ctx):
    n = int(p["n"])
    g = to_gray(img)
    H, W = g.shape
    if p["src"] == "slide":
        a = SLIDE.copy()
        vmin = int(p["v"])
    elif p["src"] == "window":
        n = min(n, H, W)
        x0 = int((W - n) * p["x"] / 100); y0 = int((H - n) * p["y"] / 100)
        a = g[y0:y0 + n, x0:x0 + n].astype(int)
        vmin = int(p["v"])
    else:
        small = cv2.resize(g, (n, n), interpolation=cv2.INTER_AREA)
        _, b = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        a = b.astype(int)
        vmin = 1
    rows, cols = a.shape
    inV = a >= vmin
    nodes = [{"x": j, "y": i, "label": str(int(a[i, j])), "on": bool(inV[i, j])} for i in range(rows) for j in range(cols)]
    idx = lambda i, j: i * cols + j
    n4 = lambda i, j: [(i + di, j + dj) for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)) if 0 <= i + di < rows and 0 <= j + dj < cols]
    edges = []
    def build(mode):
        out = []
        for i in range(rows):
            for j in range(cols):
                if not inV[i, j]:
                    continue
                for (qi, qj) in n4(i, j):
                    if (qi, qj) > (i, j) and inV[qi, qj]:
                        out.append((idx(i, j), idx(qi, qj), "n4"))
                if mode in ("8", "m"):
                    for di, dj in ((1, 1), (1, -1)):
                        qi, qj = i + di, j + dj
                        if 0 <= qi < rows and 0 <= qj < cols and inV[qi, qj]:
                            if mode == "m":
                                common = set(n4(i, j)) & set(n4(qi, qj))
                                if any(inV[ci, cj] for ci, cj in common):
                                    continue
                                out.append((idx(i, j), idx(qi, qj), "m"))
                            else:
                                out.append((idx(i, j), idx(qi, qj), "nd"))
        return out
    edges = build(p["adj"])
    def comps(mode):
        d = DSU(rows * cols)
        for u, v, _ in build(mode):
            d.union(u, v)
        roots = {d.find(idx(i, j)) for i in range(rows) for j in range(cols) if inV[i, j]}
        return len(roots)
    r = Result()
    r.graph(f"Pixel graph, {p['adj']}-adjacency", nodes, [{"a": u, "b": v, "kind": k} for u, v, k in edges],
            caption="White node = pixel belongs to V, dark node = it does not. Number inside = pixel value.")
    r.table("Pixel matrix", a.tolist(), heat=(0, max(1, int(a.max()))), caption=f"V = pixels with value >= {vmin}")
    r.stat("Nodes", rows * cols)
    r.stat("Nodes in V", int(inV.sum()))
    r.stat("Edges", len(edges))
    r.stat("Most neighbours", max([sum(1 for e in edges if i in (e[0], e[1])) for i in range(rows * cols)] or [0]))
    r.table("Connected components", [["4-adjacency", comps("4")], ["8-adjacency", comps("8")], ["m-adjacency", comps("m")]],
            columns=["Adjacency", "Components"], caption="8-adjacency can join regions that 4-adjacency keeps apart. m-adjacency removes the double paths.")
    if p["src"] == "binary":
        r.image("Binary pixels (Otsu)", (a * 255).astype(np.uint8), f"{rows} x {cols}", "binary_small")
    deg = [sum(1 for e in edges if i in (e[0], e[1])) for i in range(rows * cols)]
    r.chart("Neighbours per pixel", [{"name": "Pixels", "color": AMBER, "values": [deg.count(k) for k in range(0, max(deg) + 1)]}],
            labels=[str(k) for k in range(0, max(deg) + 1)], x_label="Number of edges", y_label="Pixels", kind="bars")
    r.source('''
G = nx.Graph()
for i in range(rows):
    for j in range(cols):
        G.add_node((i, j), value=img[i][j])
        if i + 1 < rows: G.add_edge((i, j), (i + 1, j))
        if j + 1 < cols: G.add_edge((i, j), (i, j + 1))''')
    return r


def colorfulness(c):
    b, g, r = [x.astype(np.float64) for x in cv2.split(c)]
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def noise_sigma(g):
    h, w = g.shape
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    s = np.abs(cv2.filter2D(g.astype(np.float64), -1, k)[1:-1, 1:-1]).sum()
    return float(s * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2)))


@op("report",
    desc="One-page health check of your image: brightness, contrast, sharpness, noise, colors, plus which tools to try next.",
    formula="sharpness = variance of Laplacian\nnoise sigma (Immerkaer) = sqrt(pi/2) / (6 (W-2)(H-2)) sum |I * N|")
def report(img, p, ctx):
    c = fit(img, 800)
    g = to_gray(c)
    H0, W0 = img.shape[:2]
    mean, std = float(g.mean()), float(g.std())
    lv = float(cv2.Laplacian(g, cv2.CV_64F).var())
    ns = noise_sigma(g)
    en = entropy(g)
    mich = (float(g.max()) - float(g.min())) / max(1, float(g.max()) + float(g.min()))
    cf = colorfulness(c)
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    ed = 100 * float((cv2.Canny(g, 80, 160) > 0).mean())
    dark, bright = 100 * float((g < 50).mean()), 100 * float((g > 205).mean())
    data = c.reshape(-1, 3).astype(np.float32)
    samp = data[np.random.default_rng(0).choice(len(data), min(6000, len(data)), replace=False)]
    _, lab, cen = cv2.kmeans(samp, 5, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0), 2, cv2.KMEANS_PP_CENTERS)
    cnt = np.bincount(lab.ravel(), minlength=5)
    order = np.argsort(-cnt)
    r = Result()
    r.image("Image", c, f"{W0} x {H0}", "image")
    r.stat("Size", f"{W0} x {H0}", f"{W0 * H0 / 1e6:.2f} megapixels")
    r.stat("File", f"{ctx['nbytes'] / 1024:,.0f} KB", ctx["name"])
    r.stat("Brightness", f"{mean:.1f}", "0 dark, 255 bright")
    r.stat("Contrast (std)", f"{std:.1f}")
    r.stat("Michelson", f"{mich:.3f}")
    r.stat("Sharpness", f"{lv:,.0f}", "variance of Laplacian")
    r.stat("Noise sigma", f"{ns:.2f}")
    r.stat("Entropy", f"{en:.2f} bits", "max 8")
    r.stat("Colorfulness", f"{cf:.1f}")
    r.stat("Mean saturation", f"{hsv[:, :, 1].mean() / 255:.2f}")
    r.stat("Edge pixels", f"{ed:.1f} %", "Canny 80 / 160")
    r.stat("Very dark / very bright", f"{dark:.1f} % / {bright:.1f} %")
    r.palette("Dominant colors", [cen[i][::-1].clip(0, 255).astype(int) for i in order], values=[f"{100 * cnt[i] / cnt.sum():.1f}%" for i in order])
    r.chart("Histogram, R G B", [{"name": n, "color": col, "values": hist256(c[:, :, i])} for n, col, i in (("R", RED, 2), ("G", GREEN, 1), ("B", BLUE, 0))])
    r.chart("Gray histogram", [{"name": "Gray", "color": GRAY, "values": hist256(g)}])
    tips = []
    if mean < 85:
        tips.append("The image is dark. Try Gamma correction (gamma below 1), CLAHE or Histogram equalization.")
    if mean > 175:
        tips.append("The image is bright. Try Gamma correction (gamma above 1) or Contrast and brightness with a negative beta.")
    if std < 40 or mich < 0.6:
        tips.append("Contrast is low. Try Contrast stretching, Histogram equalization or CLAHE.")
    if ns > 4:
        tips.append("There is noticeable noise. Try Noise removal (median or non-local means) or Smoothing filters.")
    if lv < 100:
        tips.append("The image looks soft or blurry. Try Sharpening (unsharp mask).")
    if ed > 12:
        tips.append("Lots of edges and texture. Feature extraction tools (Harris, HOG, LBP, GLCM) will give rich results.")
    if cf < 15:
        tips.append("The image is nearly gray. Color space tools will show little difference between channels.")
    if not tips:
        tips.append("The image is well exposed, sharp and low in noise. Any tool will give a clear result.")
    for t in tips:
        r.note(t)
    r.table("What to open next", [["Segmentation", "Thresholding, K-means, Watershed"], ["Edges", "Sobel, Canny"], ["Frequency", "FFT and filters"],
                                  ["Counting", "Count people and objects (YOLO), Contours"]], columns=["Topic", "Tools"])
    r.source('''
sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
contrast = gray.std()
brightness = gray.mean()''')
    return r
