"""Morphology."""
import cv2
import numpy as np

from .registry import op
from .utils import Result, to_gray, num, fit

SHAPES = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}


def binarize(img, auto_minor=True):
    g = to_gray(fit(img, 640))
    _, b = cv2.threshold(cv2.GaussianBlur(g, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if auto_minor and (b > 0).mean() > 0.5:
        b = 255 - b            # make the smaller part the white object
    return g, b


@op("morphology",
    desc="Grow or shrink shapes with a small structuring element: erosion, dilation, opening, closing and friends.",
    formula="erosion: pixel stays white only if the whole kernel fits inside\ndilation: pixel becomes white if the kernel touches any white\nopening = erosion then dilation\nclosing = dilation then erosion",
    params=[
        {"name": "src", "label": "Input", "type": "select", "default": "binary", "options": [["binary", "Binary (Otsu)"], ["gray", "Grayscale"]]},
        {"name": "shape", "label": "Kernel", "type": "select", "default": "rect", "options": [["rect", "Square"], ["ellipse", "Ellipse"], ["cross", "Cross"]]},
        {"name": "size", "label": "Size", "type": "slider", "min": 3, "max": 31, "step": 2, "default": 5},
        {"name": "it", "label": "Iterations", "type": "slider", "min": 1, "max": 5, "step": 1, "default": 1},
    ])
def morphology(img, p, ctx):
    g, b = binarize(img)
    src = b if p["src"] == "binary" else g
    k = cv2.getStructuringElement(SHAPES[p["shape"]], (int(p["size"]), int(p["size"])))
    it = int(p["it"])
    ops = [("Erosion", cv2.MORPH_ERODE), ("Dilation", cv2.MORPH_DILATE), ("Opening", cv2.MORPH_OPEN), ("Closing", cv2.MORPH_CLOSE),
           ("Gradient", cv2.MORPH_GRADIENT), ("Top hat", cv2.MORPH_TOPHAT), ("Black hat", cv2.MORPH_BLACKHAT)]
    r = Result().gallery()
    r.image("Input", src, "", "input")
    rows = []
    for n, f in ops:
        o = cv2.morphologyEx(src, f, k, iterations=it)
        r.image(n, o, "", n.lower().replace(" ", "_"))
        rows.append([n, round(100 * float((o > 127).mean()), 2), round(float(o.mean()), 1)])
    r.table("White pixels after each operation", [["Input", round(100 * float((src > 127).mean()), 2), round(float(src.mean()), 1)]] + rows,
            columns=["Operation", "White %", "Mean"])
    r.table("Structuring element", k.tolist(), heat=(0, 1), caption=f"{p['shape']} {p['size']} x {p['size']}, 1 = part of the kernel.")
    r.note("Opening removes small white specks. Closing fills small black holes. Gradient = dilation minus erosion = outline.")
    r.source('''
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
eroded  = cv2.erode(binary, kernel)
dilated = cv2.dilate(binary, kernel)
opened  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closed  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)''')
    return r


@op("skeleton",
    desc="Thin every shape to a one-pixel-wide centre line using repeated erosion and opening.",
    formula="skeleton = union over k of ( erode^k(A) - open(erode^k(A)) )")
def skeleton(img, p, ctx):
    g, b = binarize(img)
    b = fit(b, 400)
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    sk = np.zeros_like(b)
    cur = b.copy()
    steps = 0
    while cv2.countNonZero(cur) > 0 and steps < 200:
        op_ = cv2.morphologyEx(cur, cv2.MORPH_OPEN, k)
        sk = cv2.bitwise_or(sk, cv2.subtract(cur, op_))
        cur = cv2.erode(cur, k)
        steps += 1
    from skimage.morphology import skeletonize
    sk2 = (skeletonize(b > 0) * 255).astype(np.uint8)
    over = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    over[sk2 > 0] = (51, 201, 255)
    over2 = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    over2[sk > 0] = (51, 201, 255)
    r = Result()
    r.image("Binary shape", b, "Otsu, smaller part as object", "binary")
    r.image("Morphological skeleton", sk, f"{steps} erosion rounds", "skeleton")
    r.image("Skeleton over shape", over2, "", "overlay")
    r.image("Thinning (scikit-image)", sk2, "connected one-pixel line", "thinning")
    r.stat("Object pixels", f"{int((b > 0).sum()):,}")
    r.stat("Skeleton pixels", f"{int((sk > 0).sum()):,}")
    r.stat("Thinning pixels", f"{int((sk2 > 0).sum()):,}")
    r.stat("Reduction", f"{100 * (1 - (sk2 > 0).sum() / max(1, (b > 0).sum())):.1f} %")
    r.stat("Erosion rounds", steps)
    r.note("The morphological skeleton can have gaps. Thinning keeps the line connected.")
    r.source('''
while cv2.countNonZero(img) > 0:
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    skel |= cv2.subtract(img, opened)
    img = cv2.erode(img, kernel)''')
    return r
