"""Edge detection."""
import cv2
import numpy as np

from .registry import op
from .utils import Result, to_gray, hist256, num, fit, norm8

AMBER, GRAY, BLUE = "#ffc933", "#d8d8d8", "#4d8dff"


def _k(name, k):
    return (name, k.astype(int).tolist())


@op("edges",
    desc="Find where brightness changes quickly with Sobel, Prewitt, Roberts and Laplacian filters.",
    formula="Sobel: Gx and Gy with 3 x 3 kernels\nmagnitude = sqrt(Gx^2 + Gy^2)\nLaplacian: second derivative, responds to all directions",
    params=[
        {"name": "blur", "label": "Blur first", "type": "checkbox", "default": True},
        {"name": "t", "label": "Edge threshold %", "type": "slider", "min": 5, "max": 90, "step": 1, "default": 25},
    ])
def edges(img, p, ctx):
    g = to_gray(fit(img, 800))
    if p["blur"]:
        g = cv2.GaussianBlur(g, (3, 3), 0)
    f = g.astype(np.float32)
    sx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    sob = cv2.magnitude(sx, sy)
    px = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
    py = px.T
    pre = cv2.magnitude(cv2.filter2D(f, -1, px), cv2.filter2D(f, -1, py))
    rx = np.array([[1, 0], [0, -1]], np.float32)
    ry = np.array([[0, 1], [-1, 0]], np.float32)
    rob = cv2.magnitude(cv2.filter2D(f, -1, rx, anchor=(0, 0)), cv2.filter2D(f, -1, ry, anchor=(0, 0)))
    lap = np.abs(cv2.Laplacian(f, cv2.CV_32F, ksize=3))
    res = {"Sobel": sob, "Prewitt": pre, "Roberts": rob, "Laplacian": lap}
    r = Result().gallery()
    r.image("Input (gray)", g, "", "input")
    r.image("Sobel X (vertical edges)", norm8(np.abs(sx)), "", "sobel_x")
    r.image("Sobel Y (horizontal edges)", norm8(np.abs(sy)), "", "sobel_y")
    rows = []
    for n, m in res.items():
        m8 = norm8(m)
        r.image(n, m8, "normalized to 0..255", n.lower())
        thr = p["t"] / 100 * 255
        rows.append([n, round(float(m8.mean()), 2), int(m8.max()), round(100 * float((m8 > thr).mean()), 2)])
    r.table("Edge strength", rows, columns=["Filter", "Mean response", "Max", f"Edge pixels % (> {p['t']}%)"])
    r.image("Sobel after threshold", norm8(sob) > p["t"] / 100 * 255, f"Threshold {p['t']} %", "sobel_binary")
    kn = [("Sobel X", np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])), ("Sobel Y", np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])),
          ("Prewitt X", px), ("Roberts X", rx), ("Laplacian", np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]))]
    for n, k in kn:
        r.table(f"{n} kernel", k.astype(int).tolist(), heat=(-2, 2) if "Lap" not in n else (-4, 1))
    r.chart("Edge response histograms", [{"name": n, "color": col, "values": hist256(norm8(res[n]))}
                                         for n, col in (("Sobel", AMBER), ("Laplacian", BLUE))])
    r.source('''
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
mag = cv2.magnitude(sobelx, sobely)
lap = cv2.Laplacian(gray, cv2.CV_64F)
cv2.normalize(mag, mag, 0, 255, cv2.NORM_MINMAX)''')
    return r


@op("canny",
    desc="Canny edge detector: smooth, find gradients, keep the thin ridges and link them with two thresholds.",
    formula="strong edge: gradient > high\nweak edge: low < gradient <= high, kept only if connected to a strong edge",
    params=[
        {"name": "low", "label": "Low", "type": "slider", "min": 0, "max": 255, "step": 1, "default": 50},
        {"name": "high", "label": "High", "type": "slider", "min": 0, "max": 400, "step": 1, "default": 150},
        {"name": "ap", "label": "Aperture", "type": "select", "default": "3", "options": [["3", "3"], ["5", "5"], ["7", "7"]]},
        {"name": "blur", "label": "Blur sigma", "type": "slider", "min": 0, "max": 5, "step": 0.5, "default": 1.4},
    ])
def canny(img, p, ctx):
    g = to_gray(fit(img, 800))
    sm = cv2.GaussianBlur(g, (0, 0), p["blur"]) if p["blur"] > 0 else g
    lo, hi = min(p["low"], p["high"]), max(p["low"], p["high"])
    ap = int(p["ap"])
    ed = cv2.Canny(sm, lo, hi, apertureSize=ap, L2gradient=True)
    f = sm.astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=ap), cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=ap))
    strong = mag > hi
    weak = (mag > lo) & ~strong
    over = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    over[ed > 0] = (51, 201, 255)
    n, _ = cv2.connectedComponents((ed > 0).astype(np.uint8), connectivity=8)
    r = Result().gallery()
    r.image("Input (gray)", g, "", "input")
    r.image("After Gaussian blur", sm, f"sigma {p['blur']}", "blurred")
    r.image("Gradient magnitude", norm8(mag), "", "gradient")
    r.image("Strong pixels", strong, f"> {hi}", "strong")
    r.image("Weak pixels", weak, f"{lo} to {hi}", "weak")
    r.image("Canny result", ed, "after hysteresis", "canny")
    r.image("Edges over the image", over, "", "overlay")
    r.stat("Edge pixels", f"{int((ed > 0).sum()):,}")
    r.stat("Edge share", f"{100 * float((ed > 0).mean()):.2f} %")
    r.stat("Connected edge pieces", n - 1)
    r.stat("Strong / weak", f"{int(strong.sum()):,} / {int(weak.sum()):,}")
    h, _ = np.histogram(np.clip(mag, 0, 600), bins=60, range=(0, 600))
    r.chart("Gradient magnitude histogram", [{"name": "Pixels", "color": GRAY, "values": h.tolist()}],
            marker=[[lo / 10, f"low {lo}"], [hi / 10, f"high {hi}"]], x_label="Gradient (10 per bin)", kind="line")
    r.note("Raise the high threshold for fewer, cleaner edges. A low value close to high drops the weak linking.")
    r.source('''
blur = cv2.GaussianBlur(gray, (5, 5), 1.4)
edges = cv2.Canny(blur, 50, 150)''')
    return r
