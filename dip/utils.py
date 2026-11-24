import base64
import math

import cv2
import numpy as np


class NeedsPackage(Exception):
    """Raised when an optional library, model or data file is missing."""


# ---------------------------------------------------------------- basics
def decode_image(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR uint8


def to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def hist256(gray):
    return np.bincount(gray.ravel(), minlength=256).astype(int).tolist()


def num(x, nd=4):
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return 0
    return int(x) if x.is_integer() else round(x, nd)


def norm8(x):
    x = np.asarray(x, dtype=np.float32)
    return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def fit(img, max_side=640):
    """Shrink big images so heavy algorithms stay fast."""
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return img


def crop_pct(img, x, y, w, h):
    """Crop using percentages (0-100) of the image size."""
    H, W = img.shape[:2]
    x0 = int(W * x / 100); y0 = int(H * y / 100)
    x1 = min(W, x0 + max(2, int(W * w / 100))); y1 = min(H, y0 + max(2, int(H * h / 100)))
    return img[y0:y1, x0:x1], (x0, y0, x1, y1)


# ---------------------------------------------------------------- quality metrics
def mse(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(a, b):
    m = mse(a, b)
    return 99.0 if m < 1e-10 else float(10 * math.log10(255.0 ** 2 / m))


def ssim(a, b):
    a = to_gray(a).astype(np.float64)
    b = to_gray(b)
    if b.shape != a.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)
    m1, m2 = blur(a), blur(b)
    s11 = blur(a * a) - m1 * m1
    s22 = blur(b * b) - m2 * m2
    s12 = blur(a * b) - m1 * m2
    s = ((2 * m1 * m2 + c1) * (2 * s12 + c2)) / ((m1 * m1 + m2 * m2 + c1) * (s11 + s22 + c2))
    return float(s.mean())


def entropy(gray):
    p = np.bincount(gray.ravel(), minlength=256) / gray.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# ---------------------------------------------------------------- encoding
def to_uri(img, max_side=760):
    """PNG (lossless, so the pixel inspector is exact). Tiny images are enlarged
    with nearest neighbour so each pixel stays visible."""
    h, w = img.shape[:2]
    if max(h, w) < 300:
        k = int(math.ceil(400 / max(h, w)))
        img = cv2.resize(img, (w * k, h * k), interpolation=cv2.INTER_NEAREST)
    else:
        scale = min(1.0, max_side / max(h, w))
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode(), (w, h)


def _py(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    return o


class Result:
    """Everything the page can show for one operation."""

    def __init__(self):
        self.images, self.stats, self.charts = [], [], []
        self.tables, self.notes, self.palettes, self.graphs = [], [], [], []
        self.code = ""
        self.layout = "normal"      # "gallery" = smaller image tiles
        self.headline = None

    def image(self, label, img, caption="", file="result"):
        if img.dtype == bool:
            img = img.astype(np.uint8) * 255
        elif img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        uri, (w, h) = to_uri(img)
        self.images.append({"label": label, "src": uri, "caption": caption,
                            "gray": bool(img.ndim == 2), "w": w, "h": h, "file": file})
        return self

    def gallery(self):
        self.layout = "gallery"
        return self

    def big(self, value, text):
        self.headline = {"value": str(value), "text": text}
        return self

    def stat(self, label, value, hint=""):
        self.stats.append({"label": label, "value": str(value), "hint": hint})
        return self

    def chart(self, title, series, marker=None, x_label="Pixel intensity", y_label="Pixels",
              labels=None, kind="auto"):
        """kind: auto (lines when many points, bars when few), bars, line.
        marker: a number, or a list of [x, label]."""
        self.charts.append({"title": title, "series": series, "marker": marker,
                            "xLabel": x_label, "yLabel": y_label, "labels": labels, "kind": kind})
        return self

    def table(self, title, rows, columns=None, heat=None, caption="", highlight=None):
        """heat=(min, max) paints each cell with that gray level, like imshow.
        highlight = row index to emphasise."""
        self.tables.append({"title": title, "rows": rows, "columns": columns,
                            "heat": list(heat) if heat else None, "caption": caption,
                            "highlight": highlight})
        return self

    def palette(self, title, colors, caption="", values=None):
        """colors: list of (R, G, B)."""
        self.palettes.append({"title": title, "colors": [[int(c) for c in col] for col in colors],
                              "caption": caption, "values": values})
        return self

    def graph(self, title, nodes, edges, caption=""):
        self.graphs.append({"title": title, "nodes": nodes, "edges": edges, "caption": caption})
        return self

    def note(self, text):
        self.notes.append(text)
        return self

    def source(self, code):
        self.code = code.strip("\n")
        return self

    def to_dict(self):
        return _py(self.__dict__)
