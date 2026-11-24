"""Enhancement: contrast, gamma, contrast measures, histogram, stretching, equalization, CLAHE."""
import cv2
import numpy as np

from .registry import op
from .utils import Result, to_gray, hist256, num, fit, psnr, entropy

GRAY, AMBER, RED, GREEN, BLUE = "#d8d8d8", "#ffc933", "#ff4d4d", "#3ddc84", "#4d8dff"


def cdf_of(g):
    h = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    c = np.cumsum(h)
    return (c / c[-1]).round(5).tolist()


def michelson(g):
    mx, mn = float(g.max()), float(g.min())
    return 0.0 if mx + mn == 0 else (mx - mn) / (mx + mn)


def measures(g):
    return {"simple": float(g.max()) - float(g.min()), "michelson": michelson(g),
            "rms": float(g.astype(np.float64).std()), "entropy": entropy(g), "mean": float(g.mean())}


# ------------------------------------------------------------------ contrast
@op("contrast",
    desc="Multiply by a gain (alpha) and add an offset (beta). Alpha changes contrast, beta changes brightness.",
    formula="g(x, y) = alpha f(x, y) + beta\nauto stretch: alpha = 255 / (r_max - r_min),  beta = -alpha r_min",
    params=[
        {"name": "mode", "label": "Mode", "type": "select", "default": "single",
         "options": [["single", "One alpha and beta"], ["gallery", "Gallery of alphas (0.2 to 1.8)"], ["auto", "Auto stretch"]]},
        {"name": "alpha", "label": "alpha", "type": "slider", "min": 0.1, "max": 3, "step": 0.05, "default": 1.5},
        {"name": "beta", "label": "beta", "type": "slider", "min": -100, "max": 100, "step": 1, "default": 0},
    ])
def contrast(img, p, ctx):
    c = fit(img, 800)
    g0 = to_gray(c)
    r = Result()
    if p["mode"] == "gallery":
        r.gallery().image("Original", c, "alpha = 1", "original")
        rows = []
        for a in [0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.6, 1.8, 2.0]:
            o = cv2.convertScaleAbs(c, alpha=a, beta=p["beta"])
            r.image(f"alpha = {a}", o, f"std {to_gray(o).std():.1f}", f"alpha_{a}")
            gg = to_gray(o)
            rows.append([a, int(gg.min()), int(gg.max()), round(float(gg.mean()), 1), round(float(gg.std()), 1)])
        r.table("Effect of alpha", rows, columns=["alpha", "Min", "Max", "Mean", "Std"],
                caption="Std grows with alpha: that is the contrast.")
        r.chart("Std (contrast) against alpha", [{"name": "Std", "color": AMBER, "values": [x[4] for x in rows]}],
                labels=[str(x[0]) for x in rows], x_label="alpha", y_label="Std", kind="line")
        return r
    if p["mode"] == "auto":
        rmin, rmax = float(c.min()), float(c.max())
        a = 255.0 / max(1.0, rmax - rmin)
        b = -a * rmin
    else:
        a, b = float(p["alpha"]), float(p["beta"])
    o = cv2.convertScaleAbs(c, alpha=a, beta=b)
    g1 = to_gray(o)
    r.image("Original", c, "", "original")
    r.image("Result", o, f"alpha = {a:.3f}, beta = {b:.1f}", "contrast")
    r.stat("alpha", f"{a:.3f}")
    r.stat("beta", f"{b:.1f}")
    r.stat("Std before", num(g0.std(), 2))
    r.stat("Std after", num(g1.std(), 2))
    r.stat("Range before", f"{int(g0.min())} to {int(g0.max())}")
    r.stat("Range after", f"{int(g1.min())} to {int(g1.max())}")
    r.chart("Histogram before and after", [{"name": "Original", "color": GRAY, "values": hist256(g0)},
                                           {"name": "Result", "color": AMBER, "values": hist256(g1)}])
    sat = 100 * float(((g1 == 0) | (g1 == 255)).mean())
    r.note(f"{sat:.2f} % of pixels are pure black or white after the change. A high value means detail was clipped.")
    r.source('''
out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
# auto stretch
alpha = 255.0 / (img.max() - img.min()); beta = -alpha * img.min()''')
    return r


@op("gamma",
    desc="Brighten or darken with a power law. Gamma below 1 lifts shadows, above 1 darkens.",
    formula="s = 255 (r / 255) ^ gamma",
    params=[{"name": "g", "label": "gamma", "type": "slider", "min": 0.1, "max": 4, "step": 0.05, "default": 0.5}])
def gamma(img, p, ctx):
    gm = float(p["g"])
    c = fit(img, 800)
    lut = (255 * (np.arange(256) / 255.0) ** gm).clip(0, 255).astype(np.uint8)
    o = cv2.LUT(c, lut)
    g0, g1 = to_gray(c), to_gray(o)
    r = Result()
    r.image("Original", c, "", "original")
    r.image("Gamma corrected", o, f"gamma = {gm}", "gamma")
    r.stat("gamma", gm)
    r.stat("Mean before", num(g0.mean(), 2))
    r.stat("Mean after", num(g1.mean(), 2))
    r.stat("Gray 128 becomes", int(lut[128]))
    r.chart("Transfer curve (input to output)", [
        {"name": f"gamma {gm}", "color": AMBER, "values": lut.astype(int).tolist()},
        {"name": "gamma 1 (no change)", "color": "#555", "values": list(range(256))}],
        x_label="Input level", y_label="Output level")
    r.chart("Histogram before and after", [{"name": "Original", "color": GRAY, "values": hist256(g0)},
                                           {"name": "Result", "color": AMBER, "values": hist256(g1)}])
    r.source('''
lut = np.array([255 * (i / 255) ** gamma for i in range(256)], np.uint8)
out = cv2.LUT(img, lut)''')
    return r


@op("contrast_metrics",
    desc="Four ways to put a number on contrast, compared for the original, a low-contrast and a high-contrast version.",
    formula="simple = max - min\nMichelson = (max - min) / (max + min)\nRMS = std of intensities\nentropy = -sum p log2 p")
def contrast_metrics(img, p, ctx):
    c = fit(img, 640)
    g = to_gray(c)
    low = cv2.convertScaleAbs(g, alpha=0.5, beta=0)
    high = cv2.convertScaleAbs(g, alpha=1.8, beta=0)
    r = Result()
    rows = []
    for nm, im in (("Original", g), ("Low contrast (alpha 0.5)", low), ("High contrast (alpha 1.8)", high)):
        r.image(nm, im, "", nm.split()[0].lower())
        m = measures(im)
        rows.append([nm, round(m["simple"], 1), round(m["michelson"], 4), round(m["rms"], 2), round(m["entropy"], 3), round(m["mean"], 1)])
    r.table("Contrast measures", rows, columns=["Image", "Simple", "Michelson", "RMS", "Entropy", "Mean"])
    r.chart("RMS contrast", [{"name": "RMS", "color": AMBER, "values": [x[3] for x in rows]}],
            labels=["Original", "Low", "High"], x_label="Image", y_label="RMS", kind="bars")
    r.chart("Entropy", [{"name": "Entropy", "color": BLUE, "values": [x[4] for x in rows]}],
            labels=["Original", "Low", "High"], x_label="Image", y_label="bits", kind="bars")
    r.chart("Histograms", [{"name": "Original", "color": GRAY, "values": hist256(g)},
                           {"name": "Low", "color": BLUE, "values": hist256(low)},
                           {"name": "High", "color": RED, "values": hist256(high)}])
    r.note("High alpha clips pixels to 0 or 255, so Michelson and simple contrast saturate while entropy can drop.")
    r.source('''
michelson = (img.max() - img.min()) / (img.max() + img.min())
rms = img.std()''')
    return r


# ------------------------------------------------------------------ histogram
@op("histogram",
    desc="How many pixels have each brightness. Histogram, PDF, CDF and the numbers read from them.",
    formula="h(r_k) = n_k\nPDF: p(r_k) = n_k / MN\nCDF: c(r_k) = sum of p(r_j) for j <= k",
    params=[
        {"name": "ch", "label": "Channel", "type": "select", "default": "gray",
         "options": [["gray", "Gray"], ["color", "R, G and B"], ["R", "Red"], ["G", "Green"], ["B", "Blue"]]},
        {"name": "bins", "label": "Bins", "type": "select", "default": "256",
         "options": [["256", "256"], ["64", "64"], ["32", "32"], ["16", "16"], ["8", "8"]]},
    ])
def histogram(img, p, ctx):
    g = to_gray(img)
    bins = int(p["bins"])
    def hb(ch):
        return np.histogram(ch.ravel(), bins=bins, range=(0, 256))[0]
    r = Result()
    r.image("Original", img, "", "original")
    r.image("Grayscale", g, "", "gray")
    if p["ch"] == "color":
        series = [{"name": n, "color": col, "values": hb(img[:, :, i]).tolist()}
                  for n, col, i in (("R", RED, 2), ("G", GREEN, 1), ("B", BLUE, 0))]
        ch = g
    elif p["ch"] in "RGB" and p["ch"] != "gray":
        i = {"R": 2, "G": 1, "B": 0}[p["ch"]]
        ch = img[:, :, i]
        series = [{"name": p["ch"], "color": {"R": RED, "G": GREEN, "B": BLUE}[p["ch"]], "values": hb(ch).tolist()}]
    else:
        ch = g
        series = [{"name": "Gray", "color": GRAY, "values": hb(ch).tolist()}]
    r.chart(f"Histogram ({bins} bins)", series, x_label=f"Bin (width {256 // bins})")
    h = np.bincount(ch.ravel(), minlength=256)
    pdf = h / h.sum()
    cdf = np.cumsum(pdf)
    r.chart("PDF", [{"name": "PDF", "color": AMBER, "values": pdf.round(6).tolist()}], y_label="Probability")
    r.chart("CDF", [{"name": "CDF", "color": BLUE, "values": cdf.round(5).tolist()}], y_label="Cumulative")
    r.stat("Mean", num(ch.mean(), 2))
    r.stat("Median", num(np.median(ch), 1))
    r.stat("Mode", int(np.argmax(h)), f"{int(h.max()):,} pixels")
    r.stat("Entropy", f"{entropy(ch):.3f} bits")
    r.stat("Dark pixels (< 85)", f"{100 * h[:85].sum() / h.sum():.1f} %")
    r.stat("Bright pixels (> 170)", f"{100 * h[171:].sum() / h.sum():.1f} %")
    ex = [2, 5, 3, 4, 5, 2, 4]
    tot, cum, rows = sum(ex), 0, []
    for k, n in enumerate(ex):
        cum += n
        rows.append([k, n, round(n / tot, 2), round(cum / tot, 2)])
    r.table("Worked example: 5 x 5 image with levels 0 to 6", rows, columns=["Level", "Count", "PDF", "CDF"],
            caption="25 pixels in total. The CDF always ends at 1.")
    r.source('''
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
pdf = hist / hist.sum()
cdf = np.cumsum(pdf)''')
    return r


@op("stretching",
    desc="Stretch the used brightness range to fill 0 to 255. Percentile clipping ignores a few extreme pixels.",
    formula="s = 255 (r - r_min) / (r_max - r_min)",
    params=[
        {"name": "clip", "label": "Clip % each side", "type": "slider", "min": 0, "max": 20, "step": 0.5, "default": 1},
        {"name": "demo", "label": "Squeeze to 80..160 first (demo)", "type": "checkbox", "default": False},
    ])
def stretching(img, p, ctx):
    g = to_gray(fit(img, 800))
    if p["demo"]:
        g = (80 + g.astype(np.float32) * 80 / 255).astype(np.uint8)
    lo, hi = np.percentile(g, [p["clip"], 100 - p["clip"]])
    hi = max(hi, lo + 1)
    o = np.clip((g.astype(np.float32) - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)
    r = Result()
    r.image("Before", g, f"{int(g.min())} to {int(g.max())}", "before")
    r.image("After stretching", o, f"{int(o.min())} to {int(o.max())}", "stretched")
    r.stat("r_min used", num(lo, 1))
    r.stat("r_max used", num(hi, 1))
    r.stat("Std before", num(g.std(), 2))
    r.stat("Std after", num(o.std(), 2))
    r.chart("Histogram before and after", [{"name": "Before", "color": GRAY, "values": hist256(g)},
                                           {"name": "After", "color": AMBER, "values": hist256(o)}],
            marker=[[int(lo), f"r_min {int(lo)}"], [int(hi), f"r_max {int(hi)}"]])
    r.source('''
lo, hi = np.percentile(img, [1, 99])
out = np.clip((img - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)
# or: cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)''')
    return r


@op("equalization",
    desc="Redistribute brightness so the histogram becomes as flat as possible. Strong boost for dull images.",
    formula="s_k = (L - 1) sum_{j<=k} p(r_j)\nh(v) = round( (cdf(v) - cdf_min) / (MN - cdf_min) (L - 1) )",
    params=[{"name": "mode", "label": "Image", "type": "select", "default": "gray",
             "options": [["gray", "Grayscale"], ["color", "Color (equalize Y only)"], ["bgr", "Each RGB channel"]]}])
def equalization(img, p, ctx):
    c = fit(img, 800)
    g = to_gray(c)
    ge = cv2.equalizeHist(g)
    r = Result()
    if p["mode"] == "gray":
        r.image("Original", g, "", "original")
        r.image("Equalized", ge, "", "equalized")
        a, b = g, ge
    elif p["mode"] == "color":
        y = cv2.cvtColor(c, cv2.COLOR_BGR2YCrCb)
        y[:, :, 0] = cv2.equalizeHist(y[:, :, 0])
        o = cv2.cvtColor(y, cv2.COLOR_YCrCb2BGR)
        r.image("Original", c, "", "original")
        r.image("Equalized (Y channel)", o, "Colors stay natural", "equalized")
        a, b = g, to_gray(o)
    else:
        o = cv2.merge([cv2.equalizeHist(c[:, :, i]) for i in range(3)])
        r.image("Original", c, "", "original")
        r.image("Each channel equalized", o, "Colors can shift", "equalized")
        a, b = g, to_gray(o)
    r.stat("Std before", num(a.std(), 2))
    r.stat("Std after", num(b.std(), 2))
    r.stat("Entropy before", f"{entropy(a):.3f}")
    r.stat("Entropy after", f"{entropy(b):.3f}")
    r.stat("Michelson before", f"{michelson(a):.3f}")
    r.stat("Michelson after", f"{michelson(b):.3f}")
    r.chart("Histogram before and after", [{"name": "Before", "color": GRAY, "values": hist256(a)},
                                           {"name": "After", "color": AMBER, "values": hist256(b)}])
    r.chart("CDF before and after", [{"name": "Before", "color": GRAY, "values": cdf_of(a)},
                                     {"name": "After", "color": AMBER, "values": cdf_of(b)}], y_label="Cumulative")
    r.note("After equalization the CDF follows a straight diagonal, which is what a flat histogram looks like.")
    r.source('''
equalized = cv2.equalizeHist(gray)
# color: equalize only the Y channel of YCrCb''')
    return r


@op("clahe",
    desc="Equalize small tiles separately and limit how much contrast any tile may gain. Less noise than plain equalization.",
    formula="clipLimit caps each histogram bin, extra counts are spread equally; tiles are blended by interpolation",
    params=[
        {"name": "clip", "label": "Clip limit", "type": "slider", "min": 1, "max": 40, "step": 0.5, "default": 3},
        {"name": "tile", "label": "Tile grid", "type": "slider", "min": 2, "max": 16, "step": 1, "default": 8},
        {"name": "mode", "label": "Image", "type": "select", "default": "gray", "options": [["gray", "Grayscale"], ["lab", "Color (L channel of Lab)"]]},
    ])
def clahe(img, p, ctx):
    c = fit(img, 800)
    t = int(p["tile"])
    cl = cv2.createCLAHE(clipLimit=float(p["clip"]), tileGridSize=(t, t))
    g = to_gray(c)
    he = cv2.equalizeHist(g)
    r = Result()
    if p["mode"] == "gray":
        o = cl.apply(g)
        r.image("Original", g, "", "original")
        r.image("Global equalization", he, "", "he")
        r.image("CLAHE", o, f"clip {p['clip']}, tiles {t} x {t}", "clahe")
        og = o
    else:
        lab = cv2.cvtColor(c, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cl.apply(lab[:, :, 0])
        o = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        r.image("Original", c, "", "original")
        r.image("Global equalization (gray)", he, "", "he")
        r.image("CLAHE on L channel", o, f"clip {p['clip']}, tiles {t} x {t}", "clahe")
        og = to_gray(o)
    r.table("Comparison", [[n, round(float(x.std()), 2), round(entropy(x), 3), round(michelson(x), 3)]
                           for n, x in (("Original", g), ("Global equalization", he), ("CLAHE", og))],
            columns=["Image", "Std", "Entropy", "Michelson"])
    r.chart("Histograms", [{"name": "Original", "color": GRAY, "values": hist256(g)},
                           {"name": "Global", "color": BLUE, "values": hist256(he)},
                           {"name": "CLAHE", "color": AMBER, "values": hist256(og)}])
    r.source('''
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
out = clahe.apply(gray)''')
    return r
