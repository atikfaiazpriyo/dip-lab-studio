"""Histogram lab: equalization step by step and histogram matching."""
import cv2
import numpy as np

from .registry import op
from .utils import Result, to_gray, hist256, num, fit, psnr, ssim, mse

GRAY, AMBER, RED, BLUE = "#d8d8d8", "#ffc933", "#ff4d4d", "#4d8dff"

SLIDE = np.array([
    [52, 55, 61, 66, 70, 61, 64, 73], [63, 59, 55, 90, 109, 85, 69, 72],
    [62, 59, 68, 113, 144, 104, 66, 73], [63, 58, 71, 122, 154, 106, 70, 69],
    [67, 61, 68, 104, 126, 88, 68, 70], [79, 65, 60, 70, 77, 68, 58, 75],
    [85, 71, 64, 59, 55, 61, 65, 83], [87, 79, 69, 68, 65, 76, 78, 94]], np.uint8)


@op("eq_steps",
    desc="Histogram equalization worked out value by value: count, CDF and the new level, exactly like the lab slide.",
    formula="h(v) = round( (cdf(v) - cdf_min) / (M N - cdf_min) x (L - 1) )   normalized\ns(v) = (L - 1) cdf(v) / (M N)   basic",
    params=[{"name": "src", "label": "Source", "type": "select", "default": "slide",
             "options": [["slide", "Lab slide 8 x 8 matrix"], ["tiny", "My image shrunk to 8 x 8"], ["full", "My full image"]]}])
def eq_steps(img, p, ctx):
    if p["src"] == "slide":
        a = SLIDE
    elif p["src"] == "tiny":
        a = cv2.resize(to_gray(img), (8, 8), interpolation=cv2.INTER_AREA)
    else:
        a = to_gray(fit(img, 640))
    MN = a.size
    h = np.bincount(a.ravel(), minlength=256)
    cdf = np.cumsum(h)
    cmin = int(cdf[np.nonzero(h)[0][0]])
    norm = np.round((cdf - cmin) / max(1, MN - cmin) * 255).clip(0, 255).astype(np.uint8)
    basic = np.round(255 * cdf / MN).astype(np.uint8)
    out_n, out_b = norm[a], basic[a]
    r = Result()
    r.image("Before", a, "", "before")
    r.image("After (normalized formula)", out_n, "", "after")
    r.stat("Pixels M x N", MN)
    r.stat("cdf_min", cmin, "first non-zero cdf")
    r.stat("Distinct levels", int((h > 0).sum()))
    r.stat("Range before", f"{int(a.min())} to {int(a.max())}")
    r.stat("Range after", f"{int(out_n.min())} to {int(out_n.max())}")
    if a.shape[0] <= 8:
        r.table("Matrix before", a.tolist(), heat=(0, 255))
        r.table("Matrix after", out_n.tolist(), heat=(0, 255))
    vals = np.nonzero(h)[0]
    rows = [[int(v), int(h[v]), int(cdf[v]), int(norm[v]), int(basic[v])] for v in vals[:48]]
    r.table("Step by step" + (" (first 48 levels)" if len(vals) > 48 else ""), rows,
            columns=["v", "count", "cdf(v)", "h(v) normalized", "s basic"],
            caption="Slide check: with the 8 x 8 matrix, h(52) = 0 and h(55) = 12." if p["src"] == "slide" else "")
    r.chart("Histogram before and after", [{"name": "Before", "color": GRAY, "values": hist256(a)},
                                           {"name": "After", "color": AMBER, "values": hist256(out_n)}])
    r.chart("Mapping v to h(v)", [{"name": "h(v)", "color": AMBER, "values": norm.astype(int).tolist()},
                                  {"name": "no change", "color": "#555", "values": list(range(256))}],
            x_label="Old level v", y_label="New level")
    r.source('''
hist = np.bincount(img.ravel(), minlength=256)
cdf = np.cumsum(hist)
cdf_min = cdf[cdf > 0].min()
h = np.round((cdf - cdf_min) / (img.size - cdf_min) * 255).astype(np.uint8)
equalized = h[img]''')
    return r


def specify(src, ref):
    """Histogram specification: map src levels so its CDF follows ref's CDF."""
    sc = np.cumsum(np.bincount(src.ravel(), minlength=256)) / src.size
    rc = np.cumsum(np.bincount(ref.ravel(), minlength=256)) / ref.size
    lut = np.searchsorted(rc, sc).clip(0, 255).astype(np.uint8)
    return lut[src], lut


@op("hist_match",
    desc="Make one image follow the histogram of another (reference). Then count how many pixels are alike within a tolerance.",
    formula="Map each level r to the level z whose CDF is closest:  G(z) = F(r)\nsimilar pixel:  |source(x, y) - reference(x, y)| <= tolerance",
    params=[
        {"name": "ref", "label": "Reference image", "type": "image", "default": None, "button": "Choose reference"},
        {"name": "tol", "label": "Tolerance", "type": "slider", "min": 0, "max": 50, "step": 1, "default": 5},
    ])
def hist_match(img, p, ctx):
    src = to_gray(fit(img, 800))
    ref = p.get("ref")
    demo = ref is None
    if demo:   # no reference chosen: use a brighter, contrasty version so the page is never empty
        ref_g = np.clip((src.astype(np.float32) - 40) * 1.6, 0, 255).astype(np.uint8)
    else:
        ref_g = to_gray(ref)
    ref_r = cv2.resize(ref_g, (src.shape[1], src.shape[0]))
    out, lut = specify(src, ref_g)
    tol = int(p["tol"])
    def sim(a, b):
        return 100 * float((np.abs(a.astype(np.int16) - b.astype(np.int16)) <= tol).mean())
    before, after = sim(src, ref_r), sim(out, ref_r)
    r = Result()
    r.image("Source", src, "", "source")
    r.image("Reference" + (" (demo, no file chosen)" if demo else ""), ref_r, "Resized to the source size", "reference")
    r.image("Source after matching", out, "Histogram now follows the reference", "matched")
    sm = np.abs(src.astype(np.int16) - ref_r.astype(np.int16)) <= tol
    r.image(f"Similar pixels before (tol {tol})", sm, "White = within tolerance", "similar_before")
    r.image("Similar pixels after", np.abs(out.astype(np.int16) - ref_r.astype(np.int16)) <= tol, "White = within tolerance", "similar_after")
    r.stat("Similar before", f"{before:.2f} %", f"tolerance {tol}")
    r.stat("Similar after", f"{after:.2f} %", "after matching")
    r.stat("Different pixels after", f"{int(round((100 - after) / 100 * src.size)):,}")
    r.stat("MSE to reference", num(mse(out, ref_r), 2), f"was {mse(src, ref_r):.2f}")
    r.chart("Histograms", [{"name": "Source", "color": GRAY, "values": hist256(src)},
                           {"name": "Reference", "color": BLUE, "values": hist256(ref_g)},
                           {"name": "Matched", "color": AMBER, "values": hist256(out)}])
    cd = lambda g: (np.cumsum(np.bincount(g.ravel(), minlength=256)) / g.size).round(5).tolist()
    r.chart("CDFs", [{"name": "Source", "color": GRAY, "values": cd(src)},
                     {"name": "Reference", "color": BLUE, "values": cd(ref_g)},
                     {"name": "Matched", "color": AMBER, "values": cd(out)}], y_label="Cumulative")
    r.chart("Mapping curve", [{"name": "level r to z", "color": AMBER, "values": lut.astype(int).tolist()}],
            x_label="Source level", y_label="New level")
    if demo:
        r.note("Choose a reference image above to match your picture to it.")
    r.source('''
difference = np.abs(source.astype(np.int16) - reference.astype(np.int16))
similar = difference <= tolerance
similarity = similar.sum() / similar.size * 100''')
    return r
