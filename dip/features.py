"""Feature extraction: Harris, HOG, LBP, GLCM, keypoints, zoom."""
import cv2
import numpy as np

from .registry import op
from .utils import Result, to_gray, hist256, num, fit, norm8
from .utils import NeedsPackage

AMBER, GRAY, BLUE, RED, GREEN = "#ffc933", "#d8d8d8", "#4d8dff", "#ff4d4d", "#3ddc84"


def harris_resp(g, block=2, ksize=3, k=0.04):
    h = cv2.cornerHarris(np.float32(g), block, ksize, k)
    return h, cv2.dilate(h, None)


def pick_points(hd, thr, min_dist, limit):
    """Strongest Harris points first, no two closer than min_dist."""
    ys, xs = np.where(hd > thr)
    order = np.argsort(-hd[ys, xs])
    pts = []
    for i in order:
        x, y = int(xs[i]), int(ys[i])
        if all(abs(x - px) > min_dist or abs(y - py) > min_dist for px, py, _ in pts):
            pts.append((x, y, float(hd[y, x])))
            if len(pts) >= limit:
                break
    return pts


def make_detector(name, n):
    if name == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise NeedsPackage("This OpenCV build has no SIFT. Run: pip install --upgrade opencv-python")
        return cv2.SIFT_create(nfeatures=n)
    if name == "orb":
        return cv2.ORB_create(nfeatures=n)
    def mk(cls, legacy, *a, **kw):
        if hasattr(cv2, legacy):
            return getattr(cv2, legacy)(*a, **kw)
        return getattr(cv2, cls).create(*a, **kw)
    if name == "brisk":
        return mk("BRISK", "BRISK_create")
    if name == "akaze":
        return mk("AKAZE", "AKAZE_create")
    if name == "fast":
        return mk("FastFeatureDetector", "FastFeatureDetector_create", threshold=20)
    raise ValueError(name)


# ------------------------------------------------------------------ all at once
@op("feature_summary",
    desc="Harris, HOG, LBP, GLCM, SIFT, ORB and FAST on your image in one page, like the lab feature script.",
    formula="Harris R = det(M) - k trace(M)^2\nHOG: histograms of gradient direction per cell\nLBP: compare 8 neighbours with the centre\nGLCM: how often gray pairs occur side by side")
def feature_summary(img, p, ctx):
    from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
    from skimage import exposure
    c = fit(img, 512)
    g = to_gray(c)
    _, hd = harris_resp(g)
    hv = c.copy()
    hv[hd > 0.01 * hd.max()] = (0, 0, 255)
    feats, himg = hog(g, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    lbp = local_binary_pattern(g, 8, 1, "uniform")
    lh = np.histogram(lbp.ravel(), bins=np.arange(0, 11))[0]
    q = (g // 8).astype(np.uint8)
    glcm = graycomatrix(q, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32, symmetric=True, normed=True)
    props = {n: float(graycoprops(glcm, n).mean()) for n in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")}
    r = Result().gallery()
    r.image("Original", c, "", "original")
    r.image("Harris corners", hv, "red = corner response", "harris")
    r.image("HOG", norm8(exposure.rescale_intensity(himg, in_range=(0, 10))), f"{len(feats):,} numbers", "hog")
    r.image("LBP (uniform)", norm8(lbp), "", "lbp")
    counts = {}
    for nm in ("sift", "orb", "fast"):
        try:
            kp = make_detector(nm, 500).detect(g, None)
            counts[nm.upper()] = len(kp)
            r.image(nm.upper(), cv2.drawKeypoints(c, kp, None, (51, 201, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if nm != "fast" else 0),
                    f"{len(kp)} keypoints", nm)
        except Exception as e:
            counts[nm.upper()] = 0
    r.stat("Harris corner pixels", f"{int((hd > 0.01 * hd.max()).sum()):,}")
    r.stat("HOG vector length", f"{len(feats):,}")
    for k_, v in counts.items():
        r.stat(f"{k_} keypoints", v)
    r.table("GLCM texture (mean of 4 angles)", [[k_, round(v, 4)] for k_, v in props.items()], columns=["Property", "Value"])
    r.chart("LBP histogram (10 uniform patterns)", [{"name": "Pixels", "color": AMBER, "values": lh.tolist()}],
            labels=[str(i) for i in range(10)], x_label="LBP code", kind="bars")
    r.chart("GLCM properties", [{"name": "Value", "color": BLUE, "values": [round(v, 4) for v in props.values()]}],
            labels=list(props.keys()), x_label="Property", y_label="Value", kind="bars")
    r.source('''
harris = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
features, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
lbp = local_binary_pattern(gray, 8, 1, method="uniform")
glcm = graycomatrix(gray, [1], [0], levels=256, symmetric=True, normed=True)''')
    return r


@op("harris",
    desc="Find corners: points where the image changes strongly in every direction.",
    formula="M = sum [Ix^2  IxIy; IxIy  Iy^2]\nR = det(M) - k (trace M)^2\ncorner if R > threshold",
    params=[
        {"name": "block", "label": "Block size", "type": "slider", "min": 2, "max": 9, "step": 1, "default": 2},
        {"name": "k", "label": "k", "type": "slider", "min": 0.02, "max": 0.1, "step": 0.01, "default": 0.04},
        {"name": "t", "label": "Threshold % of max", "type": "slider", "min": 0.1, "max": 20, "step": 0.1, "default": 1},
    ])
def harris(img, p, ctx):
    c = fit(img, 640)
    g = to_gray(c)
    h, hd = harris_resp(g, int(p["block"]), 3, float(p["k"]))
    thr = p["t"] / 100 * hd.max()
    mask = (hd > thr).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
    over = c.copy()
    over[mask > 0] = (0, 0, 255)
    pts = pick_points(hd, thr, 12, 60)
    mk = c.copy()
    for x, y, _ in pts:
        cv2.circle(mk, (x, y), 6, (0, 255, 0), 1)
    r = Result().gallery()
    r.big(n - 1, "corner regions found")
    r.image("Original", c, "", "original")
    r.image("Harris response", norm8(np.log1p(np.clip(hd, 0, None) / max(1e-12, hd.max()) * 1000)), "bright = corner-like", "response")
    r.image("Corner pixels in red", over, f"R > {p['t']} % of max", "corners")
    r.image("Strongest corners", mk, f"{len(pts)} shown, at least 12 px apart", "points")
    r.stat("Corner pixels", f"{int(mask.sum()):,}")
    r.stat("Max response", f"{hd.max():.3e}")
    r.stat("Min response", f"{hd.min():.3e}", "negative = edge")
    r.stat("Edge pixels (R < 0)", f"{100 * float((hd < 0).mean()):.1f} %")
    r.table("Top corners", [[i + 1, x, y, f"{v:.3e}"] for i, (x, y, v) in enumerate(pts[:12])], columns=["#", "x", "y", "R"])
    r.source('''
harris = cv2.cornerHarris(np.float32(gray), blockSize=2, ksize=3, k=0.04)
harris = cv2.dilate(harris, None)
img[harris > 0.01 * harris.max()] = [0, 0, 255]''')
    return r


@op("hog",
    desc="Histogram of Oriented Gradients: the description of shape used for people detection.",
    formula="gradient direction in each cell is voted into orientation bins\nlength = blocks x block cells x orientations",
    params=[
        {"name": "ori", "label": "Orientations", "type": "slider", "min": 4, "max": 12, "step": 1, "default": 9},
        {"name": "cell", "label": "Cell size", "type": "select", "default": "8", "options": [["8", "8 x 8"], ["16", "16 x 16"], ["32", "32 x 32"]]},
    ], heavy=True)
def hog_op(img, p, ctx):
    from skimage.feature import hog
    from skimage import exposure
    g = to_gray(fit(img, 384))
    cs = int(p["cell"]); o = int(p["ori"])
    if min(g.shape) < 2 * cs:
        sc = 2.0 * cs / min(g.shape)
        g = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    feats, im = hog(g, orientations=o, pixels_per_cell=(cs, cs), cells_per_block=(2, 2), visualize=True)
    im = exposure.rescale_intensity(im, in_range=(0, float(np.percentile(im, 99.5)) or 1))
    hy, hx = g.shape[0] // cs, g.shape[1] // cs
    r = Result()
    r.image("Gray input", g, "", "gray")
    r.image("HOG visualization", norm8(im), "Lines show dominant edge direction in each cell", "hog")
    r.stat("Cells", f"{hx} x {hy}")
    r.stat("Blocks", f"{hx - 1} x {hy - 1}", "2 x 2 cells, step 1")
    r.stat("Feature length", f"{len(feats):,}", f"{(hx - 1) * (hy - 1)} blocks x 4 cells x {o}")
    r.stat("Largest value", f"{feats.max():.3f}")
    # mean histogram per orientation
    fo = feats.reshape(-1, o)
    r.chart("Average strength per orientation bin", [{"name": "Mean", "color": AMBER, "values": fo.mean(0).round(4).tolist()}],
            labels=[f"{int(i * 180 / o)}" for i in range(o)], x_label="Angle (degrees)", y_label="Mean", kind="bars")
    r.table("First 12 numbers of the descriptor", [[round(float(v), 4) for v in feats[:12]]], columns=[str(i) for i in range(12)])
    r.source('''
features, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=True)
print(len(features))''')
    return r


@op("lbp",
    desc="Local Binary Pattern: describe texture by comparing each pixel with its 8 neighbours.",
    formula="LBP = sum over neighbours of s(g_n - g_c) 2^n,   s(x) = 1 if x >= 0 else 0",
    params=[
        {"name": "radius", "label": "Radius", "type": "slider", "min": 1, "max": 4, "step": 1, "default": 1},
        {"name": "method", "label": "Method", "type": "select", "default": "uniform", "options": [["uniform", "Uniform"], ["default", "Default (256 codes)"], ["ror", "Rotation invariant"]]},
    ])
def lbp(img, p, ctx):
    from skimage.feature import local_binary_pattern
    g = to_gray(fit(img, 640))
    R = int(p["radius"]); P = 8 * R if p["method"] == "uniform" else 8
    L = local_binary_pattern(g, P, R, p["method"])
    nb = int(L.max()) + 1
    hist = np.bincount(L.astype(int).ravel(), minlength=nb)
    r = Result()
    r.image("Gray input", g, "", "gray")
    r.image("LBP image", norm8(L), f"{P} points, radius {R}", "lbp")
    r.stat("Distinct codes", int((hist > 0).sum()))
    r.stat("Most common code", int(np.argmax(hist)), f"{100 * hist.max() / hist.sum():.1f} % of pixels")
    r.chart("LBP histogram (the texture feature)", [{"name": "Pixels", "color": AMBER, "values": hist.tolist()}],
            x_label="LBP code", kind="bars" if nb <= 40 else "line")
    h, w = g.shape
    y, x = h // 2, w // 2
    patch = g[y - 1:y + 2, x - 1:x + 2].astype(int)
    cen = patch[1, 1]
    order = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
    bits = [1 if patch[a, b] >= cen else 0 for a, b in order]
    code = sum(b << i for i, b in enumerate(bits))
    r.table("Worked example: 3 x 3 neighbourhood at the centre", patch.tolist(), heat=(0, 255), caption=f"Centre value = {cen}")
    r.table("1 if neighbour >= centre", [[1 if patch[i, j] >= cen else 0 for j in range(3)] for i in range(3)], heat=(0, 1),
            caption=f"Read clockwise from top-left: {''.join(map(str, bits))} = {code} (radius 1, default method)")
    r.source('''
lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11))''')
    return r


@op("glcm",
    desc="Gray Level Co-occurrence Matrix: how often two gray levels appear next to each other. Gives texture numbers.",
    formula="contrast = sum (i-j)^2 P(i,j)\nhomogeneity = sum P(i,j) / (1 + |i-j|)\nenergy = sqrt(sum P(i,j)^2)\ncorrelation = sum (i-mu_i)(j-mu_j) P / (sigma_i sigma_j)",
    params=[
        {"name": "d", "label": "Distance", "type": "slider", "min": 1, "max": 10, "step": 1, "default": 1},
        {"name": "levels", "label": "Gray levels", "type": "select", "default": "32", "options": [["8", "8"], ["16", "16"], ["32", "32"], ["64", "64"], ["256", "256"]]},
    ])
def glcm(img, p, ctx):
    from skimage.feature import graycomatrix, graycoprops
    g = to_gray(fit(img, 256))
    L = int(p["levels"])
    q = (g.astype(np.int32) * L // 256).astype(np.uint8)
    angs = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    m = graycomatrix(q, [int(p["d"])], angs, levels=L, symmetric=True, normed=True)
    names = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    rows = []
    for n in names:
        v = graycoprops(m, n)[0]
        rows.append([n] + [round(float(x), 4) for x in v] + [round(float(v.mean()), 4)])
    r = Result()
    r.image("Gray input", g, "", "gray")
    r.image("GLCM at 0 degrees", norm8(np.log1p(m[:, :, 0, 0] * 1e4)), "Brighter = this pair is more common", "glcm")
    r.table("Texture properties", rows, columns=["Property", "0", "45", "90", "135", "Mean"])
    r.chart("Mean value of each property", [{"name": "Mean", "color": AMBER, "values": [x[-1] for x in rows]}],
            labels=names, x_label="Property", y_label="Value", kind="bars")
    r.chart("Contrast by angle", [{"name": "Contrast", "color": BLUE, "values": rows[0][1:5]}],
            labels=["0", "45", "90", "135"], x_label="Angle (degrees)", y_label="Contrast", kind="bars")
    s = min(16, L)
    sm = m[:s, :s, 0, 0]
    r.table(f"Top-left {s} x {s} of the GLCM (x 1000)", (sm * 1000).round(1).tolist(), heat=(0, float(max(1e-9, sm.max() * 1000))),
            caption="Diagonal cells = a level next to itself (smooth areas).")
    r.stat("Matrix size", f"{L} x {L}")
    r.stat("Pairs counted", f"{(g.shape[0]) * (g.shape[1] - int(p['d'])):,}", "at 0 degrees")
    r.source('''
glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                    levels=256, symmetric=True, normed=True)
contrast = graycoprops(glcm, "contrast")''')
    return r


def _has(name):
    try:
        make_detector(name, 10)
        return True
    except Exception:
        return False


KP_OPTIONS = [[k, v] for k, v in (("sift", "SIFT"), ("orb", "ORB"), ("fast", "FAST"), ("brisk", "BRISK"), ("akaze", "AKAZE")) if _has(k)] + [["shi", "Shi-Tomasi"]]


@op("keypoints",
    desc="Local feature points with SIFT, ORB, FAST, BRISK or AKAZE. Compare how many each method finds.",
    formula="SIFT: extrema in a difference-of-Gaussians scale space\nORB: oriented FAST + rotated BRIEF\nFAST: 16-pixel circle test",
    params=[
        {"name": "m", "label": "Method", "type": "select", "default": "sift",
         "options": KP_OPTIONS},
        {"name": "n", "label": "Max points", "type": "slider", "min": 20, "max": 1000, "step": 10, "default": 300},
    ], heavy=True)
def keypoints(img, p, ctx):
    c = fit(img, 640)
    g = to_gray(c)
    n = int(p["n"])
    if p["m"] == "shi":
        pts = cv2.goodFeaturesToTrack(g, n, 0.01, 8)
        pts = [] if pts is None else pts.reshape(-1, 2)
        kps = [cv2.KeyPoint(float(x), float(y), 6) for x, y in pts]
    else:
        kps = list(make_detector(p["m"], n).detect(g, None))
    kps = sorted(kps, key=lambda k: -k.response)[:n] if p["m"] != "shi" else kps
    vis = cv2.drawKeypoints(c, kps, None, (51, 201, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    r = Result()
    r.big(len(kps), f"{p['m'].upper()} keypoints")
    r.image("Original", c, "", "original")
    r.image("Keypoints", vis, "Circle size = scale, line = orientation", "keypoints")
    cnts = {}
    for nm, _lab in KP_OPTIONS:
        if nm != "shi":
            cnts[nm.upper()] = len(make_detector(nm, 5000).detect(g, None))
    r.chart("Keypoints found by each method (no limit)", [{"name": "Keypoints", "color": AMBER, "values": list(cnts.values())}],
            labels=list(cnts.keys()), x_label="Method", y_label="Count", kind="bars")
    if kps:
        r.table("Strongest 10", [[i + 1, round(k.pt[0]), round(k.pt[1]), round(k.size, 1), round(k.angle, 1), round(k.response, 5)]
                                 for i, k in enumerate(kps[:10])], columns=["#", "x", "y", "size", "angle", "response"])
        r.chart("Keypoint size distribution", [{"name": "Count", "color": BLUE, "values": np.histogram([k.size for k in kps], 20)[0].tolist()}],
                x_label="Size (20 bins)", y_label="Count", kind="line")
    r.source('''
sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)
out = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)''')
    return r


@op("zoom_points",
    desc="Find the strongest Harris points and zoom into each one, like checking cells or tissue in a microscope image.",
    formula="zoomed crop = resize(img[y - z : y + z, x - z : x + z], 200 x 200, cubic)",
    params=[
        {"name": "n", "label": "How many", "type": "slider", "min": 1, "max": 8, "step": 1, "default": 5},
        {"name": "zoom", "label": "Crop radius", "type": "slider", "min": 15, "max": 150, "step": 5, "default": 50},
        {"name": "dist", "label": "Min distance", "type": "slider", "min": 10, "max": 150, "step": 5, "default": 50},
        {"name": "t", "label": "Threshold %", "type": "slider", "min": 0.1, "max": 20, "step": 0.1, "default": 1},
    ])
def zoom_points(img, p, ctx):
    c = img if max(img.shape[:2]) <= 1600 else fit(img, 1600)
    g = to_gray(c)
    _, hd = harris_resp(g)
    thr = p["t"] / 100 * hd.max()
    pts = pick_points(hd, thr, int(p["dist"]), int(p["n"]))
    r = Result()
    if not pts:
        r.note("No interest points found. Lower the threshold.")
        r.image("Original", c, "", "original")
        return r
    mk = c.copy()
    z = int(p["zoom"])
    t = max(2, max(c.shape[:2]) // 400)
    for i, (x, y, _) in enumerate(pts):
        cv2.circle(mk, (x, y), z, (0, 255, 0), t)
        cv2.drawMarker(mk, (x, y), (0, 0, 255), cv2.MARKER_CROSS, max(20, z // 2), t)
        cv2.putText(mk, str(i + 1), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, c.shape[1] / 900), (0, 255, 255), t)
    r.big(len(pts), "interest points zoomed")
    r.image("Detected points", mk, "Green circle = region zoomed below", "points")
    for i, (x, y, v) in enumerate(pts):
        y1, y2 = max(0, y - z), min(c.shape[0], y + z)
        x1, x2 = max(0, x - z), min(c.shape[1], x + z)
        crop = c[y1:y2, x1:x2]
        if crop.size:
            r.image(f"Zoom #{i + 1}", cv2.resize(crop, (240, 240), interpolation=cv2.INTER_CUBIC), f"x {x}, y {y}", f"zoom_{i + 1}")
    r.layout = "gallery"
    r.table("Points", [[i + 1, x, y, f"{v:.3e}"] for i, (x, y, v) in enumerate(pts)], columns=["#", "x", "y", "Harris R"])
    r.source('''
harris = cv2.dilate(cv2.cornerHarris(np.float32(gray), 2, 3, 0.04), None)
ys, xs = np.where(harris > 0.01 * harris.max())
crop = img[y-100:y+100, x-100:x+100]
zoomed = cv2.resize(crop, (200, 200), interpolation=cv2.INTER_CUBIC)''')
    return r
