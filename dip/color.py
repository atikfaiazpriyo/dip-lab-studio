"""Noise models and color spaces."""
import math

import cv2
import numpy as np

from .registry import op
from .utils import Result, to_gray, hist256, num, fit, mse, psnr, ssim, norm8


# ------------------------------------------------------------------ noise
def add_noise(img, kind, mean, std, amount, seed):
    rng = np.random.default_rng(int(seed))
    f = img.astype(np.float64)
    if kind == "gaussian":
        out = f + rng.normal(mean, std, f.shape)
    elif kind == "saltpepper":
        out = f.copy()
        m = rng.random(f.shape[:2])
        out[m < amount / 2] = 0
        out[(m >= amount / 2) & (m < amount)] = 255
    elif kind == "speckle":
        out = f + f * rng.normal(mean / 255.0, std / 100.0, f.shape)
    else:  # poisson
        out = rng.poisson(np.clip(f, 0, 255) / 255.0 * 30.0) / 30.0 * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


@op("gaussian_noise",
    desc="Add Gaussian, salt and pepper, speckle or Poisson noise, then measure how much the image got damaged.",
    formula="Gaussian: g = f + n,  n ~ N(mean, std^2)\nSalt and pepper: random pixels set to 0 or 255\nSpeckle: g = f + f n\nPoisson: pixel value drawn from Poisson(f)",
    params=[
        {"name": "kind", "label": "Noise", "type": "select", "default": "gaussian",
         "options": [["gaussian", "Gaussian"], ["saltpepper", "Salt and pepper"], ["speckle", "Speckle"], ["poisson", "Poisson"]]},
        {"name": "mean", "label": "Mean", "type": "slider", "min": -50, "max": 50, "step": 1, "default": 0},
        {"name": "std", "label": "Std", "type": "slider", "min": 1, "max": 100, "step": 1, "default": 25},
        {"name": "amount", "label": "Amount (S&P)", "type": "slider", "min": 0.01, "max": 0.3, "step": 0.01, "default": 0.05},
        {"name": "seed", "label": "Seed", "type": "slider", "min": 0, "max": 99, "step": 1, "default": 1},
    ])
def gaussian_noise(img, p, ctx):
    g = fit(to_gray(img), 640)
    c = fit(img, 640)
    noisy = add_noise(c, p["kind"], p["mean"], p["std"], p["amount"], p["seed"])
    ng = to_gray(noisy)
    diff = ng.astype(np.int16) - g.astype(np.int16)
    r = Result()
    r.image("Original", c, "", "original")
    r.image("Noisy", noisy, "", "noisy")
    r.image("Noise only (stretched)", norm8(diff), "Difference noisy minus original", "noise_only")
    r.stat("MSE", num(mse(g, ng), 2))
    r.stat("PSNR", f"{psnr(g, ng):.2f} dB")
    r.stat("SSIM", f"{ssim(g, ng):.4f}", "1 = identical")
    r.stat("Noise mean", num(diff.mean(), 3))
    r.stat("Noise std", num(diff.std(), 3))
    hist = np.bincount((diff + 255).ravel(), minlength=511)
    nz = np.nonzero(hist)[0]
    lo, hi = max(0, int(nz.min()) - 2), min(510, int(nz.max()) + 2)
    r.chart("Noise histogram", [{"name": "Noise", "color": "#ffc933", "values": hist[lo:hi + 1].astype(int).tolist()}],
            x_label=f"Noise value (index 0 = {lo - 255})", y_label="Pixels")
    r.chart("Gray histogram, before and after",
            [{"name": "Original", "color": "#d8d8d8", "values": hist256(g)}, {"name": "Noisy", "color": "#ff4d4d", "values": hist256(ng)}])
    r.table("Top-left 3 x 3, original", g[:3, :3].tolist(), heat=(0, 255))
    r.table("Top-left 3 x 3, noise", diff[:3, :3].tolist(), heat=(-60, 60), caption="Positive = brighter, negative = darker.")
    r.table("Top-left 3 x 3, noisy", ng[:3, :3].tolist(), heat=(0, 255), caption="Original + noise, clipped to 0..255.")
    r.source('''
noise = np.random.normal(mean, std, img.shape)
noisy = np.clip(img + noise, 0, 255).astype(np.uint8)''')
    return r


# ------------------------------------------------------------------ color spaces
SPACES = {
    "hsv": ("HSV", cv2.COLOR_BGR2HSV, ["Hue", "Saturation", "Value"]),
    "hls": ("HLS", cv2.COLOR_BGR2HLS, ["Hue", "Lightness", "Saturation"]),
    "lab": ("CIE Lab", cv2.COLOR_BGR2LAB, ["L (lightness)", "a (green to red)", "b (blue to yellow)"]),
    "ycrcb": ("YCrCb", cv2.COLOR_BGR2YCrCb, ["Y (luma)", "Cr (red diff)", "Cb (blue diff)"]),
    "xyz": ("CIE XYZ", cv2.COLOR_BGR2XYZ, ["X", "Y", "Z"]),
    "luv": ("CIE Luv", cv2.COLOR_BGR2LUV, ["L", "u", "v"]),
    "rgb": ("RGB", None, ["Red", "Green", "Blue"]),
}


@op("color_spaces",
    desc="The same picture described in different color systems. Each channel is shown as its own gray image.",
    formula="Gray = 0.299 R + 0.587 G + 0.114 B\nV = max(R, G, B)\nS = (max - min) / max",
    params=[{"name": "space", "label": "Color space", "type": "select", "default": "hsv",
             "options": [[k, v[0]] for k, v in SPACES.items()]}])
def color_spaces(img, p, ctx):
    c = fit(img, 640)
    name, code, chans = SPACES[p["space"]]
    conv = c[:, :, ::-1].copy() if code is None else cv2.cvtColor(c, code)
    r = Result().gallery()
    r.image("Original", c, "", "original")
    r.image("Grayscale", to_gray(c), "0.299 R + 0.587 G + 0.114 B", "gray")
    for i, ch in enumerate(chans):
        r.image(ch, conv[:, :, i], f"{name} channel {i}", f"{p['space']}_{i}")
    r.chart(f"{name} channel histograms", [{"name": chans[i], "color": col, "values": hist256(conv[:, :, i])}
                                            for i, col in enumerate(["#ff4d4d", "#3ddc84", "#4d8dff"])])
    rows = [[chans[i], int(conv[:, :, i].min()), int(conv[:, :, i].max()), num(conv[:, :, i].mean(), 2)] for i in range(3)]
    r.table(f"{name} statistics", rows, columns=["Channel", "Min", "Max", "Mean"])
    h, w = c.shape[:2]
    px = c[h // 2, w // 2].reshape(1, 1, 3)
    r.table("Centre pixel in every space", [
        ["RGB", int(px[0, 0, 2]), int(px[0, 0, 1]), int(px[0, 0, 0])],
        ["HSV (OpenCV H is 0..179)", *cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0].tolist()],
        ["Lab (OpenCV 8-bit)", *cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].tolist()],
        ["YCrCb", *cv2.cvtColor(px, cv2.COLOR_BGR2YCrCb)[0, 0].tolist()],
    ], columns=["Space", "1st", "2nd", "3rd"])
    r.note("OpenCV stores Hue as degrees / 2 so that it fits in 8 bits (0 to 179).")
    r.source('''
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)''')
    return r


# ------------------------------------------------------------------ HSI
def rgb_to_hsi(rgb):
    """rgb float 0..1, shape (...,3) in R,G,B order. Returns H (deg), S, I."""
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    I = (R + G + B) / 3.0
    s_sum = R + G + B
    S = 1 - 3 * np.minimum(np.minimum(R, G), B) / np.where(s_sum == 0, 1, s_sum)
    S = np.where(s_sum == 0, 0, S)
    num_ = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
    th = np.degrees(np.arccos(np.clip(num_ / np.where(den == 0, 1, den), -1, 1)))
    H = np.where(B > G, 360 - th, th)
    H = np.where(den == 0, 0, H)
    return H, S, I


def hsi_to_rgb_arr(H, S, I):
    H = np.atleast_1d(np.asarray(H, np.float64)) % 360
    S = np.atleast_1d(np.asarray(S, np.float64)); I = np.atleast_1d(np.asarray(I, np.float64))
    R = np.zeros_like(H); G = np.zeros_like(H); B = np.zeros_like(H)
    cs = lambda d: np.cos(np.radians(d))
    m = H < 120
    h = H[m]
    B[m] = I[m] * (1 - S[m]); R[m] = I[m] * (1 + S[m] * cs(h) / cs(60 - h)); G[m] = 3 * I[m] - (R[m] + B[m])
    m = (H >= 120) & (H < 240)
    h = H[m] - 120
    R[m] = I[m] * (1 - S[m]); G[m] = I[m] * (1 + S[m] * cs(h) / cs(60 - h)); B[m] = 3 * I[m] - (R[m] + G[m])
    m = H >= 240
    h = H[m] - 240
    G[m] = I[m] * (1 - S[m]); B[m] = I[m] * (1 + S[m] * cs(h) / cs(60 - h)); R[m] = 3 * I[m] - (G[m] + B[m])
    return np.stack([R, G, B], -1)


@op("hsi",
    desc="Convert RGB to Hue, Saturation and Intensity using the lab formulas and see each component.",
    formula="I = (R + G + B) / 3\nS = 1 - 3 min(R, G, B) / (R + G + B)\nH = acos( 0.5((R-G)+(R-B)) / sqrt((R-G)^2 + (R-B)(G-B)) ),  H = 360 - H if B > G")
def hsi(img, p, ctx):
    c = fit(img, 640)
    rgb = c[:, :, ::-1].astype(np.float64) / 255
    H, S, I = rgb_to_hsi(rgb)
    r = Result().gallery()
    r.image("Original", c, "", "original")
    r.image("Hue", norm8(H / 360 * 255), "0 to 360 degrees mapped to gray", "hue")
    r.image("Saturation", (S * 255).astype(np.uint8), "0 = gray, 1 = pure color", "saturation")
    r.image("Intensity", (I * 255).astype(np.uint8), "Mean of R, G, B", "intensity")
    r.stat("Mean hue", f"{H.mean():.1f} deg")
    r.stat("Mean saturation", f"{S.mean():.3f}")
    r.stat("Mean intensity", f"{I.mean():.3f}")
    hh, ww = H.shape
    cy, cx = hh // 2, ww // 2
    r.table("Centre pixel", [[int(c[cy, cx, 2]), int(c[cy, cx, 1]), int(c[cy, cx, 0]),
                              round(float(H[cy, cx]), 2), round(float(S[cy, cx]), 4), round(float(I[cy, cx]), 4)]],
            columns=["R", "G", "B", "H deg", "S", "I"], caption=f"Pixel at x={cx}, y={cy}.")
    hist, _ = np.histogram(H, bins=36, range=(0, 360))
    r.chart("Hue histogram (10 degree bins)", [{"name": "Pixels", "color": "#ffc933", "values": hist.tolist()}],
            labels=[str(i * 10) for i in range(36)], x_label="Hue (degrees)", y_label="Pixels", kind="bars")
    r.chart("Saturation and intensity histograms", [
        {"name": "Saturation", "color": "#4d8dff", "values": np.histogram(S, 64, (0, 1))[0].tolist()},
        {"name": "Intensity", "color": "#d8d8d8", "values": np.histogram(I, 64, (0, 1))[0].tolist()}],
        x_label="Value (0 to 1, 64 bins)")
    r.source('''
R, G, B = r/255, g/255, b/255
I = (R + G + B) / 3
S = 1 - 3 * min(R, G, B) / (R + G + B)
theta = np.degrees(np.arccos(0.5*((R-G)+(R-B)) / np.sqrt((R-G)**2 + (R-B)*(G-B))))
H = theta if B <= G else 360 - theta''')
    return r


@op("hsi_to_rgb",
    desc="Go back from HSI to RGB with the sector formulas. Check the round trip and try your own H, S, I.",
    formula="0 <= H < 120:  B = I(1-S),  R = I[1 + S cosH / cos(60-H)],  G = 3I - (R+B)\n120 <= H < 240:  R = I(1-S),  G = I[1 + S cos(H-120) / cos(180-H)],  B = 3I - (R+G)\n240 <= H < 360:  G = I(1-S),  B = I[1 + S cos(H-240) / cos(300-H)],  R = 3I - (G+B)",
    params=[
        {"name": "H", "label": "H deg", "type": "slider", "min": 0, "max": 359, "step": 1, "default": 60},
        {"name": "S", "label": "S", "type": "slider", "min": 0, "max": 1, "step": 0.01, "default": 0.7},
        {"name": "I", "label": "I", "type": "slider", "min": 0, "max": 1, "step": 0.01, "default": 0.8},
    ])
def hsi_to_rgb(img, p, ctx):
    c = fit(img, 480)
    rgb = c[:, :, ::-1].astype(np.float64) / 255
    H, S, I = rgb_to_hsi(rgb)
    back = np.clip(hsi_to_rgb_arr(H.ravel(), S.ravel(), I.ravel()).reshape(rgb.shape), 0, 1)
    back8 = (back * 255).round().astype(np.uint8)[:, :, ::-1].copy()
    err = np.abs(back8.astype(int) - c.astype(int))
    r = Result().gallery()
    r.image("Original", c, "", "original")
    r.image("RGB to HSI to RGB", back8, "Round trip", "roundtrip")
    r.image("Absolute error x 20", np.clip(err * 20, 0, 255).astype(np.uint8), "Black = perfect", "error")
    r.stat("Max error", int(err.max()), "gray levels")
    r.stat("PSNR", f"{psnr(c, back8):.2f} dB")
    raw = hsi_to_rgb_arr(p["H"], p["S"], p["I"])[0]
    clipped = np.clip(raw, 0, 1)
    r.palette("Your H, S, I as a color", [[round(v * 255) for v in clipped]],
              caption=f"H = {p['H']} deg, S = {p['S']}, I = {p['I']}", values=["after clipping"])
    r.table("Calculator result", [[n, round(float(raw[i]), 4), round(float(clipped[i]) * 255)] for i, n in enumerate("RGB")],
            columns=["Channel", "Formula value", "8-bit after clipping"],
            caption="A value above 1 means the color is outside the RGB range, so it is clipped.")
    r.source('''
# sector 0 <= H < 120
B = I * (1 - S)
R = I * (1 + S * cos(H) / cos(60 - H))
G = 3 * I - (R + B)''')
    return r


# ------------------------------------------------------------------ pixel calculator
def _lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


@op("pixel_calc",
    desc="Pick one RGB pixel and see its HSV, YCbCr, XYZ, Lab and HSI values worked out by the lab formulas.",
    formula="Y = 0.299 R + 0.587 G + 0.114 B\nCb = 128 - 0.1687 R - 0.3313 G + 0.5 B\nCr = 128 + 0.5 R - 0.4187 G - 0.0813 B",
    params=[
        {"name": "R", "label": "R", "type": "slider", "min": 0, "max": 255, "step": 1, "default": 100},
        {"name": "G", "label": "G", "type": "slider", "min": 0, "max": 255, "step": 1, "default": 150},
        {"name": "B", "label": "B", "type": "slider", "min": 0, "max": 255, "step": 1, "default": 200},
        {"name": "use_image", "label": "Use centre pixel of my image", "type": "checkbox", "default": False},
    ])
def pixel_calc(img, p, ctx):
    if p["use_image"]:
        h, w = img.shape[:2]
        R, G, B = [int(v) for v in img[h // 2, w // 2][::-1]]
    else:
        R, G, B = int(p["R"]), int(p["G"]), int(p["B"])
    r_, g_, b_ = R / 255, G / 255, B / 255
    mx, mn = max(r_, g_, b_), min(r_, g_, b_)
    d = mx - mn
    if d == 0:
        Hh = 0.0
    elif mx == r_:
        Hh = (60 * ((g_ - b_) / d)) % 360
    elif mx == g_:
        Hh = 60 * ((b_ - r_) / d) + 120
    else:
        Hh = 60 * ((r_ - g_) / d) + 240
    Sv = 0 if mx == 0 else d / mx
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    X, Yx, Z = M @ np.array([r_, g_, b_])
    Xl, Yl, Zl = M @ _lin(np.array([r_, g_, b_]))
    wt = np.array([0.95047, 1.0, 1.08883])
    f = lambda t: np.where(t > 216 / 24389, np.cbrt(t), (24389 / 27 * t + 16) / 116)
    fx, fy, fz = f(Xl / wt[0]), f(Yl / wt[1]), f(Zl / wt[2])
    L, a, bb = 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)
    Hi, Si, Ii = rgb_to_hsi(np.array([r_, g_, b_]))
    cvhsv = cv2.cvtColor(np.uint8([[[B, G, R]]]), cv2.COLOR_BGR2HSV)[0, 0]
    cvyc = cv2.cvtColor(np.uint8([[[B, G, R]]]), cv2.COLOR_BGR2YCrCb)[0, 0]
    cvlab = cv2.cvtColor(np.float32([[[b_, g_, r_]]]), cv2.COLOR_BGR2Lab)[0, 0]
    r = Result()
    r.palette("Your pixel", [[R, G, B]], values=[f"RGB ({R}, {G}, {B})"])
    r.table("Results", [
        ["HSV", f"H = {Hh:.1f} deg", f"S = {Sv:.4f}", f"V = {mx:.4f}"],
        ["HSI", f"H = {float(Hi):.1f} deg", f"S = {float(Si):.4f}", f"I = {float(Ii):.4f}"],
        ["YCbCr (full range)", f"Y = {Y:.2f}", f"Cb = {Cb:.2f}", f"Cr = {Cr:.2f}"],
        ["XYZ (matrix on R, G, B)", f"X = {X:.4f}", f"Y = {Yx:.4f}", f"Z = {Z:.4f}"],
        ["XYZ (after gamma removal)", f"X = {Xl:.4f}", f"Y = {Yl:.4f}", f"Z = {Zl:.4f}"],
        ["CIE Lab (D65)", f"L = {L:.2f}", f"a = {a:.2f}", f"b = {bb:.2f}"],
    ], columns=["Space", "1st", "2nd", "3rd"])
    r.table("OpenCV cross-check", [
        ["HSV (H x 2, S and V x 255)", f"H = {int(cvhsv[0]) * 2} deg", f"S = {cvhsv[1]}", f"V = {cvhsv[2]}"],
        ["YCrCb", f"Y = {cvyc[0]}", f"Cr = {cvyc[1]}", f"Cb = {cvyc[2]}"],
        ["Lab (float input)", f"L = {cvlab[0]:.2f}", f"a = {cvlab[1]:.2f}", f"b = {cvlab[2]:.2f}"],
    ], columns=["Space", "1st", "2nd", "3rd"],
        caption="OpenCV rounds to integers for 8-bit images, so tiny differences are normal.")
    r.stat("Gray value", f"{Y:.2f}", "same as Y")
    r.stat("Brightest channel", "RGB"[int(np.argmax([R, G, B]))])
    r.note("Move the sliders: gray colors have S = 0, and Hue only matters when saturation is above 0.")
    r.source('''
H, S, V = colorsys.rgb_to_hsv(R/255, G/255, B/255)
Y  = 0.299*R + 0.587*G + 0.114*B
Cb = 128 - 0.168736*R - 0.331264*G + 0.5*B
Cr = 128 + 0.5*R - 0.418688*G - 0.081312*B''')
    return r
