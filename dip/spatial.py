"""Transform and spatial filters."""
import cv2
import numpy as np

from .registry import op
from .utils import Result, to_gray, hist256, num, fit, mse, psnr, ssim, norm8, crop_pct
from .color import add_noise

AMBER, GRAY, BLUE = "#ffc933", "#d8d8d8", "#4d8dff"


def lapvar(g):
    return float(cv2.Laplacian(to_gray(g), cv2.CV_64F).var())


@op("transform",
    desc="Crop, resize, rotate and flip: the geometric operations, all on your image at once.",
    formula="rotation: x' = x cos t - y sin t,  y' = x sin t + y cos t\nscale: x' = s x,  y' = s y",
    params=[
        {"name": "cx", "label": "Crop x %", "type": "slider", "min": 0, "max": 80, "step": 1, "default": 20},
        {"name": "cy", "label": "Crop y %", "type": "slider", "min": 0, "max": 80, "step": 1, "default": 20},
        {"name": "cw", "label": "Crop size %", "type": "slider", "min": 10, "max": 100, "step": 1, "default": 50},
        {"name": "scale", "label": "Scale", "type": "slider", "min": 0.1, "max": 2, "step": 0.05, "default": 0.5},
        {"name": "angle", "label": "Angle", "type": "slider", "min": -180, "max": 180, "step": 1, "default": 30},
    ])
def transform(img, p, ctx):
    c = fit(img, 640)
    H, W = c.shape[:2]
    crop, (x0, y0, x1, y1) = crop_pct(c, p["cx"], p["cy"], p["cw"], p["cw"])
    mark = c.copy()
    cv2.rectangle(mark, (x0, y0), (x1, y1), (51, 201, 255), max(2, W // 200))
    s = float(p["scale"])
    rs = cv2.resize(c, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    a = float(p["angle"])
    M = cv2.getRotationMatrix2D((W / 2, H / 2), a, 1.0)
    cs, sn = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(H * sn + W * cs), int(H * cs + W * sn)
    M[0, 2] += nw / 2 - W / 2; M[1, 2] += nh / 2 - H / 2
    rot = cv2.warpAffine(c, M, (nw, nh))
    r = Result().gallery()
    r.image("Original with crop box", mark, f"{W} x {H}", "original")
    r.image("Cropped", crop, f"{crop.shape[1]} x {crop.shape[0]}", "cropped")
    r.image(f"Resized x {s}", rs, f"{rs.shape[1]} x {rs.shape[0]}", "resized")
    r.image(f"Rotated {a:g} degrees", rot, f"{nw} x {nh}, canvas grows", "rotated")
    r.image("Rotate 90 clockwise", cv2.rotate(c, cv2.ROTATE_90_CLOCKWISE), "", "rot90")
    r.image("Rotate 180", cv2.rotate(c, cv2.ROTATE_180), "", "rot180")
    r.image("Rotate 270", cv2.rotate(c, cv2.ROTATE_90_COUNTERCLOCKWISE), "", "rot270")
    r.image("Flip horizontal", cv2.flip(c, 1), "mirror left to right", "flip_h")
    r.image("Flip vertical", cv2.flip(c, 0), "mirror top to bottom", "flip_v")
    r.image("Flip both", cv2.flip(c, -1), "same as rotate 180", "flip_both")
    r.table("Sizes", [["Original", W, H, W * H], ["Cropped", crop.shape[1], crop.shape[0], crop.shape[0] * crop.shape[1]],
                      ["Resized", rs.shape[1], rs.shape[0], rs.shape[0] * rs.shape[1]], ["Rotated", nw, nh, nw * nh]],
            columns=["Image", "Width", "Height", "Pixels"])
    r.table("Rotation matrix used", [[round(float(v), 4) for v in row] for row in M], caption="2 x 3 affine matrix from getRotationMatrix2D.")
    r.source('''
crop = img[y0:y1, x0:x1]
resized = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))
flipped = cv2.flip(img, 1)''')
    return r


@op("smoothing",
    desc="Blur filters that average neighbours. Mean and Gaussian blur everything, median and bilateral protect edges.",
    formula="mean kernel = 1/k^2 (all ones)\nGaussian: G(x, y) = exp(-(x^2 + y^2) / 2 sigma^2)\nmedian: middle value of the window",
    params=[{"name": "k", "label": "Kernel size", "type": "slider", "min": 3, "max": 21, "step": 2, "default": 5}])
def smoothing(img, p, ctx):
    c = fit(img, 640)
    k = int(p["k"])
    outs = {"Mean": cv2.blur(c, (k, k)), "Gaussian": cv2.GaussianBlur(c, (k, k), 0),
            "Median": cv2.medianBlur(c, k), "Bilateral": cv2.bilateralFilter(c, 9, 75, 75)}
    g = to_gray(c)
    r = Result().gallery()
    r.image("Original", c, "", "original")
    rows = [["Original", round(float(g.std()), 2), round(lapvar(g), 1), "-", "-"]]
    for n, o in outs.items():
        r.image(n, o, f"{k} x {k}" if n != "Bilateral" else "d = 9", n.lower())
        go = to_gray(o)
        rows.append([n, round(float(go.std()), 2), round(lapvar(go), 1), round(psnr(g, go), 2), round(ssim(g, go), 4)])
    r.table("What each filter did", rows, columns=["Filter", "Std", "Sharpness (var of Laplacian)", "PSNR dB", "SSIM"],
            caption="Smaller sharpness = more blur.")
    kk = min(k, 7)
    r.table(f"Mean kernel ({kk} x {kk}), each weight 1/{kk * kk}", np.full((kk, kk), round(1 / kk ** 2, 4)).tolist())
    gk = cv2.getGaussianKernel(kk, -1); G2 = gk @ gk.T
    r.table(f"Gaussian kernel ({kk} x {kk})", G2.round(4).tolist(), heat=(0, float(G2.max())), caption="Centre pixel counts the most.")
    r.chart("Gray histogram of each result", [{"name": "Original", "color": GRAY, "values": hist256(g)},
                                              {"name": "Gaussian", "color": AMBER, "values": hist256(to_gray(outs["Gaussian"]))},
                                              {"name": "Median", "color": BLUE, "values": hist256(to_gray(outs["Median"]))}])
    r.source('''
mean = cv2.blur(img, (k, k))
gauss = cv2.GaussianBlur(img, (k, k), 0)
median = cv2.medianBlur(img, k)
bilateral = cv2.bilateralFilter(img, 9, 75, 75)''')
    return r


@op("denoise",
    desc="Add noise to your image, then try five filters and see which one cleans it best.",
    formula="PSNR = 10 log10(255^2 / MSE)",
    params=[
        {"name": "kind", "label": "Noise", "type": "select", "default": "saltpepper",
         "options": [["saltpepper", "Salt and pepper"], ["gaussian", "Gaussian"], ["speckle", "Speckle"]]},
        {"name": "level", "label": "Strength", "type": "slider", "min": 1, "max": 50, "step": 1, "default": 10},
    ], heavy=True)
def denoise(img, p, ctx):
    g = fit(to_gray(img), 480)
    lv = int(p["level"])
    noisy = add_noise(g, p["kind"], 0, lv * 1.2, lv / 100.0, 1)
    outs = {"Mean 3 x 3": cv2.blur(noisy, (3, 3)), "Gaussian 5 x 5": cv2.GaussianBlur(noisy, (5, 5), 0),
            "Median 3 x 3": cv2.medianBlur(noisy, 3), "Bilateral": cv2.bilateralFilter(noisy, 7, 60, 60),
            "Non-local means": cv2.fastNlMeansDenoising(noisy, None, max(3, lv * 0.6), 7, 21)}
    r = Result().gallery()
    r.image("Clean original", g, "", "clean")
    r.image("Noisy", noisy, f"PSNR {psnr(g, noisy):.2f} dB", "noisy")
    rows = []
    for n, o in outs.items():
        r.image(n, o, f"PSNR {psnr(g, o):.2f} dB", n.split()[0].lower())
        rows.append([n, round(mse(g, o), 2), round(psnr(g, o), 2), round(ssim(g, o), 4)])
    best = max(range(len(rows)), key=lambda i: rows[i][2])
    r.stat("Best filter", rows[best][0], f"PSNR {rows[best][2]} dB")
    r.stat("Noisy PSNR", f"{psnr(g, noisy):.2f} dB")
    r.stat("Gain", f"+{rows[best][2] - psnr(g, noisy):.2f} dB")
    r.table("Filters ranked by quality", rows, columns=["Filter", "MSE", "PSNR dB", "SSIM"], highlight=best)
    r.chart("PSNR of each filter", [{"name": "PSNR dB", "color": AMBER, "values": [x[2] for x in rows]}],
            labels=[x[0].split()[0] for x in rows], x_label="Filter", y_label="dB", kind="bars")
    r.note("Median filters win against salt and pepper. Gaussian-type noise is handled better by Gaussian, bilateral or non-local means.")
    r.source('''
median = cv2.medianBlur(noisy, 3)
nlm = cv2.fastNlMeansDenoising(noisy, None, h=10)''')
    return r


@op("sharpen",
    desc="Make edges crisper: unsharp masking (original + detail) and a Laplacian sharpening kernel.",
    formula="unsharp: g = f + amount (f - blur(f))\nLaplacian kernel: g = f - laplacian(f)",
    params=[
        {"name": "amount", "label": "Amount", "type": "slider", "min": 0.2, "max": 4, "step": 0.1, "default": 1.5},
        {"name": "sigma", "label": "Blur sigma", "type": "slider", "min": 0.5, "max": 8, "step": 0.5, "default": 2},
    ])
def sharpen(img, p, ctx):
    c = fit(img, 800)
    a = float(p["amount"])
    bl = cv2.GaussianBlur(c, (0, 0), float(p["sigma"]))
    un = cv2.addWeighted(c, 1 + a, bl, -a, 0)
    K = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    lp = cv2.filter2D(c, -1, K)
    detail = cv2.subtract(c, bl)
    r = Result().gallery()
    r.image("Original", c, "", "original")
    r.image("Unsharp masking", un, f"amount {a}", "unsharp")
    r.image("Laplacian kernel", lp, "kernel below", "laplacian")
    r.image("Detail layer (x4)", np.clip(detail.astype(int) * 4, 0, 255).astype(np.uint8), "original minus blur", "detail")
    rows = [[n, round(lapvar(x), 1), round(float(to_gray(x).std()), 2)] for n, x in (("Original", c), ("Unsharp", un), ("Laplacian", lp))]
    r.table("Sharpness", rows, columns=["Image", "Var of Laplacian", "Std"], caption="Higher variance of the Laplacian = crisper edges.")
    r.table("Sharpening kernel", K.astype(int).tolist(), heat=(-1, 5), caption="Weights sum to 1, so brightness stays the same.")
    r.chart("Sharpness comparison", [{"name": "Var of Laplacian", "color": AMBER, "values": [x[1] for x in rows]}],
            labels=[x[0] for x in rows], x_label="Image", y_label="Variance", kind="bars")
    r.source('''
blur = cv2.GaussianBlur(img, (0, 0), sigma)
unsharp = cv2.addWeighted(img, 1 + amount, blur, -amount, 0)
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharp = cv2.filter2D(img, -1, kernel)''')
    return r
