"""Frequency domain."""
import cv2
import numpy as np

from .registry import op
from .utils import Result, to_gray, hist256, num, fit, norm8, psnr

AMBER, GRAY, BLUE, RED = "#ffc933", "#d8d8d8", "#4d8dff", "#ff4d4d"


def spectrum(g):
    F = np.fft.fftshift(np.fft.fft2(g.astype(np.float64)))
    return F


def logmag(F):
    return norm8(np.log1p(np.abs(F)))


def ifft(F):
    return np.real(np.fft.ifft2(np.fft.ifftshift(F)))


def dist_grid(shape):
    h, w = shape
    y, x = np.ogrid[:h, :w]
    return np.sqrt((y - h / 2) ** 2 + (x - w / 2) ** 2)


@op("fft",
    desc="Look at the image as a sum of waves. The center of the spectrum holds the slow changes, the edges hold fine detail.",
    formula="F(u, v) = sum f(x, y) e^(-j 2 pi (ux/M + vy/N))\nspectrum = log(1 + |F|)",
    params=[{"name": "r", "label": "Keep radius", "type": "slider", "min": 2, "max": 200, "step": 1, "default": 30}])
def fft(img, p, ctx):
    g = fit(to_gray(img), 512)
    F = spectrum(g)
    D = dist_grid(g.shape)
    rad = float(p["r"])
    low = F * (D <= rad)
    rec_low = np.clip(ifft(low), 0, 255).astype(np.uint8)
    mag_only = np.clip(ifft(np.abs(F)), 0, None)
    phase_only = ifft(np.exp(1j * np.angle(F)))
    ang = norm8(np.angle(F))
    r = Result().gallery()
    r.image("Gray input", g, "", "gray")
    r.image("Magnitude spectrum", logmag(F), "log(1 + |F|), center = DC", "spectrum")
    r.image("Phase", ang, "-pi to pi mapped to gray", "phase")
    r.image("Inverse FFT", np.clip(ifft(F), 0, 255).astype(np.uint8), "Back to the same image", "ifft")
    r.image(f"Only frequencies within {int(rad)}", rec_low, "Blurry: fine detail is gone", "lowpass_rec")
    r.image("Magnitude only", norm8(np.log1p(mag_only)), "Phase thrown away: structure lost", "mag_only")
    r.image("Phase only", norm8(phase_only), "Edges survive, so phase carries structure", "phase_only")
    tot = float((np.abs(F) ** 2).sum())
    rows = []
    for rr in (2, 5, 10, 20, 40, 80, 160):
        rows.append([rr, round(100 * float((np.abs(F[D <= rr]) ** 2).sum()) / tot, 3)])
    r.stat("Size", f"{g.shape[1]} x {g.shape[0]}")
    r.stat("DC term |F(0,0)|", f"{abs(F[g.shape[0] // 2, g.shape[1] // 2]):,.0f}", "sum of all pixels")
    r.stat(f"Energy within {int(rad)}", f"{100 * float((np.abs(low) ** 2).sum()) / tot:.2f} %")
    r.stat("PSNR of low-pass rebuild", f"{psnr(g, rec_low):.2f} dB")
    r.table("Energy inside a circle of radius r", rows, columns=["r (pixels)", "Energy %"],
            caption="Most of the energy lives near the center.")
    mx = int(D.max())
    prof = [float(np.log1p(np.abs(F[(D >= i) & (D < i + 1)]).mean())) for i in range(0, mx)]
    r.chart("Radial profile of the spectrum", [{"name": "log magnitude", "color": AMBER, "values": [round(v, 3) for v in prof]}],
            x_label="Distance from center (frequency)", y_label="log magnitude", marker=[[int(rad), f"r = {int(rad)}"]])
    r.source('''
f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)
magnitude = 20 * np.log(np.abs(fshift) + 1)
back = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift)))''')
    return r


def make_mask(kind, band, shape, d0, n, width):
    D = dist_grid(shape)
    D = np.where(D == 0, 1e-6, D)
    if kind == "ideal":
        lp = (D <= d0).astype(np.float64)
    elif kind == "butter":
        lp = 1.0 / (1.0 + (D / d0) ** (2 * n))
    else:
        lp = np.exp(-(D ** 2) / (2 * d0 ** 2))
    if band == "low":
        return lp
    if band == "high":
        return 1 - lp
    d1 = d0 + width
    if kind == "ideal":
        hi = (D <= d1).astype(np.float64)
    elif kind == "butter":
        hi = 1.0 / (1.0 + (D / d1) ** (2 * n))
    else:
        hi = np.exp(-(D ** 2) / (2 * d1 ** 2))
    return hi - lp


@op("freq_filter",
    desc="Filter in the frequency domain. Low-pass blurs, high-pass keeps edges, band-pass keeps a ring. Ideal, Butterworth or Gaussian.",
    formula="ideal low-pass: H = 1 if D <= D0 else 0\nButterworth: H = 1 / (1 + (D / D0)^(2n))\nGaussian: H = exp(-D^2 / 2 D0^2)\nhigh-pass = 1 - low-pass",
    params=[
        {"name": "kind", "label": "Filter", "type": "select", "default": "ideal", "options": [["ideal", "Ideal"], ["butter", "Butterworth"], ["gauss", "Gaussian"]]},
        {"name": "band", "label": "Type", "type": "select", "default": "low", "options": [["low", "Low-pass"], ["high", "High-pass"], ["band", "Band-pass"]]},
        {"name": "d0", "label": "D0", "type": "slider", "min": 2, "max": 200, "step": 1, "default": 40},
        {"name": "n", "label": "Order n", "type": "slider", "min": 1, "max": 10, "step": 1, "default": 2},
        {"name": "w", "label": "Band width", "type": "slider", "min": 5, "max": 100, "step": 1, "default": 30},
    ])
def freq_filter(img, p, ctx):
    g = fit(to_gray(img), 512)
    F = spectrum(g)
    H = make_mask(p["kind"], p["band"], g.shape, float(p["d0"]), int(p["n"]), float(p["w"]))
    G = F * H
    out = ifft(G)
    out8 = np.clip(out if p["band"] != "high" and p["band"] != "band" else out + (128 if False else 0), 0, 255).astype(np.uint8)
    if p["band"] in ("high", "band"):
        out8 = norm8(np.abs(out))
    r = Result().gallery()
    r.image("Gray input", g, "", "gray")
    r.image("Input spectrum", logmag(F), "", "spectrum")
    r.image("Filter H(u, v)", (H * 255).clip(0, 255).astype(np.uint8), "white = pass", "filter")
    r.image("Filtered spectrum", logmag(G), "", "filtered_spectrum")
    r.image("Result", out8, "", "result")
    tot = float((np.abs(F) ** 2).sum())
    r.stat("Energy kept", f"{100 * float((np.abs(G) ** 2).sum()) / tot:.2f} %")
    r.stat("D0", int(p["d0"]), "cutoff radius")
    r.stat("Std before", num(g.std(), 2))
    r.stat("Std after", num(out8.std(), 2))
    mid = H[H.shape[0] // 2, H.shape[1] // 2:]
    r.chart("Filter profile along a row from the center", [{"name": "H", "color": AMBER, "values": [round(float(v), 4) for v in mid]}],
            x_label="Distance D", y_label="H(D)")
    if p["kind"] == "ideal":
        r.note("Ideal filters have a sharp cutoff and cause ringing (wave-like ripples) near edges. Try Butterworth or Gaussian to remove them.")
    r.source('''
fshift = np.fft.fftshift(np.fft.fft2(gray))
mask = np.zeros_like(gray); cv2.circle(mask, (cx, cy), D0, 1, -1)      # ideal low-pass
filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * mask)))''')
    return r
