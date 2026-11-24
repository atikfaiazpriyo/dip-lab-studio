"""Compression: JPEG and DCT."""
import cv2
import numpy as np

from .registry import op
from .utils import Result, to_gray, hist256, num, fit, norm8, psnr, ssim

AMBER, GRAY, BLUE = "#ffc933", "#d8d8d8", "#4d8dff"

JPEG_Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], np.float32)


def zigzag(n=8):
    idx = sorted(((i, j) for i in range(n) for j in range(n)), key=lambda t: (t[0] + t[1], t[1] if (t[0] + t[1]) % 2 == 0 else t[0]))
    return idx


@op("jpeg",
    desc="Save the image as JPEG at a chosen quality and see how file size, quality and visible blocks change.",
    formula="compression ratio = raw size / compressed size\nPSNR = 10 log10(255^2 / MSE)",
    params=[{"name": "q", "label": "JPEG quality", "type": "slider", "min": 1, "max": 100, "step": 1, "default": 20}], heavy=True)
def jpeg(img, p, ctx):
    c = fit(img, 800)
    raw = c.size

    def enc(q):
        ok, b = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
        return len(b), cv2.imdecode(b, cv2.IMREAD_COLOR)
    q = int(p["q"])
    nb, dec = enc(q)
    ok, png = cv2.imencode(".png", c)
    diff = np.abs(c.astype(int) - dec.astype(int))
    r = Result()
    r.image("Original", c, f"{raw / 1024:,.0f} KB raw", "original")
    r.image(f"JPEG quality {q}", dec, f"{nb / 1024:,.1f} KB", "jpeg")
    r.image("Difference x 10", np.clip(diff * 10, 0, 255).astype(np.uint8), "Black = identical", "difference")
    r.stat("Compressed size", f"{nb / 1024:,.1f} KB")
    r.stat("Compression ratio", f"{raw / nb:.1f} : 1")
    r.stat("Space saved", f"{100 * (1 - nb / raw):.1f} %")
    r.stat("PSNR", f"{psnr(c, dec):.2f} dB")
    r.stat("SSIM", f"{ssim(c, dec):.4f}")
    r.stat("PNG (lossless) size", f"{len(png) / 1024:,.1f} KB", "for comparison")
    qs = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    data = [enc(x) for x in qs]
    r.chart("File size by quality", [{"name": "KB", "color": AMBER, "values": [round(d[0] / 1024, 1) for d in data]}],
            labels=[str(x) for x in qs], x_label="Quality", y_label="KB", kind="line")
    r.chart("PSNR by quality", [{"name": "PSNR dB", "color": BLUE, "values": [round(psnr(c, d[1]), 2) for d in data]}],
            labels=[str(x) for x in qs], x_label="Quality", y_label="dB", kind="line")
    r.table("Quality sweep", [[x, round(d[0] / 1024, 1), round(raw / d[0], 1), round(psnr(c, d[1]), 2), round(ssim(c, d[1]), 4)] for x, d in zip(qs, data)],
            columns=["Quality", "KB", "Ratio", "PSNR dB", "SSIM"])
    r.note("At low quality you can see 8 x 8 blocks. JPEG works on 8 x 8 blocks with a DCT.")
    r.source('''
ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 20])
decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
ratio = img.nbytes / len(buf)''')
    return r


@op("dct",
    desc="The core of JPEG: cut the image into 8 x 8 blocks, apply the DCT and keep only the strongest coefficients.",
    formula="DCT turns an 8 x 8 block into 64 frequency coefficients\nkeep the first K in zigzag order, set the rest to 0, then inverse DCT",
    params=[
        {"name": "k", "label": "Coefficients kept (of 64)", "type": "slider", "min": 1, "max": 64, "step": 1, "default": 10},
        {"name": "bx", "label": "Example block x", "type": "slider", "min": 0, "max": 100, "step": 1, "default": 40},
        {"name": "by", "label": "Example block y", "type": "slider", "min": 0, "max": 100, "step": 1, "default": 40},
    ], heavy=True)
def dct(img, p, ctx):
    g = fit(to_gray(img), 512)
    H, W = (g.shape[0] // 8) * 8, (g.shape[1] // 8) * 8
    g = g[:H, :W]
    k = int(p["k"])
    zz = zigzag()
    keep = np.zeros((8, 8), np.float32)
    for (i, j) in zz[:k]:
        keep[i, j] = 1
    f = g.astype(np.float32) - 128
    out = np.zeros_like(f)
    energy = np.zeros(64)
    for y in range(0, H, 8):
        for x in range(0, W, 8):
            d = cv2.dct(f[y:y + 8, x:x + 8])
            energy += np.array([d[i, j] ** 2 for i, j in zz])
            out[y:y + 8, x:x + 8] = cv2.idct(d * keep)
    rec = np.clip(out + 128, 0, 255).astype(np.uint8)
    by = min(H // 8 - 1, int((H // 8 - 1) * p["by"] / 100)); bx = min(W // 8 - 1, int((W // 8 - 1) * p["bx"] / 100))
    blk = g[by * 8:by * 8 + 8, bx * 8:bx * 8 + 8]
    d = cv2.dct(blk.astype(np.float32) - 128)
    mark = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(mark, (bx * 8 - 2, by * 8 - 2), (bx * 8 + 10, by * 8 + 10), (51, 201, 255), 2)
    r = Result()
    r.image("Gray input", mark, "Amber box = example block", "gray")
    r.image(f"Rebuilt from {k} of 64 coefficients", rec, "", "dct_rebuilt")
    r.image("Error x 5", np.clip(np.abs(g.astype(int) - rec.astype(int)) * 5, 0, 255).astype(np.uint8), "", "error")
    r.stat("Coefficients kept", f"{k} / 64", f"{100 * k / 64:.1f} %")
    r.stat("Ideal ratio", f"{64 / k:.1f} : 1")
    r.stat("PSNR", f"{psnr(g, rec):.2f} dB")
    r.stat("SSIM", f"{ssim(g, rec):.4f}")
    r.table("Example block, pixels", blk.tolist(), heat=(0, 255))
    r.table("Example block, DCT coefficients", d.round(0).astype(int).tolist(), caption="Top-left is the DC term. Large values gather there.")
    r.table("Kept coefficients (1 = kept)", keep.astype(int).tolist(), heat=(0, 1), caption="Chosen in zigzag order, low frequencies first.")
    r.table("JPEG luminance quantization table", JPEG_Q.astype(int).tolist(), heat=(0, 121), caption="Standard table at quality 50. Larger numbers = coarser steps.")
    cum = np.cumsum(energy) / max(float(energy.sum()), 1e-9)
    r.chart("Energy compaction: cumulative energy by zigzag index", [{"name": "Energy", "color": AMBER, "values": [round(float(v), 5) for v in cum]}],
            x_label="Coefficients kept (zigzag order)", y_label="Share of energy", marker=[[k - 1, f"K = {k}"]])
    r.source('''
block = gray[y:y+8, x:x+8].astype(np.float32) - 128
coef = cv2.dct(block)
coef *= keep_mask                    # keep first K zigzag coefficients
rebuilt = cv2.idct(coef) + 128''')
    return r
