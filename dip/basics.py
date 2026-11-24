"""Step 1 - Basics (Lab 1: Module 1.2, mean/variance/std, binary image)."""
import cv2
import numpy as np

from .registry import op
from .utils import Result, to_gray, hist256, num

RGB_COL = {"R": "#ff4d4d", "G": "#3ddc84", "B": "#4d8dff", "Gray": "#d8d8d8"}


@op("info",
    desc="What OpenCV sees after cv2.imread: size, channels, data type and the raw top-left pixels.")
def info(img, p, ctx):
    h, w, c = img.shape
    g = to_gray(img)
    r = Result()
    r.image("Original", img, file="original")
    r.stat("Shape", f"({h}, {w}, {c})", "img.shape = (rows, cols, channels)")
    r.stat("Width x Height", f"{w} x {h}", "pixels")
    r.stat("Channels", c, "B, G, R order in OpenCV")
    r.stat("Data type", str(img.dtype), "8 bits per channel, 0 to 255")
    r.stat("Total pixels", f"{h * w:,}", "M x N")
    r.stat("Values stored", f"{img.size:,}", "M x N x 3")
    r.stat("Memory", f"{img.nbytes / 1024:,.0f} KB", "img.nbytes")
    r.stat("File size", f"{ctx['nbytes'] / 1024:,.0f} KB", ctx["name"])
    r.table("Top-left 5 x 5 pixels (gray)", g[:5, :5].tolist(), heat=(0, 255),
            caption="Each cell is one pixel. Brighter cell = larger value.")
    r.table("Top-left pixel in each channel",
            [[int(img[0, 0, 2]), int(img[0, 0, 1]), int(img[0, 0, 0])]],
            columns=["R", "G", "B"],
            caption="OpenCV stores it as [B, G, R] = " + str(img[0, 0].tolist()))
    r.source('''
import cv2
img = cv2.imread(image_path)
print("Image loaded successfully.")
print("Shape:", img.shape)      # (rows, cols, channels)
print("Data type:", img.dtype)  # uint8''')
    return r


@op("display",
    desc="OpenCV reads BGR, Matplotlib shows RGB. See what happens without cvtColor, then the grayscale version.")
def display(img, p, ctx):
    r = Result()
    gray = to_gray(img)
    r.image("RGB (after BGR2RGB)", img, "Correct colors", "rgb")
    r.image("Without conversion", img[:, :, ::-1], "Red and blue swapped, the classic lab mistake", "bgr_wrong")
    r.image("Grayscale", gray, "cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)", "gray")
    r.stat("Gray formula", "0.299 R + 0.587 G + 0.114 B", "weights used by OpenCV")
    r.stat("Gray shape", str(gray.shape), "one channel only")
    r.note("Grayscale has one channel, so it needs 3 times less memory than the color image.")
    r.source('''
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb); plt.axis("off"); plt.show()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap="gray")''')
    return r


@op("negative",
    desc="Flip every pixel around the middle of the intensity range. Dark becomes bright.",
    formula="g(x, y) = L - 1 - f(x, y) = 255 - f(x, y)",
    params=[{"name": "mode", "label": "Image", "type": "select", "default": "color",
             "options": [["color", "Color"], ["gray", "Grayscale"]]}])
def negative(img, p, ctx):
    src = to_gray(img) if p["mode"] == "gray" else img
    neg = 255 - src
    g0, g1 = to_gray(img), to_gray(neg if neg.ndim == 3 else cv2.merge([neg] * 3))
    r = Result()
    r.image("Original", src, file="original")
    r.image("Negative", neg, "255 - img", "negative")
    r.stat("Mean before", num(g0.mean(), 2))
    r.stat("Mean after", num(g1.mean(), 2), "about 255 minus the mean before")
    r.chart("Histogram, original and negative (gray)",
            [{"name": "Original", "color": RGB_COL["Gray"], "values": hist256(g0)},
             {"name": "Negative", "color": "#ffc933", "values": hist256(g1)}])
    r.table("Top-left 3 x 3, original", to_gray(img)[:3, :3].tolist(), heat=(0, 255))
    r.table("Top-left 3 x 3, negative", (255 - to_gray(img))[:3, :3].tolist(), heat=(0, 255),
            caption="Check any cell: original + negative = 255.")
    r.source('''
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# Image Negative
negative = 255 - img''')
    return r


@op("stats",
    desc="Average brightness, how spread out the pixels are, and the histogram of each channel.",
    formula="mean = (1/MN) sum f(x, y)\nvariance = (1/MN) sum (f(x, y) - mean)^2\nstd = sqrt(variance)")
def stats(img, p, ctx):
    r = Result()
    g = to_gray(img)
    chans = {"Gray": g, "R": img[:, :, 2], "G": img[:, :, 1], "B": img[:, :, 0]}
    r.image("Original", img, file="original")
    r.image("Grayscale", g, file="gray")
    a = g.astype(np.float64)
    r.stat("Mean", num(a.mean(), 4), "np.mean(img)")
    r.stat("Variance", num(a.var(), 4), "np.var(img)")
    r.stat("Std deviation", num(a.std(), 4), "np.std(img)")
    r.stat("Minimum", int(g.min()), "np.min(img)")
    r.stat("Maximum", int(g.max()), "np.max(img)")
    r.stat("Median", num(np.median(g), 2), "np.median(img)")
    rows = []
    for k, ch in chans.items():
        c = ch.astype(np.float64)
        rows.append([k, num(c.mean(), 3), num(c.var(), 3), num(c.std(), 3), int(ch.min()), int(ch.max())])
    r.table("Per channel", rows, columns=["Channel", "Mean", "Variance", "Std", "Min", "Max"])
    r.chart("Histogram, color channels",
            [{"name": k, "color": RGB_COL[k], "values": hist256(chans[k])} for k in ("R", "G", "B")])
    r.chart("Histogram, grayscale", [{"name": "Gray", "color": RGB_COL["Gray"], "values": hist256(g)}])
    r.note("Low std = flat, low contrast image. High std = pixels spread over many intensities.")
    r.source('''
img = cv2.imread(Img_path)
# Estimate parameters
mean = np.mean(img)
variance = np.var(img)
std = np.std(img)''')
    return r


@op("binary",
    desc="Turn the image into pure black and white by choosing a threshold.",
    formula="b(x, y) = 255 if f(x, y) > T else 0",
    params=[
        {"name": "method", "label": "Threshold", "type": "select", "default": "fixed",
         "options": [["fixed", "Fixed value"], ["otsu", "Otsu (automatic)"]]},
        {"name": "t", "label": "T", "type": "slider", "min": 0, "max": 255, "step": 1, "default": 128},
        {"name": "invert", "label": "Invert", "type": "checkbox", "default": False},
    ])
def binary(img, p, ctx):
    g = to_gray(img)
    if p["method"] == "otsu":
        t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        t = float(p["t"])
    b = np.where(g > t, 255, 0).astype(np.uint8)
    small = cv2.resize(g, (8, 8), interpolation=cv2.INTER_AREA)
    sb = (small > t).astype(int)
    if p["invert"]:
        b = 255 - b
        sb = 1 - sb
    white = float((b == 255).mean() * 100)
    r = Result()
    r.image("Grayscale", g, file="gray")
    r.image("Binary", b, f"T = {num(t)}", "binary")
    r.stat("Threshold used", num(t), "Otsu picks it from the histogram" if p["method"] == "otsu" else "your slider")
    r.stat("White pixels", f"{white:.2f} %")
    r.stat("Black pixels", f"{100 - white:.2f} %")
    r.stat("Unique values", str(np.unique(b).tolist()), "only 0 and 255")
    r.chart("Gray histogram with threshold",
            [{"name": "Gray", "color": RGB_COL["Gray"], "values": hist256(g)}], marker=num(t))
    r.table("Your image shrunk to 8 x 8, as 0 and 1", sb.tolist(), heat=(0, 1),
            caption="Same idea as the lab's 4 x 4 binary image matrix.")
    r.source('''
# Create a Binary Image
_, binary = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
# or automatic
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)''')
    return r
