"""Detection and recognition: YOLO, contours, Hough, template matching, faces, OCR."""
import os

import cv2
import numpy as np

from .registry import op
from .utils import Result, to_gray, hist256, num, fit, norm8, NeedsPackage

AMBER, GRAY, BLUE, RED, GREEN = "#ffc933", "#d8d8d8", "#4d8dff", "#ff4d4d", "#3ddc84"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL = {}


def yolo_model():
    if "m" in _MODEL:
        return _MODEL["m"]
    try:
        from ultralytics import YOLO
    except Exception:
        raise NeedsPackage("YOLO needs the ultralytics package. Run in the terminal:  pip install ultralytics  (then restart python app.py)")
    for cand in (os.path.join(ROOT, "yolo11n.pt"), os.path.join(ROOT, "models", "yolo11n.pt"), os.path.join(os.getcwd(), "yolo11n.pt")):
        if os.path.isfile(cand):
            _MODEL["m"] = YOLO(cand)
            return _MODEL["m"]
    try:
        _MODEL["m"] = YOLO("yolo11n.pt")      # downloads once if internet is available
    except Exception as e:
        raise NeedsPackage("yolo11n.pt not found. Put the file yolo11n.pt in the project folder (next to app.py). " + str(e)[:120])
    return _MODEL["m"]


def detect(img, conf, iou=0.7, imgsz=640):
    m = yolo_model()
    res = m.predict(img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    out = []
    for b in res.boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        k = int(b.cls[0])
        out.append({"cls": m.names[k], "conf": float(b.conf[0]), "box": (x1, y1, x2, y2)})
    return res, out


PRESETS = {
    "classroom": ("People and screens (lab script)", {"person", "tv"}),
    "person": ("People only", {"person"}),
    "vehicles": ("Vehicles", {"car", "bus", "truck", "motorcycle", "bicycle", "train", "airplane", "boat"}),
    "animals": ("Animals", {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}),
    "devices": ("Devices", {"tv", "laptop", "cell phone", "keyboard", "mouse", "remote"}),
    "all": ("Everything YOLO knows", None),
}
COLORS = {"person": (0, 255, 0), "tv": (255, 0, 0)}


def col_for(name):
    if name in COLORS:
        return COLORS[name]
    h = abs(hash(name)) % 180
    return tuple(int(v) for v in cv2.cvtColor(np.uint8([[[h, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


@op("count_objects",
    desc="Count how many people (and screens or other objects) are in the picture using YOLO, with boxes drawn on each one.",
    formula="YOLO predicts boxes with a class and a confidence.\nkeep a box if confidence >= threshold",
    params=[
        {"name": "preset", "label": "Count", "type": "select", "default": "classroom", "options": [[k, v[0]] for k, v in PRESETS.items()]},
        {"name": "conf", "label": "Confidence", "type": "slider", "min": 0.05, "max": 0.9, "step": 0.05, "default": 0.25},
    ], heavy=True)
def count_objects(img, p, ctx):
    wanted = PRESETS[p["preset"]][1]
    res, dets = detect(img, float(p["conf"]))
    dets = [d for d in dets if wanted is None or d["cls"] in wanted]
    ann = img.copy()
    H, W = ann.shape[:2]
    th = max(2, int(max(H, W) / 450))
    fs = max(0.6, max(H, W) / 800)
    counts = {}
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        c = col_for(d["cls"])
        cv2.rectangle(ann, (x1, y1), (x2, y2), c, th)
        label = {"person": "Person", "tv": "Display"}.get(d["cls"], d["cls"]) + f" {d['conf']:.2f}"
        cv2.putText(ann, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, c, th)
        counts[d["cls"]] = counts.get(d["cls"], 0) + 1
    if p["preset"] == "classroom":
        txt = f"Total Heads (Persons): {counts.get('person', 0)} | Total Displays: {counts.get('tv', 0)}"
    else:
        txt = "Total: " + ", ".join(f"{k} {v}" for k, v in counts.items()) if counts else "Nothing found"
    cv2.putText(ann, txt, (20, int(40 * fs + 10)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), th + 1)
    total = len(dets)
    main = counts.get("person", total) if p["preset"] in ("classroom", "person") else total
    label = "people counted" if p["preset"] in ("classroom", "person") else "objects counted"
    r = Result()
    r.big(main, label)
    r.image("Original", img, "", "original")
    r.image("Detected", ann, txt, "detected")
    for k, v in counts.items():
        r.stat(k.capitalize(), v)
    r.stat("Confidence used", p["conf"], "lower finds more, but also more mistakes")
    r.stat("Objects of any class", len(detect(img, float(p["conf"]))[1]) if False else len(res.boxes))
    if counts:
        r.chart("Count per class", [{"name": "Count", "color": AMBER, "values": list(counts.values())}],
                labels=list(counts.keys()), x_label="Class", y_label="Count", kind="bars")
    if dets:
        r.chart("Confidence of every detection", [{"name": "Confidence", "color": BLUE, "values": [round(d["conf"], 3) for d in sorted(dets, key=lambda d: -d["conf"])]}],
                x_label="Detection (sorted)", y_label="Confidence", kind="bars")
        r.table("Detections", [[i + 1, d["cls"], round(d["conf"], 3), d["box"][0], d["box"][1], d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]]
                               for i, d in enumerate(sorted(dets, key=lambda d: -d["conf"]))][:40],
                columns=["#", "Class", "Conf", "x", "y", "w", "h"])
    else:
        r.note("Nothing of this kind was found. Lower the confidence or choose another count type.")
    r.source('''
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
results = model(image, conf=0.25)
for box in results[0].boxes:
    name = model.names[int(box.cls[0])]
    if name == "person":
        head_count += 1''')
    return r


@op("yolo_detect",
    desc="Run YOLO11 and draw every object it finds (80 everyday classes), with a table of what was detected.",
    formula="IoU = overlap area / union area\nnon-maximum suppression drops boxes that overlap more than the IoU limit",
    params=[
        {"name": "conf", "label": "Confidence", "type": "slider", "min": 0.05, "max": 0.9, "step": 0.05, "default": 0.1},
        {"name": "iou", "label": "IoU limit", "type": "slider", "min": 0.1, "max": 0.95, "step": 0.05, "default": 0.7},
        {"name": "sz", "label": "Image size", "type": "select", "default": "640", "options": [["320", "320"], ["640", "640"], ["960", "960"]]},
    ], heavy=True)
def yolo_detect(img, p, ctx):
    res, dets = detect(img, float(p["conf"]), float(p["iou"]), int(p["sz"]))
    plot = res.plot()
    counts = {}
    for d in dets:
        counts[d["cls"]] = counts.get(d["cls"], 0) + 1
    r = Result()
    r.big(len(dets), "objects detected")
    r.image("Original", img, "", "original")
    r.image("YOLO detection", plot, f"confidence >= {p['conf']}", "yolo")
    r.stat("Classes found", len(counts))
    if dets:
        r.stat("Most common", max(counts, key=counts.get), f"{max(counts.values())} times")
        r.stat("Highest confidence", f"{max(d['conf'] for d in dets):.2f}")
        r.stat("Average confidence", f"{np.mean([d['conf'] for d in dets]):.2f}")
        items = sorted(counts.items(), key=lambda kv: -kv[1])
        r.chart("Objects per class", [{"name": "Count", "color": AMBER, "values": [v for _, v in items]}],
                labels=[k for k, _ in items], x_label="Class", y_label="Count", kind="bars")
        r.chart("Confidence histogram", [{"name": "Detections", "color": BLUE, "values": np.histogram([d["conf"] for d in dets], 10, (0, 1))[0].tolist()}],
                labels=[f"{i / 10:.1f}" for i in range(10)], x_label="Confidence", y_label="Detections", kind="bars")
        H, W = img.shape[:2]
        r.table("All detections", [[i + 1, d["cls"], round(d["conf"], 3), d["box"][0], d["box"][1], d["box"][2], d["box"][3],
                                    round(100 * (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]) / (W * H), 2)]
                                   for i, d in enumerate(sorted(dets, key=lambda d: -d["conf"]))][:60],
                columns=["#", "Class", "Conf", "x1", "y1", "x2", "y2", "Area %"])
    else:
        r.note("No objects found. Lower the confidence threshold.")
    r.source('''
model = YOLO("yolo11n.pt")
result = model(image, conf=0.1)
print("Total detected objects:", len(result[0].boxes))
detected_image = result[0].plot()''')
    return r


@op("contours",
    desc="Outline every blob, count them and measure area, perimeter and roundness.",
    formula="circularity = 4 pi area / perimeter^2   (1 = perfect circle)",
    params=[
        {"name": "inv", "label": "Objects are dark", "type": "checkbox", "default": False},
        {"name": "minarea", "label": "Ignore blobs smaller than", "type": "slider", "min": 0, "max": 2000, "step": 10, "default": 80},
    ])
def contours(img, p, ctx):
    c = fit(img, 800)
    g = cv2.GaussianBlur(to_gray(c), (5, 5), 0)
    mode = cv2.THRESH_BINARY_INV if p["inv"] else cv2.THRESH_BINARY
    _, b = cv2.threshold(g, 0, 255, mode + cv2.THRESH_OTSU)
    cs, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = [k for k in cs if cv2.contourArea(k) >= p["minarea"]]
    cs.sort(key=lambda k: -cv2.contourArea(k))
    out = c.copy()
    rows = []
    for i, k in enumerate(cs):
        cv2.drawContours(out, [k], -1, (51, 201, 255), 2)
        x, y, w, h = cv2.boundingRect(k)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 1)
        a, pe = cv2.contourArea(k), cv2.arcLength(k, True)
        M = cv2.moments(k)
        cx, cy = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] else (x, y)
        if i < 60:
            cv2.putText(out, str(i + 1), (cx - 6, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        rows.append([i + 1, round(a), round(pe), w, h, round(4 * np.pi * a / (pe * pe), 3) if pe else 0, cx, cy])
    r = Result().gallery()
    r.big(len(cs), "blobs counted")
    r.image("Binary (Otsu)", b, "", "binary")
    r.image("Contours and boxes", out, "", "contours")
    if rows:
        r.stat("Largest area", f"{rows[0][1]:,}", "pixels")
        r.stat("Smallest area", f"{rows[-1][1]:,}", "pixels")
        r.stat("Mean area", f"{np.mean([x[1] for x in rows]):,.0f}")
        r.stat("Roundest", f"{max(rows, key=lambda x: x[5])[5]}", f"blob #{max(rows, key=lambda x: x[5])[0]}")
        r.table("Blob measurements (largest 25)", rows[:25], columns=["#", "Area", "Perimeter", "Width", "Height", "Circularity", "cx", "cy"])
        r.chart("Blob areas", [{"name": "Area", "color": AMBER, "values": [x[1] for x in rows[:40]]}], x_label="Blob (largest first)", y_label="Pixels", kind="bars")
    else:
        r.note("No blobs found. Try 'Objects are dark' or a smaller size limit.")
    r.source('''
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))''')
    return r


@op("hough",
    desc="Find straight lines and circles with the Hough transform.",
    formula="line: rho = x cos(theta) + y sin(theta)\ncircle: (x - a)^2 + (y - b)^2 = r^2",
    params=[
        {"name": "lt", "label": "Line votes", "type": "slider", "min": 20, "max": 300, "step": 5, "default": 80},
        {"name": "ml", "label": "Min line length", "type": "slider", "min": 10, "max": 300, "step": 5, "default": 50},
        {"name": "p2", "label": "Circle strictness", "type": "slider", "min": 10, "max": 100, "step": 1, "default": 35},
        {"name": "rmin", "label": "Min radius", "type": "slider", "min": 3, "max": 100, "step": 1, "default": 10},
        {"name": "rmax", "label": "Max radius", "type": "slider", "min": 10, "max": 300, "step": 5, "default": 120},
    ], heavy=True)
def hough(img, p, ctx):
    c = fit(img, 640)
    g = to_gray(c)
    ed = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 1.4), 50, 150)
    lines = cv2.HoughLinesP(ed, 1, np.pi / 180, int(p["lt"]), minLineLength=int(p["ml"]), maxLineGap=8)
    lo = c.copy()
    nl = 0
    angs = []
    if lines is not None:
        nl = len(lines)
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(lo, (x1, y1), (x2, y2), (0, 0, 255), 2)
            angs.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)
    circ = cv2.HoughCircles(cv2.medianBlur(g, 5), cv2.HOUGH_GRADIENT, 1, max(10, g.shape[0] // 12), param1=120, param2=int(p["p2"]),
                            minRadius=int(p["rmin"]), maxRadius=int(max(p["rmax"], p["rmin"] + 1)))
    co = c.copy()
    nc = 0
    circ = None if circ is None else circ.reshape(-1, 3)[None]
    if circ is not None:
        for x, y, rr in np.round(circ[0]).astype(int)[:80]:
            cv2.circle(co, (x, y), rr, (0, 255, 0), 2)
            cv2.circle(co, (x, y), 2, (0, 0, 255), 3)
        nc = min(80, len(circ[0]))
    r = Result()
    r.image("Canny edges used", ed, "", "edges")
    r.image("Lines found", lo, f"{nl} segments", "lines")
    r.image("Circles found", co, f"{nc} circles", "circles")
    r.stat("Line segments", nl)
    r.stat("Circles", nc)
    if angs:
        r.chart("Line directions", [{"name": "Lines", "color": AMBER, "values": np.histogram(angs, 18, (0, 180))[0].tolist()}],
                labels=[str(i * 10) for i in range(18)], x_label="Angle (degrees)", y_label="Lines", kind="bars")
    if circ is not None:
        r.table("Circles (first 15)", [[i + 1, int(a), int(b), int(c_)] for i, (a, b, c_) in enumerate(np.round(circ[0][:15]).astype(int))], columns=["#", "x", "y", "radius"])
    r.source('''
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=50, maxLineGap=8)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30, param1=120, param2=35)''')
    return r


@op("template_match",
    desc="Pick a small patch of your image (or upload one) and find where else it appears.",
    formula="TM_CCOEFF_NORMED: correlation between the patch and every window, -1 to 1",
    params=[
        {"name": "tpl", "label": "Template image (optional)", "type": "image", "default": None, "button": "Choose template"},
        {"name": "x", "label": "Patch x %", "type": "slider", "min": 0, "max": 90, "step": 1, "default": 40},
        {"name": "y", "label": "Patch y %", "type": "slider", "min": 0, "max": 90, "step": 1, "default": 40},
        {"name": "s", "label": "Patch size %", "type": "slider", "min": 5, "max": 40, "step": 1, "default": 12},
        {"name": "m", "label": "Method", "type": "select", "default": "ccoeff", "options": [["ccoeff", "Correlation coefficient"], ["ccorr", "Cross correlation"], ["sqdiff", "Squared difference"]]},
        {"name": "thr", "label": "Match threshold", "type": "slider", "min": 0.3, "max": 0.99, "step": 0.01, "default": 0.8},
    ], heavy=True)
def template_match(img, p, ctx):
    c = fit(img, 800)
    g = to_gray(c)
    H, W = g.shape
    if p.get("tpl") is not None:
        t = to_gray(p["tpl"])
        if t.shape[0] >= H or t.shape[1] >= W:
            s = 0.5 * min(H / t.shape[0], W / t.shape[1])
            t = cv2.resize(t, None, fx=s, fy=s)
        src = "uploaded template"
    else:
        sz = max(8, int(min(H, W) * p["s"] / 100))
        x0, y0 = min(W - sz, int(W * p["x"] / 100)), min(H - sz, int(H * p["y"] / 100))
        t = g[y0:y0 + sz, x0:x0 + sz].copy()
        src = f"patch from ({x0}, {y0})"
    meth = {"ccoeff": cv2.TM_CCOEFF_NORMED, "ccorr": cv2.TM_CCORR_NORMED, "sqdiff": cv2.TM_SQDIFF_NORMED}[p["m"]]
    res = cv2.matchTemplate(g, t, meth)
    score = (1 - res) if p["m"] == "sqdiff" else res
    th, tw = t.shape
    work = score.copy()
    hits = []
    while len(hits) < 50:
        _, mx, _, loc = cv2.minMaxLoc(work)
        if mx < p["thr"]:
            break
        hits.append((loc[0], loc[1], float(mx)))
        cv2.rectangle(work, (max(0, loc[0] - tw // 2), max(0, loc[1] - th // 2)), (loc[0] + tw // 2, loc[1] + th // 2), -2, -1)
    out = c.copy()
    for x, y, s in hits:
        cv2.rectangle(out, (x, y), (x + tw, y + th), (0, 255, 0), 2)
    heat = cv2.applyColorMap(norm8(np.clip(score, -1, 1)), cv2.COLORMAP_INFERNO)
    r = Result()
    r.big(len(hits), "matches found")
    r.image("Template", t, f"{tw} x {th}, {src}", "template")
    r.image("Matches", out, f"score >= {p['thr']}", "matches")
    r.image("Match score map", heat, "Bright = good match", "heatmap")
    r.stat("Best score", f"{max([h[2] for h in hits], default=float(score.max())):.3f}")
    r.stat("Matches", len(hits))
    if hits:
        r.table("Matches", [[i + 1, x, y, round(s, 3)] for i, (x, y, s) in enumerate(hits[:20])], columns=["#", "x", "y", "Score"])
    r.note("With a patch cut from the same image, the best match is the patch itself with a score near 1.")
    r.source('''
res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
ys, xs = np.where(res >= 0.8)''')
    return r


@op("faces",
    desc="Find faces (and eyes inside them) with a Haar cascade classifier.",
    formula="Haar features: sums of rectangles compared by a boosted cascade",
    params=[
        {"name": "scale", "label": "Scale factor", "type": "slider", "min": 1.05, "max": 1.5, "step": 0.01, "default": 1.1},
        {"name": "nb", "label": "Min neighbours", "type": "slider", "min": 1, "max": 10, "step": 1, "default": 5},
        {"name": "eyes", "label": "Also find eyes", "type": "checkbox", "default": True},
    ])
def faces(img, p, ctx):
    d = cv2.data.haarcascades if hasattr(cv2, "data") else ""
    fp = os.path.join(d, "haarcascade_frontalface_default.xml")
    c = fit(img, 1000)
    g = cv2.equalizeHist(to_gray(c))
    out = c.copy()
    ne = 0
    if os.path.isfile(fp):
        method = "Haar cascade (OpenCV)"
        fc = cv2.CascadeClassifier(fp)
        ec = cv2.CascadeClassifier(os.path.join(d, "haarcascade_eye.xml"))
        fs = fc.detectMultiScale(g, float(p["scale"]), int(p["nb"]), minSize=(30, 30))
        fs = [tuple(int(v) for v in f) for f in fs]
        for x, y, w, h in fs:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if p["eyes"]:
                for ex, ey, ew, eh in ec.detectMultiScale(g[y:y + h // 2 + h // 8, x:x + w], 1.1, 6):
                    cv2.rectangle(out, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 160, 0), 2)
                    ne += 1
    else:
        try:
            from skimage import data as skdata
            from skimage.feature import Cascade
        except Exception:
            raise NeedsPackage("No face cascade found. Run: pip install scikit-image")
        method = "LBP cascade (scikit-image)"
        det = Cascade(skdata.lbp_frontal_face_cascade_filename())
        hits = det.detect_multi_scale(g, scale_factor=float(p["scale"]) + 0.1, step_ratio=1, min_size=(30, 30),
                                      max_size=(g.shape[1], g.shape[0]))
        fs = [(int(h["c"]), int(h["r"]), int(h["width"]), int(h["height"])) for h in hits]
        for x, y, w, h in fs:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    r = Result()
    r.big(len(fs), "faces found")
    r.image("Original", c, "", "original")
    r.image("Faces", out, "green = face" + (", orange = eye" if ne else ""), "faces")
    r.stat("Method", method)
    r.stat("Faces", len(fs))
    if ne or os.path.isfile(fp):
        r.stat("Eyes", ne)
    if len(fs):
        r.table("Faces", [[i + 1, x, y, w, h] for i, (x, y, w, h) in enumerate(fs)], columns=["#", "x", "y", "w", "h"])
        for i, (x, y, w, h) in enumerate(fs[:6]):
            r.image(f"Face {i + 1}", c[max(0, y):y + h, max(0, x):x + w], "", f"face_{i + 1}")
    r.note("Cascades work best on frontal, well lit faces. To count people in general use the YOLO counter.")
    r.source('''
face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)''')
    return r


@op("ocr",
    desc="Read text out of the picture. Uses EasyOCR if installed, otherwise Tesseract.",
    formula="detect text regions, then recognize characters in each region",
    params=[{"name": "engine", "label": "Engine", "type": "select", "default": "auto",
             "options": [["auto", "Automatic"], ["easyocr", "EasyOCR"], ["tesseract", "Tesseract"]]}],
    heavy=True)
def ocr(img, p, ctx):
    c = fit(img, 1400)
    items = []
    used = None
    err = ""
    if p["engine"] in ("auto", "easyocr"):
        try:
            import easyocr
            if "ocr" not in _MODEL:
                _MODEL["ocr"] = easyocr.Reader(["en"], gpu=False, verbose=False)
            for box, txt, cf in _MODEL["ocr"].readtext(c[:, :, ::-1].copy()):
                xs = [pt[0] for pt in box]; ys = [pt[1] for pt in box]
                items.append((txt, float(cf), int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
            used = "EasyOCR"
        except ImportError:
            err = "EasyOCR is not installed."
        except Exception as e:
            err = f"EasyOCR failed: {str(e)[:150]}"
    if used is None and p["engine"] in ("auto", "tesseract"):
        try:
            import pytesseract
            d = pytesseract.image_to_data(to_gray(c), output_type=pytesseract.Output.DICT)
            for i, t in enumerate(d["text"]):
                if t.strip() and float(d["conf"][i]) > 0:
                    items.append((t, float(d["conf"][i]) / 100, d["left"][i], d["top"][i], d["left"][i] + d["width"][i], d["top"][i] + d["height"][i]))
            used = "Tesseract"
        except ImportError:
            err += " pytesseract is not installed."
        except Exception as e:
            err += f" Tesseract failed: {str(e)[:150]}"
    if used is None:
        raise NeedsPackage("No OCR engine is ready. " + err + "  Install one:  pip install easyocr   (first run downloads a model, needs internet)")
    out = c.copy()
    for t, cf, x1, y1, x2, y2 in items:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    r = Result()
    r.big(len(items), "text pieces found")
    r.image("Original", c, "", "original")
    r.image("Text regions", out, f"engine: {used}", "ocr")
    r.stat("Engine", used)
    if items:
        r.stat("Average confidence", f"{100 * np.mean([i[1] for i in items]):.0f} %")
        r.table("Recognized text", [[i + 1, t, round(cf, 3), x1, y1] for i, (t, cf, x1, y1, x2, y2) in enumerate(items)][:60],
                columns=["#", "Text", "Confidence", "x", "y"])
        r.note("Full text: " + " ".join(i[0] for i in items))
    else:
        r.note("No text found in this image.")
    r.source('''
import easyocr
reader = easyocr.Reader(["en"])
result = reader.readtext(image)''')
    return r
