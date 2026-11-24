import io
import os
import uuid
import logging
from collections import OrderedDict

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

import dip  # registers all operations
from dip.registry import OPS, catalog, default_params
from dip.utils import decode_image, NeedsPackage

logging.getLogger("werkzeug").setLevel(logging.ERROR)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024

# Phone photos are often 12 MP. On a small server that is slow, so big uploads are
# shrunk to this many pixels on the longest side. Set MAX_SIDE=0 to keep the original.
MAX_SIDE = int(os.environ.get("MAX_SIDE", "0"))

# Uploaded images live in memory, so every click does not re-upload the file.
STORE = OrderedDict()
MAX_IMAGES = 12


def keep(img, name, nbytes):
    if MAX_SIDE and max(img.shape[:2]) > MAX_SIDE:
        k = MAX_SIDE / max(img.shape[:2])
        img = cv2.resize(img, (max(1, int(img.shape[1] * k)), max(1, int(img.shape[0] * k))),
                         interpolation=cv2.INTER_AREA)
    iid = uuid.uuid4().hex[:12]
    STORE[iid] = {"img": img, "name": name, "nbytes": nbytes}
    while len(STORE) > MAX_IMAGES:
        STORE.popitem(last=False)
    h, w = img.shape[:2]
    return {"id": iid, "name": name, "width": w, "height": h, "kb": round(nbytes / 1024)}


def sample_image():
    h, w = 420, 640
    hue = np.tile(np.linspace(0, 150, w, dtype=np.uint8), (h, 1))
    val = np.tile(np.linspace(255, 90, h, dtype=np.uint8).reshape(-1, 1), (1, w))
    hsv = np.dstack([hue, np.full((h, w), 200, np.uint8), val])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.circle(img, (200, 210), 110, (255, 255, 255), -1)
    cv2.circle(img, (200, 210), 60, (30, 30, 30), -1)
    cv2.rectangle(img, (380, 90), (580, 250), (20, 20, 20), -1)
    cv2.rectangle(img, (400, 110), (560, 230), (60, 190, 255), -1)
    cv2.putText(img, "DIP", (395, 340), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 255), 6, cv2.LINE_AA)
    return img


@app.route("/")
def home():
    return render_template("index.html")


@app.get("/healthz")
def healthz():
    return "ok"


@app.get("/api/catalog")
def api_catalog():
    return jsonify(catalog())


@app.post("/api/upload")
def api_upload():
    f = request.files.get("image")
    if not f or f.filename == "":
        return jsonify(error="Choose an image file first."), 400
    data = f.read()
    img = decode_image(data)
    if img is None:
        return jsonify(error="This file is not an image OpenCV can read. Try JPG, PNG, BMP or TIFF."), 400
    return jsonify(keep(img, f.filename, len(data)))


@app.get("/api/sample")
def api_sample():
    img = sample_image()
    ok, buf = cv2.imencode(".png", img)
    return jsonify(keep(img, "sample.png", len(buf)))


@app.post("/api/run")
def api_run():
    body = request.get_json(silent=True) or {}
    entry = STORE.get(body.get("id"))
    if entry is None:
        return jsonify(error="Upload an image first."), 404
    op_id = body.get("op")
    if op_id not in OPS:
        return jsonify(error="This tool is not built yet."), 404
    params = default_params(op_id)
    params.update(body.get("params") or {})
    for spec in OPS[op_id]["params"]:          # reference images come in as ids
        if spec["type"] == "image":
            ref = STORE.get(params.get(spec["name"]))
            params[spec["name"]] = ref["img"] if ref else None
    try:
        res = OPS[op_id]["fn"](entry["img"], params, entry)
    except NeedsPackage as e:
        return jsonify(error=str(e)), 501
    except Exception as e:  # show the reason in the page instead of a blank screen
        return jsonify(error=f"{type(e).__name__}: {e}"), 500
    return jsonify(res.to_dict())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    on_server = bool(os.environ.get("PORT"))      # Render sets PORT
    if not on_server:
        import webbrowser
        from threading import Timer
        Timer(1.2, lambda: webbrowser.open_new(f"http://127.0.0.1:{port}/")).start()
    # host 0.0.0.0 lets a phone on the same Wi-Fi open http://<pc-ip>:5000
    app.run(debug=False, host="0.0.0.0", port=port)
