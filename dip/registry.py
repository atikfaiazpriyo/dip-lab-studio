"""Operation registry.

Every image-processing operation is a plain Python function that returns a
Result. Adding a new one = write the function + decorate it with @op(...).
The sidebar of the website is built automatically from ROADMAP + OPS.
"""

ROADMAP = [
    {"name": "Basics", "lab": "Lab 1", "items": [
        ("info", "Image information"),
        ("display", "Read, display and grayscale"),
        ("negative", "Image negative"),
        ("stats", "Mean, variance and std"),
        ("binary", "Binary image"),
    ]},
    {"name": "Digital image", "lab": "Lab 1", "items": [
        ("pixel_grid", "Pixel value grid"),
        ("sampling", "Sampling"),
        ("quantization", "Quantization"),
        ("interpolation", "Interpolation"),
        ("index_image", "Indexed image"),
    ]},
    {"name": "Noise and color", "lab": "Lab 1", "items": [
        ("gaussian_noise", "Noise models"),
        ("color_spaces", "Color spaces"),
        ("hsi", "RGB to HSI"),
        ("hsi_to_rgb", "HSI to RGB"),
        ("pixel_calc", "Pixel color calculator"),
    ]},
    {"name": "Enhancement", "lab": "Lab 2", "items": [
        ("contrast", "Contrast and brightness"),
        ("gamma", "Gamma correction"),
        ("contrast_metrics", "Contrast measures"),
        ("histogram", "Histogram"),
        ("stretching", "Contrast stretching"),
        ("equalization", "Histogram equalization"),
        ("clahe", "CLAHE"),
    ]},
    {"name": "Histogram lab", "lab": "Lab 2", "items": [
        ("eq_steps", "Equalization step by step"),
        ("hist_match", "Histogram matching"),
    ]},
    {"name": "Transform and filters", "lab": "Lab 2", "items": [
        ("transform", "Crop, resize, rotate, flip"),
        ("smoothing", "Smoothing filters"),
        ("denoise", "Noise removal"),
        ("sharpen", "Sharpening"),
    ]},
    {"name": "Edge detection", "lab": "Lab 2", "items": [
        ("edges", "Sobel, Prewitt, Roberts, Laplacian"),
        ("canny", "Canny"),
    ]},
    {"name": "Segmentation", "lab": "Lab 2", "items": [
        ("thresholding", "Thresholding"),
        ("watershed", "Watershed"),
        ("region_growing", "Region growing"),
        ("kmeans", "K-means"),
    ]},
    {"name": "Morphology", "lab": "Lab 4", "items": [
        ("morphology", "Erosion, dilation, opening, closing"),
        ("skeleton", "Skeletonization"),
    ]},
    {"name": "Frequency and compression", "lab": "Lab 4", "items": [
        ("fft", "DFT and FFT spectrum"),
        ("freq_filter", "Low-pass and high-pass"),
        ("jpeg", "JPEG compression"),
        ("dct", "DCT compression"),
    ]},
    {"name": "Feature extraction", "lab": "Lab 5", "items": [
        ("feature_summary", "All features at once"),
        ("harris", "Harris corners"),
        ("hog", "HOG"),
        ("lbp", "LBP"),
        ("glcm", "GLCM texture"),
        ("keypoints", "SIFT, ORB, FAST"),
        ("zoom_points", "Zoom into interest points"),
    ]},
    {"name": "Detection and recognition", "lab": "Lab 5 and 6", "items": [
        ("count_objects", "Count people and objects"),
        ("yolo_detect", "YOLO object detection"),
        ("contours", "Contours and blob counting"),
        ("hough", "Hough lines and circles"),
        ("template_match", "Template matching"),
        ("faces", "Face detection"),
        ("ocr", "Text recognition (OCR)"),
    ]},
    {"name": "Graph and report", "lab": "Lab 1 and final", "items": [
        ("pixel_graph", "Pixel connectivity graph"),
        ("report", "Full image report"),
    ]},
]

OPS = {}


def _title_for(op_id):
    for g in ROADMAP:
        for i, t in g["items"]:
            if i == op_id:
                return t
    return op_id


def op(op_id, desc="", formula="", params=None, heavy=False):
    def deco(fn):
        OPS[op_id] = {
            "fn": fn,
            "title": _title_for(op_id),
            "desc": desc,
            "formula": formula,
            "params": params or [],
            "heavy": heavy,
        }
        return fn
    return deco


def catalog():
    """Only tools that are already built are listed, so the sidebar never
    shows dead buttons."""
    groups = []
    for g in ROADMAP:
        items = []
        for op_id, title in g["items"]:
            o = OPS.get(op_id)
            if o:
                items.append({"id": op_id, "title": title, "desc": o["desc"], "formula": o["formula"],
                              "params": o["params"], "heavy": o["heavy"]})
        if items:
            groups.append({"name": g["name"], "lab": g["lab"], "items": items})
    return groups


def default_params(op_id):
    return {p["name"]: p.get("default") for p in OPS[op_id]["params"]}
