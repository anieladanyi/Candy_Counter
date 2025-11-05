import cv2
import numpy as np
import pandas as pd
import json, time
from dataclasses import dataclass

# --- (opcionális) debug: per-szín maszkok mentése és pixelek száma ---
def _debug_save_masks(masks, out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True)
    totals = {}
    for name, m in masks.items():
        totals[name] = int((m > 0).sum())
        cv2.imwrite(os.path.join(out_dir, f"mask_{name}.png"), m)
    print("[DEBUG] nonzero pixels per color mask:", totals, flush=True)

# HSV színtartományok (finomítható – jelenleg kissé szigorúbb S/V a háttér kizárására)
COLOR_RANGES = {
    # red: szűkebb alsó sáv, hogy ne menjen az orange-ra
    "red1":   ((0,   45, 45), (7, 255, 255)),     # H: 0..7
    "red2":   ((168, 45, 45), (179, 255, 255)),

    # orange: külön sáv, felfelé nem ér el a yellow-ba
    "orange": ((8,   40, 40), (22, 255, 255)),    # H: 8..22

    # yellow: lejjebb nem ér be az orange-ba
    "yellow": ((23,  40, 40), (40, 255, 255)),    # H: 23..40

    "green":  ((50,  40, 40), (85, 255, 255)),
    "blue":   ((92,  40, 40), (120,255, 255)),
    "purple": ((125, 40, 40), (165,255, 255)),
}




# vizuális színek (BGR)
VIS_COLORS = {
    "red":    (0, 0, 255),
    "yellow": (0, 255, 255),
    "green":  (0, 255, 0),
    "blue":   (255, 0, 0),
    "orange": (0, 165, 255),
    "purple": (211, 0, 211),
    "mixed":  (200, 200, 200),
}

@dataclass
class Params:
    # alap pipeline
    blur_ksize: int = 5
    morph_open: int = 3
    morph_close: int = 3
    min_area: int = 150
    max_area: int = 200000
    circularity_min: float = 0.6
    downscale_max: int = 1400

    # watershed (érintkező cukorkák)
    use_watershed: bool = False
    dt_thresh_ratio: float = 0.4

    # Hough-kördetektálás (színfüggetlen)
    use_hough: bool = False
    hough_dp: float = 1.2
    hough_minDist: int = 60
    hough_param1: int = 100
    hough_param2: int = 30
    hough_minRadius: int = 20
    hough_maxRadius: int = 120

def _ensure_odd(k: int) -> int:
    return max(1, (k // 2) * 2 + 1)

def preprocess(bgr, p: Params):
    h, w = bgr.shape[:2]
    scale = 1.0
    if max(h, w) > p.downscale_max:
        scale = p.downscale_max / max(h, w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    blur_k = _ensure_odd(p.blur_ksize)
    if blur_k > 1:
        bgr = cv2.GaussianBlur(bgr, (blur_k, blur_k), 0)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return bgr, hsv, scale

def make_color_masks(hsv):
    masks = {}
    # piros két tartományból
    lower1, upper1 = COLOR_RANGES["red1"]
    lower2, upper2 = COLOR_RANGES["red2"]
    red_mask = cv2.inRange(hsv, np.array(lower1), np.array(upper1)) | \
               cv2.inRange(hsv, np.array(lower2), np.array(upper2))
    masks["red"] = red_mask
    for name in ["yellow", "green", "blue", "orange", "purple"]:
        lo, hi = COLOR_RANGES[name]
        masks[name] = cv2.inRange(hsv, np.array(lo), np.array(hi))
    return masks

def maybe_save_masks(masks, out_png):
    import os
    if os.environ.get("SAVE_MASKS") == "1" and out_png:
        mask_dir = os.path.join(os.path.dirname(out_png), "masks")
        _debug_save_masks(masks, mask_dir)


def combine_masks(masks):
    out = np.zeros_like(next(iter(masks.values())))
    for m in masks.values():
        out = out | m
    return out

def morphology(mask, p: Params):
    out = mask.copy()
    if p.morph_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.morph_open, p.morph_open))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if p.morph_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.morph_close, p.morph_close))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    return out

def _contours_from_binary(bin_img):
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def _filter_and_pack_contours(contours, bgr, p: Params):
    detections = []
    for cid, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < p.min_area or area > p.max_area:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        circularity = float((4.0 * np.pi * area) / (peri * peri))
        if circularity < p.circularity_min:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        detections.append({
            "id": cid,
            "cnt": cnt,
            "bbox": (x, y, w, h),
            "cx": float(cx),
            "cy": float(cy),
            "r": float(radius),
            "area": float(area),
            "circularity": circularity
        })
    return detections

def find_candies(bgr, mask_all, p: Params):
    contours = _contours_from_binary(mask_all)
    return _filter_and_pack_contours(contours, bgr, p)

def _nms_circles(dets, center_thresh_ratio=0.65):
    """
    egyszerű NMS: nagyobb körök előnyt élveznek; ha két kör közepe túl közel van
    (dist < ratio * min(r1, r2)), a kisebbet eldobjuk.
    """
    if not dets:
        return dets
    dets = sorted(dets, key=lambda d: d["r"], reverse=True)
    kept = []
    for d in dets:
        keep = True
        for k in kept:
            dx, dy = d["cx"] - k["cx"], d["cy"] - k["cy"]
            dist = (dx*dx + dy*dy) ** 0.5
            if dist < center_thresh_ratio * min(d["r"], k["r"]):
                keep = False
                break
        if keep:
            kept.append(d)
    return kept


    

def find_candies_hough(bgr, hsv, p: Params):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.medianBlur(gray, 5)

    # 1) Szürke + kontrasztjavítás
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)                  # jobb éldetektáláshoz
    # 2) Zajszűrés, de élek megtartása
    gray = cv2.medianBlur(gray, 5)                 # stabilabb Hough
    # (Maradhat a Gaussian helyett/ mellett — a medianBlurt általában kedveli a Hough)

    # 3) Hough kördetektálás
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=p.hough_dp,
        minDist=p.hough_minDist,
        param1=p.hough_param1,    # Canny felső küszöb
        param2=p.hough_param2,    # akkumulátor küszöb (LEJJEBB = több kör)
        minRadius=p.hough_minRadius,
        maxRadius=p.hough_maxRadius
    )

    detections = []
    if circles is not None:
        for cid, c in enumerate(np.round(circles[0, :]).astype("int")):
            x, y, r = c
            if r <= 0:
                continue
            area = float(np.pi * r * r)
            if area < p.min_area or area > p.max_area:
                continue
            detections.append({
                "id": cid, "cnt": None,
                "bbox": (x - r, y - r, 2 * r, 2 * r),
                "cx": float(x), "cy": float(y),
                "r": float(r), "area": area,
                "circularity": 1.0
            })
    # ÚJ: duplikátumok kiszűrése
    detections = _nms_circles(detections, center_thresh_ratio=0.65)
    return detections


# --- Watershed szeparáció (érintkező objektumok bontása) ---
def watershed_split(bgr, mask_all, p: Params):
    # bináris 0/255
    bin8 = (mask_all > 0).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(bin8, kernel, iterations=2)

    dist = cv2.distanceTransform(bin8, distanceType=cv2.DIST_L2, maskSize=5)
    _, sure_fg = cv2.threshold(dist, p.dt_thresh_ratio * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    unknown = cv2.subtract(sure_bg, sure_fg)

    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    bgr_ws = bgr.copy()
    cv2.watershed(bgr_ws, markers)

    detections = []
    for label in np.unique(markers):
        if label <= 1:  # 0 ismeretlen, 1 háttér
            continue
        comp_mask = (markers == label).astype(np.uint8) * 255
        contours = _contours_from_binary(comp_mask)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        detections.extend(_filter_and_pack_contours([cnt], bgr, p))
    return detections, dist

def classify_color(hsv, det, masks):
    cnt = det["cnt"]
    mask_cnt = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask_cnt, [cnt], -1, 255, thickness=cv2.FILLED)
    best_color, best_hits = "mixed", 0
    for cname, cmask in masks.items():
        hits = cv2.countNonZero(cv2.bitwise_and(cmask, mask_cnt))
        if hits > best_hits:
            best_hits = hits
            best_color = cname
    return best_color

def classify_color_in_circle(hsv, cx, cy, r, color_ranges):
    # körmaszk (szűkítve, hogy a peremfény ne zavarjon)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(r * 0.9), 255, -1)

    # csak elég színes és világos pixelek számítsanak (háttér kizárása)
    S = hsv[:, :, 1]; V = hsv[:, :, 2]
    sv_mask = ((S >= 45) & (V >= 45)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, sv_mask)

    # H-hisztogram (ha nincs elég pixel, mixed)
    H = hsv[:, :, 0]
    hist = cv2.calcHist([H], [0], mask, [180], [0, 180]).flatten()
    if hist.sum() < 1:
        return "mixed"
    peak_h = int(hist.argmax())

    def in_band(h, lo, hi): return lo <= h <= hi

    # SORREND: yellow -> orange -> red -> green/blue/purple
    if in_band(peak_h, COLOR_RANGES["yellow"][0][0], COLOR_RANGES["yellow"][1][0]):
        return "yellow"
    if in_band(peak_h, COLOR_RANGES["orange"][0][0], COLOR_RANGES["orange"][1][0]):
        return "orange"
    if in_band(peak_h, COLOR_RANGES["red1"][0][0], COLOR_RANGES["red1"][1][0]) or \
       in_band(peak_h, COLOR_RANGES["red2"][0][0], COLOR_RANGES["red2"][1][0]):
        return "red"

    for name in ["green", "blue", "purple"]:
        lo, hi = COLOR_RANGES[name][0][0], COLOR_RANGES[name][1][0]
        if in_band(peak_h, lo, hi):
            return name
    return "mixed"


    print(f"[DBG] peakH={peak_h} -> mixed", flush=True)
    return "mixed"


def _draw_legend(vis, counts, start=(10, 10), box=18, gap=6):
    x, y = start
    for cname in ["red", "yellow", "green", "blue", "orange", "purple"]:
        if cname not in VIS_COLORS:
            continue
        color = VIS_COLORS[cname]
        cv2.rectangle(vis, (x, y), (x + box, y + box), color, thickness=-1)
        txt = f"{cname}: {counts.get(cname, 0)}"
        cv2.putText(vis, txt, (x + box + 8, y + box - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 2, cv2.LINE_AA)
        y += box + gap

def annotate_and_export(bgr, hsv, detections, masks, out_png, out_csv):
    vis = bgr.copy()
    rows = []

    for det in detections:
        if det["cnt"] is not None:
            color = classify_color(hsv, det, masks)
        else:
            color = classify_color_in_circle(hsv, det["cx"], det["cy"], det["r"], COLOR_RANGES)

        cx, cy = int(det["cx"]), int(det["cy"])
        r = int(det["r"])
        bgr_col = VIS_COLORS.get(color, VIS_COLORS["mixed"])

        cv2.circle(vis, (cx, cy), r, bgr_col, 2)
        cv2.putText(vis, color, (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_col, 1, cv2.LINE_AA)

        rows.append({
            "id": det["id"],
            "x": cx,
            "y": cy,
            "radius_px": det["r"],
            "area_px": det["area"],
            "circularity": det["circularity"],
            "color": color
        })

    df = pd.DataFrame(rows)
    counts = df["color"].value_counts().to_dict() if not df.empty else {}
    _draw_legend(vis, counts)

    if out_png:
        cv2.imwrite(out_png, vis)
    if out_csv:
        # 1) részletes táblázat (változatlan)
        df.to_csv(out_csv, index=False)

        # 2) summary csv: per-color + összeg + átlagok + paraméterek
        summary = {
            "total": int(len(df)),
            "avg_radius_px": float(df["radius_px"].mean()) if not df.empty else 0.0,
            "avg_area_px": float(df["area_px"].mean()) if not df.empty else 0.0,
            "circularity_mean": float(df["circularity"].mean()) if not df.empty else 0.0,
            "counts": counts,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        # paraméterek egyszerű dumpja
        try:
            from dataclasses import asdict
            summary["params"] = asdict(params_attached)  # lásd lent: params átadása
        except Exception:
            pass

        import os
        stem, ext = os.path.splitext(out_csv)
        pd.DataFrame([{
            **{f"count_{k}": v for k, v in counts.items()},
            **{k: v for k, v in summary.items() if k not in ["counts","params"]},
            "params_json": json.dumps(summary.get("params", {})),
        }]).to_csv(stem + "_summary.csv", index=False)

    return vis, df, counts


def run(image_path: str, params: Params, out_png: str, out_csv: str):
    global params_attached
    params_attached = params  # így a metrikák exportja el fogja érni
    bgr = cv2.imread(image_path)

    if bgr is None:
        raise FileNotFoundError(image_path)

    bgr_p, hsv, scale = preprocess(bgr, params)

    if params.use_hough:
        # Színtől független kördetektálás; a színt a kör belsejéből becsüljük
        detections = find_candies_hough(bgr_p, hsv, params)
        masks = make_color_masks(hsv)  # legenda miatt hasznos
        maybe_save_masks(masks, out_png)
        mask_all = None
    else:
        masks = make_color_masks(hsv)
        # (opcionális) debug maszkok
        # _debug_save_masks(masks, os.path.join(os.path.dirname(out_png), "masks"))

        mask_all = morphology(combine_masks(masks), params)
        if params.use_watershed:
            detections, _ = watershed_split(bgr_p, mask_all, params)
        else:
            detections = find_candies(bgr_p, mask_all, params)

    vis, df, counts = annotate_and_export(bgr_p, hsv, detections, masks, out_png, out_csv)
    return vis, df, mask_all, counts, scale
