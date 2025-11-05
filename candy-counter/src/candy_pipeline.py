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
    # A vörös továbbra is két sávból áll, így lefedi a 0 körüli H értékeket
    "red1":   ((0,   36, 30), (8, 255, 255)),
    "red2":   ((168, 36, 30), (179, 255, 255)),

    # Narancs és sárga határértékek: lazább S/V, hogy a világosabb cukorkák és fakó sarkok se essenek ki
    "orange": ((7,   24, 40), (22, 255, 255)),
    "yellow": ((18,  18, 55), (45, 255, 255)),

    # A többi színhez elegendő a korábbi tartomány, de engedünk a minimális telítettségen/fényességen
    "green":  ((50,  28, 40), (85, 255, 255)),
    "blue":   ((90,  28, 40), (125,255, 255)),
    "purple": ((125, 28, 40), (170,255, 255)),
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
    min_area: int = 120
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

def _filter_and_pack_contours(contours, bgr, p: Params, *, color_hint=None, source="contour"):
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
            "circularity": circularity,
            "color_hint": color_hint,
            "source": source
        })
    return detections

def find_candies(bgr, mask_all, p: Params):
    contours = _contours_from_binary(mask_all)
    return _filter_and_pack_contours(contours, bgr, p)


def find_candies_per_color(bgr, masks, p: Params):
    detections = []
    for cname, cmask in masks.items():
        if cv2.countNonZero(cmask) == 0:
            continue
        cleaned = morphology(cmask, p)
        contours = _contours_from_binary(cleaned)
        if not contours:
            continue
        detections.extend(
            _filter_and_pack_contours(
                contours,
                bgr,
                p,
                color_hint=cname,
                source=f"mask_{cname}"
            )
        )
    return detections

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


def _merge_detections(*det_lists, center_thresh_ratio=0.45):
    merged = []

    for dets in det_lists:
        if not dets:
            continue
        for det in dets:
            candidate = dict(det)
            keep = True
            for idx, kept in enumerate(merged):
                dx = candidate["cx"] - kept["cx"]
                dy = candidate["cy"] - kept["cy"]
                dist = (dx * dx + dy * dy) ** 0.5
                min_r = max(1e-6, min(candidate["r"], kept["r"]))
                if dist < center_thresh_ratio * min_r:
                    cand_has_cnt = candidate.get("cnt") is not None
                    kept_has_cnt = kept.get("cnt") is not None
                    cand_has_hint = candidate.get("color_hint") is not None
                    kept_has_hint = kept.get("color_hint") is not None
                    if cand_has_cnt and not kept_has_cnt:
                        merged[idx] = candidate
                    elif cand_has_hint and not kept_has_hint:
                        merged[idx] = candidate
                    elif cand_has_cnt == kept_has_cnt and cand_has_hint == kept_has_hint and \
                            candidate.get("area", 0) > kept.get("area", 0):
                        merged[idx] = candidate
                    keep = False
                    break
            if keep:
                merged.append(candidate)

    for idx, det in enumerate(merged):
        det["id"] = idx
    return merged


def _rescale_detection(det, scale):
    if scale == 1.0:
        return dict(det)

    inv = 1.0 / max(scale, 1e-6)
    scaled = dict(det)
    scaled["cx"] = det["cx"] * inv
    scaled["cy"] = det["cy"] * inv
    scaled["r"] = det["r"] * inv
    scaled["area"] = det["area"] * (inv * inv)

    if det.get("bbox") is not None:
        x, y, w, h = det["bbox"]
        scaled["bbox"] = (x * inv, y * inv, w * inv, h * inv)

    cnt = det.get("cnt")
    if cnt is not None:
        scaled["cnt"] = (cnt.astype(np.float32) * inv).astype(cnt.dtype)

    return scaled


def find_candies_hough(bgr, hsv, p: Params):
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
                "circularity": 1.0,
                "color_hint": None,
                "source": "hough"
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
        detections.extend(_filter_and_pack_contours([cnt], bgr, p, source="watershed"))
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

def classify_color_in_circle(hsv, cx, cy, r, masks):
    region_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.circle(region_mask, (int(cx), int(cy)), max(int(r * 0.88), 1), 255, -1)

    S = hsv[:, :, 1]; V = hsv[:, :, 2]
    sv_mask = ((S >= 24) & (V >= 42)).astype(np.uint8) * 255
    region_mask = cv2.bitwise_and(region_mask, sv_mask)

    best_color, best_hits = "mixed", 0
    for cname, cmask in masks.items():
        hits = cv2.countNonZero(cv2.bitwise_and(cmask, region_mask))
        if hits > best_hits:
            best_hits = hits
            best_color = cname

    if best_hits > 0:
        return best_color

    H = hsv[:, :, 0]
    hist = cv2.calcHist([H], [0], region_mask, [180], [0, 180]).flatten()
    if hist.sum() < 1:
        return "mixed"
    peak_h = int(hist.argmax())

    def in_band(h, lo, hi):
        return lo <= h <= hi

    if any(in_band(peak_h, rng[0][0], rng[1][0]) for rng in (COLOR_RANGES["red1"], COLOR_RANGES["red2"])):
        return "red"
    for cname in ["yellow", "orange", "green", "blue", "purple"]:
        lo, hi = COLOR_RANGES[cname][0][0], COLOR_RANGES[cname][1][0]
        if in_band(peak_h, lo, hi):
            return cname
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

def annotate_and_export(base_bgr, detections, *, out_png, out_csv, params, scale):
    vis = base_bgr.copy()
    rows = []

    for det in detections:
        color = det.get("color", "mixed")
        cx, cy = int(round(det["cx"])), int(round(det["cy"]))
        r = max(1, int(round(det["r"])))
        bgr_col = VIS_COLORS.get(color, VIS_COLORS["mixed"])

        cv2.circle(vis, (cx, cy), r, bgr_col, 2)
        cv2.putText(vis, color, (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_col, 1, cv2.LINE_AA)

        rows.append({
            "id": det["id"],
            "x": float(det["cx"]),
            "y": float(det["cy"]),
            "radius_px": float(det["r"]),
            "area_px": float(det["area"]),
            "circularity": float(det["circularity"]),
            "color": color,
            "source": det.get("source", "contour"),
        })

    df = pd.DataFrame(rows)
    counts = df["color"].value_counts().to_dict() if not df.empty else {}
    _draw_legend(vis, counts)

    if out_png:
        cv2.imwrite(out_png, vis)
    if out_csv:
        df.to_csv(out_csv, index=False)

        summary = {
            "total": int(len(df)),
            "avg_radius_px": float(df["radius_px"].mean()) if not df.empty else 0.0,
            "avg_area_px": float(df["area_px"].mean()) if not df.empty else 0.0,
            "circularity_mean": float(df["circularity"].mean()) if not df.empty else 0.0,
            "counts": counts,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scale_used": scale,
        }

        from dataclasses import asdict
        summary["params"] = asdict(params)

        import os
        stem, _ = os.path.splitext(out_csv)
        pd.DataFrame([{
            **{f"count_{k}": v for k, v in counts.items()},
            **{k: v for k, v in summary.items() if k not in ["counts", "params"]},
            "params_json": json.dumps(summary.get("params", {})),
        }]).to_csv(stem + "_summary.csv", index=False)

    return vis, df, counts


def run(image_path: str, params: Params, out_png: str, out_csv: str):
    bgr = cv2.imread(image_path)

    if bgr is None:
        raise FileNotFoundError(image_path)

    bgr_proc, hsv, scale = preprocess(bgr, params)

    masks = make_color_masks(hsv)
    maybe_save_masks(masks, out_png)

    combined_mask = combine_masks(masks)
    mask_all = morphology(combined_mask, params)

    color_dets = find_candies_per_color(bgr_proc, masks, params)

    if params.use_watershed:
        contour_dets, _ = watershed_split(bgr_proc, mask_all, params)
    else:
        contour_dets = find_candies(bgr_proc, mask_all, params)

    hough_dets = find_candies_hough(bgr_proc, hsv, params) if params.use_hough else []
    detections = _merge_detections(color_dets, contour_dets, hough_dets)

    for det in detections:
        if det.get("color_hint"):
            det["color"] = det["color_hint"].replace("mask_", "") if det["color_hint"].startswith("mask_") else det["color_hint"]
        elif det.get("cnt") is not None:
            det["color"] = classify_color(hsv, det, masks)
        else:
            det["color"] = classify_color_in_circle(hsv, det["cx"], det["cy"], det["r"], masks)

    rescaled = [_rescale_detection(det, scale) for det in detections]

    vis, df, counts = annotate_and_export(
        bgr,
        rescaled,
        out_png=out_png,
        out_csv=out_csv,
        params=params,
        scale=scale,
    )

    mask_export = mask_all
    if scale != 1.0:
        mask_export = cv2.resize(mask_all, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    return vis, df, mask_export, counts, scale
