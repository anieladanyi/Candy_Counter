import argparse, os
from src.candy_pipeline import Params, run

def apply_preset(p, name):
    p.use_hough = True
    if name == "clean":
        p.hough_dp = 1.0
        p.hough_minDist = 80
        p.hough_param1 = 95
        p.hough_param2 = 22
        p.hough_minRadius = 30
        p.hough_maxRadius = 100
        p.min_area = 130
        p.blur_ksize = 3
        p.use_watershed = False
    elif name == "balanced":
        p.hough_dp = 0.95
        p.hough_minDist = 72
        p.hough_param1 = 95
        p.hough_param2 = 18
        p.hough_minRadius = 30
        p.hough_maxRadius = 105
        p.min_area = 110
        p.blur_ksize = 3
    elif name == "aggressive":
        p.hough_dp = 0.9
        p.hough_minDist = 60
        p.hough_param1 = 90
        p.hough_param2 = 14
        p.hough_minRadius = 28
        p.hough_maxRadius = 110
        p.min_area = 90
        p.blur_ksize = 3
    elif name == "stuck":
        p.use_watershed = True
        p.dt_thresh_ratio = 0.45
        p.morph_open = 5
        p.morph_close = 5
        p.use_hough = False
    return p

def main():
    ap = argparse.ArgumentParser(description="Színes cukorkaszámláló és -osztályozó")

    ap.add_argument("--input", "-i", required=True, help="Bemeneti kép (pl. data/samples/candy.jpg)")
    ap.add_argument("--outdir", "-o", default="exports", help="Kimeneti mappa")
    ap.add_argument("--min-area", type=int, default=150)
    ap.add_argument("--circ-min", type=float, default=0.6)
    ap.add_argument("--open", type=int, default=3)
    ap.add_argument("--close", type=int, default=3)
    ap.add_argument("--blur", type=int, default=5)
    ap.add_argument("--watershed", action="store_true")
    ap.add_argument("--dt-thresh", type=float, default=0.4)

    ap.add_argument("--hough", action="store_true")
    ap.add_argument("--h-dp", type=float, default=1.2)
    ap.add_argument("--h-minDist", type=int, default=60)
    ap.add_argument("--h-p1", type=int, default=100)
    ap.add_argument("--h-p2", type=int, default=30)
    ap.add_argument("--h-minR", type=int, default=20)
    ap.add_argument("--h-maxR", type=int, default=120)

    ap.add_argument("--preset", choices=["clean", "balanced", "aggressive", "stuck"])
    ap.add_argument("--save-masks", action="store_true")

    args = ap.parse_args()

    p = Params(
        blur_ksize=args.blur,
        morph_open=args.open,
        morph_close=args.close,
        min_area=args.min_area,
        circularity_min=args.circ_min,
        use_watershed=args.watershed,
        dt_thresh_ratio=args.dt_thresh,
        use_hough=args.hough,
        hough_dp=args.h_dp,
        hough_minDist=args.h_minDist,
        hough_param1=args.h_p1,
        hough_param2=args.h_p2,
        hough_minRadius=args.h_minR,
        hough_maxRadius=args.h_maxR
    )

    if args.preset:
        p = apply_preset(p, args.preset)

    if args.save_masks:
        os.environ["SAVE_MASKS"] = "1"

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]
    out_png = os.path.join(args.outdir, f"{stem}_vis.png")
    out_csv = os.path.join(args.outdir, f"{stem}_metrics.csv")

    _, df, _, counts, _ = run(args.input, p, out_png, out_csv)

    if df is None or df.empty:
        print("[WARN] Nincs detektált cukorka. Próbáld lazítani a paramétereket vagy használj --preset stuck")
    print("[INFO] Darabszám színenként:", counts)
    print("[INFO] Mentve:", out_png, "és", out_csv)

if __name__ == "__main__":
    main()
