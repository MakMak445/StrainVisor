"""
Runs py2DIC's Main.DIC() headlessly (no GUI) against a single AOI of a single
test's raw TIFF frame sequence, saving the incremental dx/dy displacement
grids into <output_dir>/OutputPlots/ (py2DIC hardcodes that subfolder name,
so this script chdir's into output_dir before calling it).

Two AOIs are used for the dual-AOI trajectory pipeline:
  - "between_bars": the bar-to-bar gap (specimen region)
  - "under_bar": a thin strip spanning the full frame width, from the bottom
    of the bar down to the top of the on-frame timestamp overlay

Usage:
  python run_dual_aoi_dic.py \
      --images "/path/to/Camera Njord/Njord_10_59_53" \
      --rectx1 80 --rectx2 188 --recty1 0 --recty2 196 \
      --output_dir "/path/to/DIC_v2/Njord_10_59_53/between_bars" \
      --temp_dim 11 --b 15 --d 15 --ml 1 --levels 15 --sampling 8
"""
import argparse
import os
import sys
import time

PY2DIC_SOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "py2DIC-master", "sources"
)


def parse_args():
    p = argparse.ArgumentParser(description="Headless single-AOI py2DIC run.")
    p.add_argument("--images", required=True, help="Folder of raw .tiff frames for the test")
    p.add_argument("--output_dir", required=True, help="Where OutputPlots/ will be created")
    p.add_argument("--rectx1", type=int, required=True)
    p.add_argument("--rectx2", type=int, required=True)
    p.add_argument("--recty1", type=int, required=True)
    p.add_argument("--recty2", type=int, required=True)
    p.add_argument("--dim_pixel", type=float, default=1.0, help="mm-per-pixel calibration; 1.0 keeps units as raw pixels")
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--levels", type=int, default=15, help="Number of chained increments to compute")
    p.add_argument("--sampling", type=int, default=8, help="Frames between each chained comparison")
    p.add_argument("--temp_dim", type=int, default=11, help="Template subset width [pixel]")
    p.add_argument("--b", type=int, default=15, help="Vertical search margin around template [pixel]")
    p.add_argument("--d", type=int, default=15, help="Horizontal search margin around template [pixel]")
    p.add_argument("--ml", type=int, default=1, help="Sub-pixel upsampling factor (py2DIC default is 10; 1 = pixel-level, much faster)")
    p.add_argument("--defor", action="store_true", help="Also compute strain fields (slower, not needed for trajectory tracking)")
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, PY2DIC_SOURCES)

    # py2DIC's DIC() unconditionally calls plt.switch_backend("Qt5Agg"), which
    # throws in a headless environment with no display. Neutralize it before
    # importing Main, since we only need the numeric dx/dy output, not any
    # interactive plot.
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    plt.switch_backend = lambda *a, **k: None

    import Main as DIC_main  # noqa: E402  (py2DIC-master/sources/Main.py)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "OutputPlots"), exist_ok=True)

    H = 2 * args.d + args.temp_dim
    V = 2 * args.b + args.temp_dim

    print(f"Images: {args.images}")
    print(f"AOI: x=[{args.rectx1},{args.rectx2}]  y=[{args.recty1},{args.recty2}]")
    print(f"temp_dim={args.temp_dim} b={args.b} d={args.d} -> H={H} V={V}  ml={args.ml}")
    print(f"levels={args.levels} sampling={args.sampling} dim_pixel={args.dim_pixel}")
    print(f"Output: {args.output_dir}/OutputPlots/")

    prev_cwd = os.getcwd()
    os.chdir(args.output_dir)
    try:
        start = time.time()
        msg, _ = DIC_main.DIC(
            args.images, args.dim_pixel, args.start_index, args.levels,
            args.sampling, args.temp_dim, args.b, args.d,
            args.recty1, args.recty2, args.rectx1, args.rectx2,
            H, V, args.defor, args.ml,
        )
        elapsed = time.time() - start
        print(msg)
        print(f"\nDone in {elapsed:.1f}s")
    finally:
        os.chdir(prev_cwd)


if __name__ == "__main__":
    main()
