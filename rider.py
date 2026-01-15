import argparse
import os
import time
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

try:
    import pyrealsense2 as rs
except ImportError:
    raise SystemExit("pyrealsense2 not found. Install Intel RealSense SDK + python bindings.")


def rotate_depth_and_intrinsics(depth_m: np.ndarray, fx, fy, ppx, ppy, deg: int):
    deg = int(deg) % 360
    if deg == 0:
        return depth_m, fx, fy, ppx, ppy

    H, W = depth_m.shape

    if deg == 180:
        out = np.rot90(depth_m, k=2)
        return out, fx, fy, (W - 1) - ppx, (H - 1) - ppy

    if deg == 90:
        out = np.rot90(depth_m, k=-1)
        fx2, fy2 = fy, fx
        ppx2 = ppy
        ppy2 = (W - 1) - ppx
        return out, fx2, fy2, ppx2, ppy2

    if deg == 270:
        out = np.rot90(depth_m, k=1)
        fx2, fy2 = fy, fx
        ppx2 = (H - 1) - ppy
        ppy2 = ppx
        return out, fx2, fy2, ppx2, ppy2

    raise ValueError("deg must be one of 0,90,180,270")


def fit_plane_svd(P: np.ndarray):
    c = P.mean(axis=0)
    Q = P - c
    _, _, vh = np.linalg.svd(Q, full_matrices=False)
    n = vh[-1]
    n = n / (np.linalg.norm(n) + 1e-12)
    return c, n


def make_plane_frame(c: np.ndarray, n: np.ndarray):
    w = n / (np.linalg.norm(n) + 1e-12)
    a = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(a, w)) > 0.9:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u = np.cross(a, w); u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(w, u); v /= (np.linalg.norm(v) + 1e-12)
    return c.astype(np.float64), u.astype(np.float64), v.astype(np.float64), w.astype(np.float64)


def depth_to_points(depth_m: np.ndarray, fx, fy, ppx, ppy):
    H, W = depth_m.shape
    uu = np.arange(W, dtype=np.float32)
    vv = np.arange(H, dtype=np.float32)
    u, v = np.meshgrid(uu, vv)
    z = depth_m
    x = (u - ppx) * z / fx
    y = (v - ppy) * z / fy
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


def grid_mean(u: np.ndarray, v: np.ndarray, h: np.ndarray, cell: float, umin, umax, vmin, vmax):
    nu = int(np.floor((umax - umin) / cell)) + 1
    nv = int(np.floor((vmax - vmin) / cell)) + 1

    iu = np.floor((u - umin) / cell).astype(np.int32)
    iv = np.floor((v - vmin) / cell).astype(np.int32)

    valid = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv) & np.isfinite(h)
    iu = iu[valid]; iv = iv[valid]; hh = h[valid]

    lin = iv.astype(np.int64) * nu + iu.astype(np.int64)
    count = np.bincount(lin, minlength=nu * nv).astype(np.float64)
    summ = np.bincount(lin, weights=hh.astype(np.float64), minlength=nu * nv)

    dem = np.full(nu * nv, np.nan, dtype=np.float32)
    m = count > 0
    dem[m] = (summ[m] / count[m]).astype(np.float32)
    return dem.reshape(nv, nu)


def hillshade(elev_m: np.ndarray, cell_m: float, azimuth_deg=315.0, altitude_deg=45.0):
    dz_dy, dz_dx = np.gradient(elev_m, cell_m, cell_m)
    slope = np.pi / 2 - np.arctan(np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy))
    aspect = np.arctan2(-dz_dy, dz_dx)
    az = np.deg2rad(azimuth_deg)
    alt = np.deg2rad(altitude_deg)
    hs = (np.sin(alt) * np.sin(slope) +
          np.cos(alt) * np.cos(slope) * np.cos(az - aspect))
    return np.clip(hs, 0, 1)


def robust_to_u8(img01: np.ndarray, p_lo=2, p_hi=98):
    finite = img01[np.isfinite(img01)]
    if finite.size < 200:
        a, b = float(np.nanmin(img01)), float(np.nanmax(img01))
    else:
        a, b = np.percentile(finite, [p_lo, p_hi])
    x = np.clip((img01 - a) / (b - a + 1e-12), 0, 1)
    return (x * 255).astype(np.uint8)


def nan_fill_no_wrap(dem: np.ndarray, iters=2):
    out = dem.astype(np.float32).copy()
    for _ in range(iters):
        mask = np.isfinite(out).astype(np.float32)
        if mask.sum() == mask.size:
            break
        out0 = np.nan_to_num(out, nan=0.0).astype(np.float32)
        num = cv2.blur(out0, ksize=(3, 3), borderType=cv2.BORDER_REFLECT)
        den = cv2.blur(mask, ksize=(3, 3), borderType=cv2.BORDER_REFLECT)
        filled = np.where(den > 1e-6, num / den, np.nan).astype(np.float32)
        out = np.where(np.isfinite(out), out, filled)
    return out


def rect_to_mask(shape, rect):
    h, w = shape
    x0, y0, x1, y1 = rect
    x0, x1 = int(np.clip(min(x0, x1), 0, w - 1)), int(np.clip(max(x0, x1), 0, w - 1))
    y0, y1 = int(np.clip(min(y0, y1), 0, h - 1)), int(np.clip(max(y0, y1), 0, h - 1))
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1 + 1, x0:x1 + 1] = True
    return m


def volume_change_roi(dem_now: np.ndarray, dem_base: np.ndarray, cell_m: float,
                      meas_mask: np.ndarray, ref_mask: np.ndarray | None, sign_flip: float):
    valid = np.isfinite(dem_now) & np.isfinite(dem_base) & meas_mask
    if valid.sum() < 50:
        return np.nan, np.nan, np.nan, np.nan

    dh0 = (dem_now - dem_base).astype(np.float64) * float(sign_flip)

    bias = 0.0
    if ref_mask is not None:
        ref_valid = np.isfinite(dem_now) & np.isfinite(dem_base) & ref_mask
        if ref_valid.sum() >= 50:
            bias = float(np.median(dh0[ref_valid]))

    dh = dh0 - bias
    A = float(cell_m * cell_m)
    net = float(np.nansum(dh[valid]) * A)

    pos = dh[valid] > 0
    neg = dh[valid] < 0
    fill = float(np.nansum(dh[valid][pos]) * A)
    cut = float(-np.nansum(dh[valid][neg]) * A)
    return net, cut, fill, bias


def append_csv(path, row, header):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)


def update_plot(csv_path, out_png):
    ts, net, cut, fill = [], [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            ts.append(row["timestamp"])
            net.append(float(row["net_m3"]))
            cut.append(float(row["cut_m3"]))
            fill.append(float(row["fill_m3"]))

    if len(net) < 1:
        return

    x = np.arange(len(net))
    plt.figure()
    plt.plot(x, net, marker="o", label="Net (m³)")
    plt.plot(x, cut, marker="o", label="Cut magnitude (m³)")
    plt.plot(x, fill, marker="o", label="Fill (m³)")
    plt.xlabel("Snapshot #")
    plt.ylabel("Volume (m³)")
    plt.title("ROI volume change (relative to run start; bias from reference ROI)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def next_run_dir(root: str) -> str:
    os.makedirs(root, exist_ok=True)
    nums = []
    for name in os.listdir(root):
        m = __import__("re").match(r"^run(\d{2})$", name)
        if m:
            nums.append(int(m.group(1)))
    n = (max(nums) + 1) if nums else 1
    return os.path.join(root, f"run{n:02d}")


def render_hillshade_bgr(dem: np.ndarray, cell_m: float, az: float, alt: float):
    dem_disp = nan_fill_no_wrap(dem, iters=2)
    hs = hillshade(dem_disp, cell_m, az, alt)
    hs8 = robust_to_u8(hs)
    return cv2.cvtColor(hs8, cv2.COLOR_GRAY2BGR)


def render_elevation_bgr(dem: np.ndarray, cell_m: float, az: float, alt: float):
    dem_disp = nan_fill_no_wrap(dem, iters=2)
    hs = hillshade(dem_disp, cell_m, az, alt)
    hs8 = robust_to_u8(hs)

    finite = dem_disp[np.isfinite(dem_disp)]
    if finite.size > 200:
        lo, hi = np.percentile(finite, [2, 98])
    else:
        lo, hi = float(np.nanmin(dem_disp)), float(np.nanmax(dem_disp))

    norm = np.clip((dem_disp - lo) / (hi - lo + 1e-12), 0, 1)
    cm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    light = (0.35 + 0.65 * (hs8.astype(np.float32) / 255.0))[:, :, None]
    return np.clip(cm.astype(np.float32) * light, 0, 255).astype(np.uint8)


def write_png(path: str, bgr: np.ndarray):
    cv2.imwrite(path, bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])


class ROIState:
    def __init__(self):
        self.mode = None
        self.dragging = False
        self.x0 = self.y0 = self.x1 = self.y1 = 0
        self.meas_rect = None
        self.ref_rect = None


def mouse_cb(event, x, y, flags, param):
    st: ROIState = param
    if st.mode is None:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        st.dragging = True
        st.x0, st.y0 = x, y
        st.x1, st.y1 = x, y
    elif event == cv2.EVENT_MOUSEMOVE and st.dragging:
        st.x1, st.y1 = x, y
    elif event == cv2.EVENT_LBUTTONUP and st.dragging:
        st.dragging = False
        st.x1, st.y1 = x, y
        rect = (st.x0, st.y0, st.x1, st.y1)
        if st.mode == "meas":
            st.meas_rect = rect
        elif st.mode == "ref":
            st.ref_rect = rect
        st.mode = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="runs")
    ap.add_argument("--w", type=int, default=848)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--rotate", type=int, default=90, help="0, 90, 180, 270")
    ap.add_argument("--rs_filters", action="store_true")
    ap.add_argument("--stride", type=int, default=1, help="1 for max precision; 2-3 for speed")
    ap.add_argument("--zmin", type=float, default=0.2)
    ap.add_argument("--zmax", type=float, default=2.5)
    ap.add_argument("--cell", type=float, default=0.003)
    ap.add_argument("--az", type=float, default=315.0)
    ap.add_argument("--alt", type=float, default=45.0)
    ap.add_argument("--update_hz", type=float, default=10.0)
    ap.add_argument("--plane_samples", type=int, default=50000)
    ap.add_argument("--pad", type=float, default=0.10)
    args = ap.parse_args()

    run_dir = next_run_dir(args.out)
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, "volume_timeseries.csv")
    plot_path = os.path.join(run_dir, "volume_plot.png")

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, args.w, args.h, rs.format.z16, args.fps)
    profile = pipeline.start(cfg)

    spat = rs.spatial_filter() if args.rs_filters else None
    temp = rs.temporal_filter() if args.rs_filters else None
    hole = rs.hole_filling_filter() if args.rs_filters else None

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    intr = stream.get_intrinsics()
    fx, fy, ppx, ppy = intr.fx, intr.fy, intr.ppx, intr.ppy

    print(f"Depth scale: {depth_scale} m/unit")
    print(f"Output: {run_dir}")
    print("Keys: l relock | m meas ROI | r ref ROI | s snapshot | y undo | f flip | v vis | o rotate | q quit")

    window = "RIDER DEM"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    roi = ROIState()
    cv2.setMouseCallback(window, mouse_cb, roi)

    locked = False
    plane_c = plane_u = plane_v = plane_w = None
    bounds = None

    baseline_dem = None
    baseline_ts = None
    meas_mask = None
    ref_mask = None
    sign_flip = 1.0

    vis_mode = 0  # 0 hillshade, 1 elevation, 2 delta
    rotate_deg = int(args.rotate) % 360

    last_compute_t = 0.0
    last_dem = None
    last_vis = None
    last_hill_clean = None
    last_elev_clean = None

    snap_idx = 0
    snap_history = []

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth = frames.get_depth_frame()
            if not depth:
                continue

            now = time.time()
            if now - last_compute_t >= (1.0 / max(0.5, args.update_hz)):
                last_compute_t = now

                d = depth
                if spat is not None:
                    d = spat.process(d)
                if temp is not None:
                    d = temp.process(d)
                if hole is not None:
                    d = hole.process(d)

                depth_raw = np.asanyarray(d.get_data()).astype(np.float32)
                depth_m = depth_raw * float(depth_scale)

                depth_m, fx_r, fy_r, ppx_r, ppy_r = rotate_depth_and_intrinsics(
                    depth_m, fx, fy, ppx, ppy, rotate_deg
                )

                s = max(1, int(args.stride))
                if s > 1:
                    depth_m = depth_m[::s, ::s]
                    fx_s, fy_s, ppx_s, ppy_s = fx_r / s, fy_r / s, ppx_r / s, ppy_r / s
                else:
                    fx_s, fy_s, ppx_s, ppy_s = fx_r, fy_r, ppx_r, ppy_r

                z = depth_m
                m = (z > args.zmin) & (z < args.zmax) & np.isfinite(z)
                if m.sum() < 800:
                    if last_vis is not None:
                        cv2.imshow(window, last_vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    continue

                P = depth_to_points(depth_m, fx_s, fy_s, ppx_s, ppy_s)
                P = P[m.reshape(-1)].astype(np.float64)

                if not locked:
                    npts = P.shape[0]
                    k = min(args.plane_samples, npts)
                    idx = np.random.choice(npts, k, replace=False) if npts > k else np.arange(npts)
                    c_cam, n_cam = fit_plane_svd(P[idx])
                    if n_cam[1] > 0:
                        n_cam = -n_cam
                    plane_c, plane_u, plane_v, plane_w = make_plane_frame(c_cam, n_cam)

                    Qs = P - plane_c
                    uu = Qs @ plane_u
                    vv = Qs @ plane_v
                    umin, umax = np.percentile(uu, [1, 99])
                    vmin, vmax = np.percentile(vv, [1, 99])
                    pad = float(args.pad)
                    bounds = (float(umin - pad), float(umax + pad), float(vmin - pad), float(vmax + pad))

                    locked = True
                    meas_mask = None
                    ref_mask = None
                    print(f"Locked. DEM size ≈ {(bounds[1]-bounds[0]):.3f} m × {(bounds[3]-bounds[2]):.3f} m")

                Q = P - plane_c
                ucoord = Q @ plane_u
                vcoord = Q @ plane_v
                elev = Q @ plane_w

                umin, umax, vmin, vmax = bounds
                dem = grid_mean(ucoord, vcoord, elev, args.cell, umin, umax, vmin, vmax)
                last_dem = dem

                h, w = dem.shape
                if roi.meas_rect is not None:
                    meas_mask = rect_to_mask((h, w), roi.meas_rect)
                if roi.ref_rect is not None:
                    ref_mask = rect_to_mask((h, w), roi.ref_rect)

                hill_clean = render_hillshade_bgr(dem, args.cell, args.az, args.alt)
                elev_clean = render_elevation_bgr(dem, args.cell, args.az, args.alt)
                last_hill_clean = hill_clean
                last_elev_clean = elev_clean

                if vis_mode == 0:
                    vis = hill_clean.copy()
                elif vis_mode == 1:
                    vis = elev_clean.copy()
                else:
                    if baseline_dem is not None and baseline_dem.shape == dem.shape:
                        dem_disp = nan_fill_no_wrap(dem, iters=2)
                        hs = hillshade(dem_disp, args.cell, args.az, args.alt)
                        hs8 = robust_to_u8(hs)
                        dh = (dem - baseline_dem).astype(np.float32) * float(sign_flip)
                        finite = dh[np.isfinite(dh)]
                        s99 = float(np.percentile(np.abs(finite), 99)) if finite.size > 200 else float(np.nanmax(np.abs(dh)))
                        s99 = max(s99, 1e-6)
                        norm = np.clip((dh / s99 + 1.0) * 0.5, 0, 1)
                        cm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TWILIGHT)
                        light = (0.35 + 0.65 * (hs8.astype(np.float32) / 255.0))[:, :, None]
                        vis = np.clip(cm.astype(np.float32) * light, 0, 255).astype(np.uint8)
                        cv2.putText(vis, f"Δ vs base (±{s99*1000:.1f} mm @99%)", (10, 48),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    else:
                        vis = hill_clean.copy()

                if meas_mask is not None:
                    x0, y0, x1, y1 = roi.meas_rect
                    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    cv2.putText(vis, "MEASURE", (min(x0, x1) + 5, min(y0, y1) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if ref_mask is not None:
                    x0, y0, x1, y1 = roi.ref_rect
                    cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 255), 2)
                    cv2.putText(vis, "REFERENCE", (min(x0, x1) + 5, min(y0, y1) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                cv2.putText(vis, f"LOCKED | sign={int(sign_flip)}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 80), 2)

                if roi.dragging:
                    cv2.rectangle(vis, (roi.x0, roi.y0), (roi.x1, roi.y1), (200, 200, 50), 1)

                last_vis = vis
                cv2.imshow(window, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            if key == ord('y'):
                if not snap_history:
                    print("Nothing to undo.")
                    continue

                last = snap_history.pop()
                for f in last.get("files", []):
                    p = os.path.join(run_dir, f)
                    if os.path.exists(p):
                        os.remove(p)

                if os.path.exists(csv_path):
                    with open(csv_path, "r", encoding="utf-8") as f:
                        lines = [ln for ln in f.read().splitlines() if ln.strip() != ""]
                    if len(lines) > 1:
                        lines = lines[:-1]
                        with open(csv_path, "w", encoding="utf-8", newline="") as f:
                            f.write("\n".join(lines) + "\n")
                    else:
                        os.remove(csv_path)

                if os.path.exists(csv_path):
                    update_plot(csv_path, plot_path)
                else:
                    if os.path.exists(plot_path):
                        os.remove(plot_path)

                if last.get("was_baseline", False):
                    baseline_dem = None
                    baseline_ts = None
                    print("Baseline undone. Next snapshot will re-zero volume.")

                snap_idx = max(0, snap_idx - 1)
                print("Last snapshot undone.")

            if key == ord('l'):
                locked = False
                roi.meas_rect = None
                roi.ref_rect = None
                meas_mask = None
                ref_mask = None
                print("Relock requested.")

            if key == ord('m') and last_dem is not None:
                roi.mode = "meas"
                print("Draw MEASUREMENT ROI (mouse drag).")

            if key == ord('r') and last_dem is not None:
                roi.mode = "ref"
                print("Draw REFERENCE ROI (mouse drag).")

            if key == ord('f'):
                sign_flip *= -1.0
                print(f"Sign flipped: {sign_flip}")

            if key == ord('v'):
                vis_mode = (vis_mode + 1) % 3
                print(f"Visualization: {['hillshade','elevation','delta'][vis_mode]}")

            if key == ord('o'):
                rotate_deg = 0 if rotate_deg != 0 else 90
                locked = False
                roi.meas_rect = None
                roi.ref_rect = None
                meas_mask = None
                ref_mask = None
                print(f"Rotation set to {rotate_deg}°; relock next frame.")

            if key == ord('s'):
                if last_dem is None or last_hill_clean is None or last_elev_clean is None:
                    print("No frame ready yet.")
                    continue

                ts = time.strftime("%Y%m%d_%H%M%S")
                base = f"snap_{ts}_{snap_idx:04d}"

                saved_files = []
                write_png(os.path.join(run_dir, base + "_hillshade.png"), last_hill_clean)
                saved_files.append(base + "_hillshade.png")
                write_png(os.path.join(run_dir, base + "_elevation.png"), last_elev_clean)
                saved_files.append(base + "_elevation.png")
                np.save(os.path.join(run_dir, base + "_dem.npy"), last_dem)
                saved_files.append(base + "_dem.npy")

                if baseline_dem is None:
                    baseline_dem = last_dem.copy()
                    baseline_ts = ts

                    append_csv(
                        csv_path,
                        [ts, snap_idx, 0.0, 0.0, 0.0, 0.0, args.cell, args.stride, args.zmin, args.zmax, baseline_ts,
                         0, 0],
                        header=["timestamp", "snap_idx", "net_m3", "cut_m3", "fill_m3", "bias_m",
                                "cell_m", "stride", "zmin_m", "zmax_m", "zero_timestamp",
                                "ref_cells", "meas_cells"]
                    )
                    update_plot(csv_path, plot_path)

                    snap_history.append({"files": saved_files, "was_baseline": True})

                    print(f"Saved {base}. Volume zero initialized.")
                    snap_idx += 1
                    continue

                if meas_mask is None:
                    print("No MEASUREMENT ROI set. Press 'm' and draw a box.")
                    snap_idx += 1
                    continue

                net, cut, fill, bias = volume_change_roi(
                    last_dem, baseline_dem, args.cell,
                    meas_mask=meas_mask, ref_mask=ref_mask, sign_flip=sign_flip
                )

                append_csv(
                    csv_path,
                    [ts, snap_idx, net, cut, fill, bias, args.cell, args.stride, args.zmin, args.zmax, baseline_ts,
                     int(ref_mask.sum()) if ref_mask is not None else 0, int(meas_mask.sum())],
                    header=["timestamp", "snap_idx", "net_m3", "cut_m3", "fill_m3", "bias_m",
                            "cell_m", "stride", "zmin_m", "zmax_m", "zero_timestamp",
                            "ref_cells", "meas_cells"]
                )
                update_plot(csv_path, plot_path)

                snap_history.append({"files": saved_files, "was_baseline": False})

                print(f"Saved {base} (hillshade + elevation + dem). ROI volume (relative to run start):")
                print(f"  net={net:.6e} m^3   cut={cut:.6e}   fill={fill:.6e}   bias(ref)={bias:.3e} m")
                print(f"  plot: {plot_path}")

                snap_idx += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

