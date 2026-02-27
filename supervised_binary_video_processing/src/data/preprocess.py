import cv2
import numpy as np

def largest_cc(mask: np.ndarray):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        return None, None
    # skip background at 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + np.argmax(areas)
    cc_mask = (labels == idx)
    return cc_mask, (stats[idx], centroids[idx])

def frame_features(gray: np.ndarray, pct=99.5):
    # blur
    g = cv2.GaussianBlur(gray, (3,3), 0)

    # threshold by percentile
    thr = np.percentile(g, pct)
    mask = (g >= thr)

    # clean
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k, iterations=1).astype(bool)

    main_mask, main_info = largest_cc(mask)
    if main_mask is None:
        return None

    stats, (cx, cy) = main_info
    x, y, w, h, area = stats[cv2.CC_STAT_LEFT], stats[cv2.CC_STAT_TOP], stats[cv2.CC_STAT_WIDTH], stats[cv2.CC_STAT_HEIGHT], stats[cv2.CC_STAT_AREA]
    aspect = w / (h + 1e-6)

    # spatter: small CCs excluding main blob
    sp_mask = mask & (~main_mask)
    num, labels, stats2, cents2 = cv2.connectedComponentsWithStats(sp_mask.astype(np.uint8), connectivity=8)
    sp_count = 0
    sp_area = 0
    sp_dist_sum = 0.0
    for i in range(1, num):
        a = stats2[i, cv2.CC_STAT_AREA]
        if 1 <= a <= 30:
            sp_count += 1
            sp_area += a
            sx, sy = cents2[i]
            sp_dist_sum += np.hypot(sx - cx, sy - cy)

    sp_mean_dist = (sp_dist_sum / sp_count) if sp_count > 0 else 0.0

    return {
        "area": float(area),
        "w": float(w),
        "h": float(h),
        "aspect": float(aspect),
        "cx": float(cx),
        "cy": float(cy),
        "sp_count": float(sp_count),
        "sp_area": float(sp_area),
        "sp_mean_dist": float(sp_mean_dist),
    }

def aggregate(ts: np.ndarray):
    # ts: (T,) array
    if len(ts) == 0:
        return [0,0,0,0,0]
    p10, p50, p90 = np.percentile(ts, [10,50,90])
    return [ts.mean(), ts.std(), p10, p50, p90]

def video_to_features(video_path: str, max_frames=2000, step=1):
    cap = cv2.VideoCapture(video_path)
    feats = []
    prev_small = None
    flow_mags = []

    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if t % step != 0:
            t += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        f = frame_features(gray, pct=99.5)
        if f is not None:
            feats.append(f)

        # optional optical flow on downsampled frames
        small = cv2.resize(gray, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        if prev_small is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_small, small, None,
                                                pyr_scale=0.5, levels=2, winsize=15,
                                                iterations=2, poly_n=5, poly_sigma=1.2, flags=0)
            mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
            flow_mags.append(np.mean(mag))
        prev_small = small

        t += 1
        if t >= max_frames:
            break

    cap.release()

    if len(feats) == 0:
        return np.zeros(68, dtype=np.float32)

    keys = list(feats[0].keys())
    mat = np.array([[f[k] for k in keys] for f in feats], dtype=np.float32)

    out = []
    for j, k in enumerate(keys):
        out += aggregate(mat[:, j])
        # diff-based stability
        d = np.diff(mat[:, j])
        out += [d.std() if len(d) else 0.0, np.max(np.abs(d)) if len(d) else 0.0]

    # flow aggregation
    flow_mags = np.array(flow_mags, dtype=np.float32)
    out += aggregate(flow_mags)

    return np.array(out, dtype=np.float32)