"""
star_engine.py
──────────────
Core pipeline: Star Detection → Feature Extraction → AI/Template Matching → Line Drawing
"""
import math
import cv2
import numpy as np
from itertools import combinations, permutations
from pathlib import Path
from constellation_data import STAR_CATALOG, get_star_positions_pixels

# ─────────────────────────────────────────────────────────────
# 1. STAR DETECTION
# ─────────────────────────────────────────────────────────────

def _prune_close_points(points, min_distance=20):
    """Keep only the highest-scoring star in each cluster of nearby detections."""
    kept = []
    for point, score in sorted(points, key=lambda item: item[1], reverse=True):
        if all(np.hypot(point[0]-px, point[1]-py) >= min_distance for px, py in kept):
            kept.append(point)
    return kept


def _find_star_peaks(gray, max_stars=16):
    """
    Finds bright star-like peaks using local maximum detection.

    BUG FIX 1: Old code used area > 20 which filtered out ALL Paint-drawn
    stars (8px radius = 81px area). Now uses adaptive threshold based on
    image size: max(300, image_pixels * 0.001) so it works for both
    Paint images and real astrophotos.

    Tries progressively lower brightness cutoffs until >= 3 stars found.
    """
    blurred   = cv2.GaussianBlur(gray, (0, 0), 1.2)
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    local_max = cv2.dilate(blurred, kernel)

    # Adaptive area limit — handles both tiny real stars and large Paint dots
    img_pixels   = gray.shape[0] * gray.shape[1]
    max_star_area = max(300, img_pixels * 0.001)

    candidates = []
    for pct in (99.8, 99.5, 99.2, 98.8, 97.0, 95.0):
        cutoff = max(float(np.percentile(blurred, pct)), 80.0)
        mask   = np.logical_and(blurred >= local_max,
                                blurred >= cutoff).astype(np.uint8) * 255

        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        current = []
        for idx in range(1, num_labels):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area > max_star_area:        # FIX: adaptive, not hardcoded 20
                continue
            cx, cy = centroids[idx]
            x, y   = int(round(cx)), int(round(cy))
            if not (0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]):
                continue
            x0, x1 = max(0, x-2), min(gray.shape[1], x+3)
            y0, y1 = max(0, y-2), min(gray.shape[0], y+3)
            patch  = gray[y0:y1, x0:x1]
            score  = float(gray[y, x]) + float(patch.mean()) * 0.35 - float(area) * 0.1
            current.append(((x, y), score))

        candidates = _prune_close_points(current, min_distance=20)
        if len(candidates) >= 3:
            break

    return candidates[:max_stars]


def detect_stars(image_path):
    """
    Detects bright star-like blobs in a night sky image.
    Works on both Paint-style test images and real astrophotos.

    Returns: (list of (cx,cy) tuples, annotated BGR image)
             or (None, None) if image cannot be loaded.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    peaks = _find_star_peaks(gray)

    star_positions = peaks
    for cx, cy in star_positions:
        cv2.circle(image, (cx, cy), 4, (0, 255, 0), -1)

    return star_positions, image


# ─────────────────────────────────────────────────────────────
# 2. FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────

MAX_STARS     = 8
FEATURE_COUNT = 38   # 28 pairwise + 8 radial + count + aspect


def _normalize_points(stars):
    pts = np.asarray(stars, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        return None
    center = pts.mean(axis=0)
    pts   -= center
    scale  = float(np.linalg.norm(pts, axis=1).max())
    if scale <= 1e-6:
        return None
    pts /= scale
    angles = np.arctan2(pts[:, 1], pts[:, 0])
    return pts[np.argsort(angles)]


def extract_features(stars):
    """
    Converts star positions into a 38-element rotation/scale-invariant
    feature vector:
      - 28 pairwise distances (sorted descending, zero-padded)
      -  8 radial distances from centroid (sorted descending, zero-padded)
      -  1 star count
      -  1 bounding-box aspect ratio
    """
    pts = _normalize_points(stars)
    if pts is None:
        return None

    pts = pts[:MAX_STARS]

    pairwise = sorted(
        [math.dist(pts[i], pts[j]) for i, j in combinations(range(len(pts)), 2)],
        reverse=True
    )
    pairwise = (pairwise + [0.0] * 28)[:28]

    radial = sorted(np.linalg.norm(pts, axis=1).tolist(), reverse=True)
    radial = (radial + [0.0] * 8)[:8]

    span_x  = float(pts[:, 0].max() - pts[:, 0].min())
    span_y  = float(pts[:, 1].max() - pts[:, 1].min())
    aspect  = span_x / (span_y + 1e-6)

    return np.asarray(pairwise + radial + [float(len(stars)), aspect],
                      dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# 3. MATCHING ENGINE  (KNN AI + Template + Rules)
# ─────────────────────────────────────────────────────────────

_MODEL_PATH     = Path(__file__).with_name("constellation_model.npz")
_MODEL          = None
_TEMPLATE_CACHE = None


def _load_model():
    global _MODEL
    if _MODEL:
        return _MODEL
    try:
        b      = np.load(_MODEL_PATH, allow_pickle=True)
        _MODEL = {
            "x":    b["x"].astype(np.float32),
            "y":    b["y"],
            "mean": b["mean"].astype(np.float32),
            "std":  b["std"].astype(np.float32),
            "k":    int(b["k"][0]) if "k" in b else 9,
        }
        print("[AI] KNN model loaded — trained on real IAU star coordinates.")
        return _MODEL
    except Exception as e:
        print(f"[AI] Model load failed: {e}")
        return None


def _load_templates():
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE
    _TEMPLATE_CACHE = {}
    for name in STAR_CATALOG:
        pts = get_star_positions_pixels(name, 500, 60)
        _TEMPLATE_CACHE[name] = {
            "count":    len(pts),
            "points":   pts,
            "features": extract_features(pts),
        }
    return _TEMPLATE_CACHE


def _shape_distance(points_a, points_b):
    """
    Compare two equal-sized star sets after normalising for translation,
    scale, and cyclic ordering around the centroid.

    This acts as a geometry verification step after the faster feature match,
    which helps prevent tiny subsets like Aries from outscoring larger
    constellations inside a noisy image.
    """
    norm_a = _normalize_points(points_a)
    norm_b = _normalize_points(points_b)
    if norm_a is None or norm_b is None or len(norm_a) != len(norm_b):
        return float("inf")

    best = float("inf")
    for candidate in (norm_b, norm_b[::-1]):
        for shift in range(len(candidate)):
            rolled = np.roll(candidate, shift, axis=0)
            dist = float(np.linalg.norm(norm_a - rolled, axis=1).mean())
            if dist < best:
                best = dist
    return best


def _ai_match(stars):
    """Weighted KNN classifier using constellation_model.npz."""
    if not (3 <= len(stars) <= 16):
        return None, 0.0
    feats  = extract_features(stars)
    if feats is None:
        return None, 0.0
    bundle = _load_model()
    if not bundle:
        return None, 0.0

    x_n    = (bundle["x"] - bundle["mean"]) / bundle["std"]
    f_n    = (feats - bundle["mean"]) / bundle["std"]
    dists  = np.linalg.norm(x_n - f_n, axis=1)
    nn     = np.argsort(dists)[:bundle["k"]]

    scores = {}
    for idx in nn:
        label           = str(bundle["y"][idx])
        scores[label]   = scores.get(label, 0.0) + 1.0 / (float(dists[idx]) + 1e-6)

    total  = sum(scores.values()) or 1.0
    pred   = max(scores, key=scores.get)
    return pred, round(scores[pred] / total * 100, 1)


def _template_match_details(stars, max_combinations=150, allowed_names=None):
    """Compare detected stars against IAU catalog templates."""
    templates  = _load_templates()
    candidates = list(stars[:12])
    if len(candidates) < 3:
        return None, 0.0, [], float("inf"), float("inf")

    items = [
        (n, t) for n, t in templates.items()
        if allowed_names is None or n in allowed_names
    ]

    best_name, best_score, best_subset = None, float("inf"), []
    all_scores = []

    for name, tmpl in items:
        if len(candidates) < tmpl["count"]:
            continue
        class_best, class_sub = float("inf"), []
        for i, subset in enumerate(combinations(candidates, tmpl["count"])):
            if i >= max_combinations:
                break
            feats = extract_features(list(subset))
            if feats is None:
                continue
            sc = float(np.linalg.norm(feats - tmpl["features"]))
            if sc < class_best:
                class_best, class_sub = sc, list(subset)
        if not math.isfinite(class_best) or not class_sub:
            continue
        shape_score = _shape_distance(class_sub, tmpl["points"])
        if not math.isfinite(shape_score):
            continue

        coverage = tmpl["count"] / max(len(candidates), 1)
        adjusted_score = shape_score / max(coverage ** 1.2, 1e-6)

        all_scores.append(adjusted_score)
        if adjusted_score < best_score:
            best_name, best_score, best_subset = name, adjusted_score, class_sub

    if not best_name:
        return None, 0.0, [], float("inf"), float("inf")

    all_scores.sort()
    runner_up = all_scores[1] if len(all_scores) > 1 else best_score + 1.0
    sep       = max(runner_up - best_score, 0.0) / max(runner_up, 1e-6)
    best_count = len(best_subset)
    coverage   = best_count / max(len(candidates), 1)
    conf       = round(
        (0.55 / (1 + best_score) + 0.25 * sep + 0.20 * coverage) * 100,
        1,
    )
    return best_name, conf, best_subset, best_score, runner_up


def _rule_match(stars):
    """Star-count rule-based fallback. Always returns something."""
    c = len(stars)
    if c == 3: return "Aries",      100.0
    if c == 4: return "Crux",       100.0
    if c == 5: return "Cassiopeia", 80.0
    if c == 7: return "Ursa Major", 100.0
    if c in (8, 9): return "Orion", 100.0
    return "Unknown constellation", 0.0


def match_constellation_details(stars):
    """
    Main matching function — tries 3 layers in order:
      1. KNN AI classifier
      2. Template matching against IAU catalog
      3. Rule-based star count fallback

    Returns: (constellation_name, confidence_percent, matched_star_subset)
    """
    if not stars:
        return "Unknown constellation", 0.0, []

    t_name, t_conf, t_sub, t_best, t_run = _template_match_details(stars)
    a_name, a_conf                        = _ai_match(stars)

    pts = np.asarray(stars, dtype=np.float32)
    span_x = float(pts[:, 0].max() - pts[:, 0].min()) if len(pts) else 0.0
    span_y = float(pts[:, 1].max() - pts[:, 1].min()) if len(pts) else 0.0
    raw_aspect = span_x / max(span_y, 1e-6)

    def _template_is_strong(name, conf, best):
        if not name or not math.isfinite(best):
            return False
        return conf >= 52.0 and best <= 0.45

    def _template_is_usable(name, conf, best):
        if not name or not math.isfinite(best):
            return False
        return conf >= 40.0 and best <= 0.50

    # Reject obviously degenerate layouts that are unlikely to be real
    # constellation patterns, such as UI text or icon columns from screenshots.
    if len(stars) >= 6 and (raw_aspect <= 0.12 or raw_aspect >= 8.0):
        return "Unknown constellation", 0.0, []

    # Both agree — strong result
    if t_name and a_name == t_name and _template_is_usable(t_name, t_conf, t_best):
        return t_name, round(max(t_conf, a_conf), 1), t_sub

    # For clean synthetic/user-drawn patterns, the AI class is often right even
    # when the full catalog geometry is only an approximate fit. Accept it when
    # the same class also passes a direct template verification.
    if a_name and a_conf >= 95.0:
        v_name, v_conf, v_sub, v_best, _ = _template_match_details(stars, allowed_names=[a_name])
        if v_name == a_name and v_conf >= 50.0 and v_best <= 1.25:
            return a_name, round(max(a_conf, v_conf), 1), v_sub if v_sub else list(stars)

    # Template confident alone
    if _template_is_strong(t_name, t_conf, t_best):
        return t_name, t_conf, t_sub

    # If template confidence is weak, let a strong AI guess recover the result,
    # but only after verifying the same class against its own template.
    if a_name and a_conf >= 92.0 and t_conf < 25.0:
        _, sc, sub, _, _ = _template_match_details(stars, allowed_names=[a_name])
        if sc >= 60.0:
            return a_name, round(max(a_conf, sc), 1), sub if sub else list(stars)

    # Template confident with clear separation
    if _template_is_usable(t_name, t_conf, t_best) and (t_run - t_best) >= 0.14:
        return t_name, t_conf, t_sub

    # Rule-based fallback only after the same class still passes a geometry check.
    r_name, r_conf = _rule_match(stars)
    if r_name != "Unknown constellation" and len(stars) <= 9:
        v_name, v_conf, v_sub, v_best, _ = _template_match_details(stars, allowed_names=[r_name])
        if v_name == r_name and v_conf >= 52.0 and v_best <= 0.38:
            return r_name, max(r_conf, v_conf), v_sub if v_sub else list(stars)

    return "Unknown constellation", 0.0, []


# ─────────────────────────────────────────────────────────────
# 4. LINE DRAWING (connect constellation stars)
# ─────────────────────────────────────────────────────────────

def _edges_aries(s):
    bx = sorted(range(len(s)), key=lambda i: s[i][0])
    return [(bx[0], bx[1]), (bx[1], bx[2])]


def _edges_crux(s):
    t = min(range(len(s)), key=lambda i: s[i][1])
    b = max(range(len(s)), key=lambda i: s[i][1])
    l = min(range(len(s)), key=lambda i: s[i][0])
    r = max(range(len(s)), key=lambda i: s[i][0])
    return [(t, b), (l, r)]


def _edges_cassiopeia(s):
    bx = sorted(range(len(s)), key=lambda i: s[i][0])
    return [(bx[k], bx[k+1]) for k in range(len(bx)-1)]


def _edges_ursa_major(s):
    bx   = sorted(range(len(s)), key=lambda i: s[i][0])
    bowl, hand = bx[:4], bx[4:]
    by   = sorted(bowl, key=lambda i: s[i][1])
    tl, tr = sorted(by[:2], key=lambda i: s[i][0])
    bl, br = sorted(by[2:], key=lambda i: s[i][0])
    return [
        (tl, tr), (tr, br), (br, bl), (bl, tl),  # bowl
        (tr, hand[0]), (hand[0], hand[1]), (hand[1], hand[2])  # handle
    ]


def _edges_orion(s):
    by          = sorted(range(len(s)), key=lambda i: s[i][1])
    ls, rs      = sorted(by[0:2], key=lambda i: s[i][0])
    lb, rb      = sorted(by[2:4], key=lambda i: s[i][0])
    bel,bem,ber = sorted(by[4:7], key=lambda i: s[i][0])
    feet        = by[7]
    return [
        (ls, rs),
        (ls, lb), (rs, rb),
        (lb, bel), (rb, ber),
        (bel, bem), (bem, ber),
        (bem, feet),
    ]


EDGE_MAP = {
    "Aries":      _edges_aries,
    "Crux":       _edges_crux,
    "Cassiopeia": _edges_cassiopeia,
    "Ursa Major": _edges_ursa_major,
    "Orion":      _edges_orion,
}

# BUG FIX 2: Removed strict EXPECTED_COUNTS check.
# Old code: if len(stars) != expected: return image  (no lines drawn!)
# This broke detection whenever star count was off by even 1.
# Now we just try to draw with whatever stars we have.


def _similarity_residual(src_points, dst_points):
    """Return the residual after the best-fit similarity transform."""
    src = np.asarray(src_points, dtype=np.float32)
    dst = np.asarray(dst_points, dtype=np.float32)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        return float("inf")

    src_centered = src - src.mean(axis=0)
    dst_centered = dst - dst.mean(axis=0)

    src_norm = float(np.linalg.norm(src_centered))
    dst_norm = float(np.linalg.norm(dst_centered))
    if src_norm <= 1e-6 or dst_norm <= 1e-6:
        return float("inf")

    src_unit = src_centered / src_norm
    dst_unit = dst_centered / dst_norm

    h = src_unit.T @ dst_unit
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T

    aligned = src_unit @ r
    return float(np.linalg.norm(aligned - dst_unit, axis=1).mean())


def _match_detected_to_catalog(matched_stars, constellation):
    """
    Align detected stars to the catalog order for the chosen constellation.

    We compare cyclic permutations of the normalised detected shape against the
    normalised catalog shape and keep the best ordering. This gives us a stable
    mapping back to the catalog edge list, which is much more reliable than
    reconstructing the pattern from generic geometry heuristics.
    """
    if constellation not in STAR_CATALOG or not matched_stars:
        return None

    catalog_pts = get_star_positions_pixels(constellation, 500, 60)
    if len(catalog_pts) != len(matched_stars):
        return None

    det = np.asarray(matched_stars, dtype=np.float32)
    cat = np.asarray(catalog_pts, dtype=np.float32)
    best_map = None
    best_score = float("inf")
    count = len(matched_stars)

    for perm in permutations(range(count)):
        ordered_detected = det[list(perm)]
        score = _similarity_residual(cat, ordered_detected)
        if score < best_score:
            best_score = score
            best_map = {
                catalog_idx: tuple(map(int, matched_stars[det_idx]))
                for catalog_idx, det_idx in enumerate(perm)
            }

    return best_map


def connect_stars(image, stars, constellation):
    """
    Draw constellation lines using constellation-specific geometry rules.

    The catalog remapping approach is useful for validation, but for
    user-drawn simplified shapes it can distort the edge assignment.
    The geometry-based edge builders produce much cleaner visual results
    for the supported constellations.
    """
    if not stars or len(stars) < 2:
        return image

    fn = EDGE_MAP.get(constellation)
    if not fn:
        return image

    try:
        edges = fn(stars)
    except Exception:
        return image

    for i, j in edges:
        if 0 <= i < len(stars) and 0 <= j < len(stars):
            cv2.line(image, stars[i], stars[j], (255, 180, 0), 2)

    return image


# ─────────────────────────────────────────────────────────────
# 5. STAR NAME LABELS
# ─────────────────────────────────────────────────────────────

def label_stars(image, matched_stars, constellation):
    """
    Draw catalog star names next to detected stars.
    Uses nearest-neighbour matching between normalised detected positions
    and normalised catalog positions to assign names.
    """
    from constellation_data import STAR_CATALOG

    if not matched_stars or constellation not in STAR_CATALOG:
        return image

    img = image.copy()
    stars_data = STAR_CATALOG[constellation]["stars"]
    assignments = _match_detected_to_catalog(matched_stars, constellation)

    if assignments is None:
        return img

    for ci, point in assignments.items():
        if ci < len(stars_data):
            x, y = point
            name = stars_data[ci]["name"]
            # Dark outline for readability on any background
            cv2.putText(img, name, (x + 9, y - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, name, (x + 9, y - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 200), 1, cv2.LINE_AA)
    return img
