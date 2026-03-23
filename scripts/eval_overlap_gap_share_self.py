#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute inter-class overlap, intra-class overlap, gap rate,
and shared-edge consistency from per-category COCO-style
polygon predictions.

Rules & implementation:
- Rasterize polygons onto an H×W grid using closed-set filling (cv2.fillPoly).
- Inter-class overlap: union polygons per class, then count pixels covered by >1 class.
- Intra-class overlap: instance-level overlap within each class (count ≥2).
- Shared-edge exclusion: render all exterior rings and holes as polylines
  into polyline_count; overlap pixels with polyline_count ≥2 are treated as
  shared boundaries and excluded, while polyline_count ==1 is kept as true overlap.
- Gap: pixels with cover_count == 0 (boundaries are not excluded).
- Shared-edge consistency: ratio of boundary pixels with polyline_count ≥2
  to all boundary pixels (polyline_count ≥1), computed per image and averaged
  over images with valid boundaries only.

Inputs:
  --pred_dir
    Directory containing per-category prediction JSON files
    (e.g., building.json, water.json, ...).

  JSON format:
    list of {
      "image_id": ...,
      "segmentation": [ exterior, hole1, hole2, ... ]
    }
    or standard COCO-style ground-truth format
    (dict with an "annotations" field).

Outputs:
- Per-image metrics:
    gap_rate,
    inter_overlap_rate,
    intra_overlap_rate,
    shared_edge_consistency,
    and per-class intra-class overlap rates.
- Dataset-level averages over all evaluated images.
- Optional: debug PNG visualizations for each image
  (enabled via --draw_debug and saved to out_dir).

Notes:
- Integer pixel coordinates are used (no 0.5 offset).
- lineType=cv2.LINE_8 and thickness=1 are enforced to avoid
  anti-aliasing artifacts and gray-valued boundary pixels.
"""

import os
import json
import cv2
import numpy as np
from collections import defaultdict


NAME2ID = {
    "artificial_structure": 0,
    "building": 1,
    "road_bridge": 2,
    "unvegetated": 3,
    "vegetation": 4,
    "water": 5,
    "unknown": 6,
}


def infer_cat_id_from_stem(stem: str):
    s = stem.lower()
    matches = [(len(k), NAME2ID[k]) for k in NAME2ID if k in s]
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][1]


def flatten_to_xylist(flat):
    """[x1,y1,x2,y2,...] -> Nx2 int32"""
    a = np.asarray(flat, dtype=np.float32).reshape(-1, 2)
    a = np.rint(a).astype(np.int32)
    return a


def clip_poly(poly, W, H):
    """clip polygon coordinates to [0,W-1]/[0,H-1]"""
    poly[:, 0] = np.clip(poly[:, 0], 0, W - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)
    return poly


def load_per_image_instances(pred_dir):
    """
    Load per-category JSON files and merge them as:
      per_image[image_id] -> list of dict:
        { "class_key": <file stem>, "seg": [ext, hole1, ...] }
    Also keep per-image, per-class instance lists for intra-class counting.
    """
    per_image_items = defaultdict(list)
    per_image_per_class = defaultdict(lambda: defaultdict(list))

    for fname in sorted(os.listdir(pred_dir)):
        if not fname.lower().endswith(".json"):
            continue
        stem = os.path.splitext(fname)[0]
        # Use the file stem as a stable class key.
        class_key = stem

        path = os.path.join(pred_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "annotations" in data:
                data = data["annotations"]
            if not isinstance(data, list):
                raise ValueError(f"{fname} must be a list or a COCO dict with 'annotations'.")

        for ann in data:
            image_id = int(ann["image_id"])
            seg = ann.get("segmentation", [])
            if not isinstance(seg, list) or len(seg) == 0:
                continue
            per_image_items[image_id].append({"class_key": class_key, "seg": seg})
            per_image_per_class[image_id][class_key].append(seg)

    return per_image_items, per_image_per_class


def _segments_from_ring(pts):
    """Return closed ring edges as [(p0, p1), ..., (p_{n-1}, p0)]."""
    n = pts.shape[0]
    if n < 2:
        return []
    segs = []
    for i in range(n):
        a = (int(pts[i,0]), int(pts[i,1]))
        b = (int(pts[(i+1)%n,0]), int(pts[(i+1)%n,1]))
        if a != b:
            segs.append((a,b))
    return segs


def _bbox_overlap(a,b,c,d):
    return (min(a[0],b[0]) <= max(c[0],d[0]) and
            min(c[0],d[0]) <= max(a[0],b[0]) and
            min(a[1],b[1]) <= max(c[1],d[1]) and
            min(c[1],d[1]) <= max(a[1],b[1]))


def _orient(a,b,c):
    """Return the signed oriented area of three points."""
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def _on_segment(a,b,c):
    """Return whether c lies on segment ab, assuming collinearity."""
    return (min(a[0],b[0]) <= c[0] <= max(a[0],b[0]) and
            min(a[1],b[1]) <= c[1] <= max(a[1],b[1]))


def _segments_intersect_nontrivial(a, b, c, d, eps=1e-6, endpoint_buffer=0.5):
    """
    Count only proper intersections:
      - exclude endpoint touching, including near-endpoint raster artifacts
      - exclude collinear overlap
    Segments are slightly shrunk inward to suppress endpoint jitter.
    """
    # Quick bounding-box rejection.
    if not _bbox_overlap(a, b, c, d):
        return False

    # Shrink endpoints slightly to avoid near-endpoint false positives.
    a2, b2 = _shrink_segment_inward(a, b, endpoint_buffer)
    c2, d2 = _shrink_segment_inward(c, d, endpoint_buffer)

    # Reject again after shrinking.
    if not _bbox_overlap(a2, b2, c2, d2):
        return False

    # Oriented-area tests with tolerance.
    o1 = _orient(a2, b2, c2)
    o2 = _orient(a2, b2, d2)
    o3 = _orient(c2, d2, a2)
    o4 = _orient(c2, d2, b2)

    def sgn(x):
        if x > eps: return 1
        if x < -eps: return -1
        return 0

    s1, s2, s3, s4 = sgn(o1), sgn(o2), sgn(o3), sgn(o4)

    # Proper crossing requires opposite signs in both tests.
    return (s1 * s2 < 0) and (s3 * s4 < 0)


def count_self_intersections_one_instance(segmentation_flat_list, W, H):
    """
    Count self-intersections within one COCO-style instance segmentation.

    Adjacent edges from the same ring are excluded so shared endpoints are
    not counted as self-intersections.
    """
    # Collect all ring edges and record their source ring / edge index.
    all_segs = []
    tags = []
    for r_id, flat in enumerate(segmentation_flat_list):
        if not isinstance(flat, list) or len(flat) < 4:
            continue
        pts = flatten_to_xylist(flat)
        if pts.shape[0] < 2:
            continue
        pts = clip_poly(pts, W, H)
        segs = _segments_from_ring(pts)
        for e_idx, s in enumerate(segs):
            all_segs.append(s)
            tags.append((r_id, e_idx))
    n = len(all_segs)
    if n <= 1:
        return 0

    # Count pairwise proper intersections.
    cnt = 0
    ring_edge_counts = {}
    for r_id, flat in enumerate(segmentation_flat_list):
        if isinstance(flat, list) and len(flat) >= 4:
            pts = flatten_to_xylist(flat)
            if pts.shape[0] >= 2:
                ring_edge_counts[r_id] = pts.shape[0]

    for i in range(n):
        a,b = all_segs[i]
        r1, e1 = tags[i]
        for j in range(i+1, n):
            c,d = all_segs[j]
            r2, e2 = tags[j]

            # Skip the same edge and adjacent edges from the same ring.
            if r1 == r2 and r1 in ring_edge_counts:
                m = ring_edge_counts[r1]
                if (e1 == e2) or ((e1+1)%m == e2) or ((e2+1)%m == e1):
                    continue

            if _segments_intersect_nontrivial(a, b, c, d, eps=1e-6, endpoint_buffer=0.5):
                cnt += 1
    return cnt


def _shrink_segment_inward(p, q, delta=0.5):
    """Shrink segment endpoints inward by delta pixels."""
    px, py = float(p[0]), float(p[1])
    qx, qy = float(q[0]), float(q[1])
    vx, vy = qx - px, qy - py
    L = (vx * vx + vy * vy) ** 0.5
    if L <= 1e-12:
        return (px, py), (qx, qy)
    sx, sy = (vx / L) * delta, (vy / L) * delta
    return (px + sx, py + sy), (qx - sx, qy - sy)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute overlap & gap rates from per-category polygon predictions (COCO-style)."
    )
    parser.add_argument('--pred_dir', required=True,
                        help='Directory of per-category prediction JSONs (e.g., building.json, water.json, ...)')
    parser.add_argument('--width',  type=int, required=True)
    parser.add_argument('--height', type=int, required=True)

    # Optional: evaluate a single image only
    parser.add_argument('--image_id', type=str, default=None,
                        help='If set, only evaluate this image_id')

    # Optional visualization
    parser.add_argument('--out_dir', type=str, default=None, help='If set, save per-image debug PNGs here')
    parser.add_argument('--draw_debug', action='store_true', default=False, help='Render debug overlays')

    # Boundary handling
    parser.add_argument('--line_thickness', type=int, default=1, help='Polyline thickness for boundary counting')
    parser.add_argument('--dilate_boundary', type=int, default=0, help='Optional dilation iterations on polyline_count>0 before counting >=2')

    args = parser.parse_args()
    H, W = args.height, args.width

    # ------------------------------------------------------------------
    # 1. Load predictions
    # ------------------------------------------------------------------
    per_image_items, per_image_per_class = load_per_image_instances(args.pred_dir)

    # Select images to evaluate
    if args.image_id is not None:
        key_try = args.image_id
        if key_try.isdigit():
            key_try = int(key_try)
        if key_try not in per_image_items:
            key_alt = str(args.image_id)
            if key_alt not in per_image_items:
                print(f"[WARN] image_id={args.image_id} not found. Available: {sorted(per_image_items.keys())[:10]} ...")
                return
            image_ids = [key_alt]
        else:  
            image_ids = [key_try]
    else:
        image_ids = sorted(per_image_items.keys())

    if args.out_dir and args.draw_debug:
        os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Global accumulators
    # ------------------------------------------------------------------
    n_images = 0
    sum_gap = 0.0
    sum_inter = 0.0
    sum_intra = 0.0
    sum_shared_consistency = 0.0
    shared_consistency_bad = []
    cnt_shared_considered = 0
    
    all_class_keys = set()
    per_class_sum = defaultdict(float)

    sum_self_intersections = 0
    self_intersect_image_ids = []

    inter_overlap_image_id = []
    intra_overlap_image_id = []

    # ------------------------------------------------------------------
    # 2. Process each image
    # ------------------------------------------------------------------
    for image_id in image_ids:
        per_class_segs = per_image_per_class[image_id] 
        all_class_keys.update(per_class_segs.keys())

        class_union = {}
        class_inst_count = {}

        cover_count = np.zeros((H, W), dtype=np.uint16)
        polyline_count = np.zeros((H, W), dtype=np.uint16)

        # --------------------------------------------------------------
        # Build per-class union masks and instance-overlap counters.
        for class_key, seg_list in per_class_segs.items():
            union_mask = np.zeros((H, W), dtype=np.uint8)
            inst_count = np.zeros((H, W), dtype=np.uint16)

            for seg in seg_list:
                if not isinstance(seg, list) or len(seg) == 0:
                    continue

                inst_mask = np.zeros((H, W), dtype=np.uint8)

                # Exterior ring.
                ext = flatten_to_xylist(seg[0])
                if ext.shape[0] < 3:
                    continue
                ext = clip_poly(ext, W, H)
                cv2.fillPoly(inst_mask, [ext.reshape(-1, 1, 2)], 1, lineType=cv2.LINE_8)

                # Hole rings.
                for h in seg[1:]:
                    if isinstance(h, list) and len(h) >= 6:
                        hole = flatten_to_xylist(h)
                        if hole.shape[0] >= 3:
                            hole = clip_poly(hole, W, H)
                            cv2.fillPoly(inst_mask, [hole.reshape(-1, 1, 2)], 0, lineType=cv2.LINE_8)

                union_mask |= inst_mask
                inst_count += inst_mask.astype(np.uint16)

                # Accumulate boundary pixels for shared-edge detection.
                tmp = np.zeros((H, W), dtype=np.uint8)
                cv2.polylines(tmp, [ext.reshape(-1, 1, 2)], True,
                              color=1, thickness=args.line_thickness, lineType=cv2.LINE_8)
                polyline_count += tmp.astype(np.uint16)

                for h in seg[1:]:
                    if isinstance(h, list) and len(h) >= 4:
                        hole = flatten_to_xylist(h)
                        if hole.shape[0] >= 2:
                            hole = clip_poly(hole, W, H)
                            tmp2 = np.zeros((H, W), dtype=np.uint8)
                            cv2.polylines(tmp2, [hole.reshape(-1, 1, 2)], True,
                                          color=1, thickness=args.line_thickness, lineType=cv2.LINE_8)
                            polyline_count += tmp2.astype(np.uint16)

            class_union[class_key] = union_mask.astype(bool)
            class_inst_count[class_key] = inst_count
            cover_count += union_mask.astype(np.uint16)

        # --------------------------------------------------------------
        # Raw gap and overlap masks.
        if len(class_union) > 0:
            stacked = np.stack([m.astype(np.uint8) for m in class_union.values()], axis=0)
            class_cover = stacked.sum(axis=0)
            inter_raw = (class_cover > 1)
            covered = (class_cover >= 1)
        else:
            inter_raw = np.zeros((H, W), dtype=bool)
            covered  = np.zeros((H, W), dtype=bool)

        intra_raw_by_class = {k: (cnt >= 2) for k, cnt in class_inst_count.items()}

        intra_raw_total = np.zeros((H, W), dtype=bool)
        for m in intra_raw_by_class.values():
            intra_raw_total |= m

        gap_mask = (~covered)

        # --------------------------------------------------------------
        # Remove shared boundaries from overlap statistics.
        if args.dilate_boundary > 0:
            kernel = np.ones((3, 3), np.uint8)
            pl_bin = (polyline_count > 0).astype(np.uint8)
            for _ in range(args.dilate_boundary):
                pl_bin = cv2.dilate(pl_bin, kernel, iterations=1)
            boundary_shared = (polyline_count >= 2)
        else:
            boundary_shared = (polyline_count >= 2)

        inter_mask = inter_raw & (~boundary_shared)
        intra_mask_total = intra_raw_total & (~boundary_shared)
        intra_by_class = {k: (m & (~boundary_shared)) for k, m in intra_raw_by_class.items()}

        # --------------------------------------------------------------
        # Shared-edge consistency.
        # Denominator: all interior boundary pixels.
        boundary_all = (polyline_count >= 1)
        interior = np.ones((H, W), dtype=bool)
        interior[0, :] = interior[-1, :] = False
        interior[:, 0] = interior[:, -1] = False
        denom_mask = boundary_all & interior
        denom_cnt = int(denom_mask.sum())

        # Numerator: interior boundary pixels shared by at least two polylines.
        numer_mask = (polyline_count >= 2) & interior
        numer_cnt = int(numer_mask.sum())

        shared_edge_consistency = (numer_cnt / denom_cnt) if denom_cnt > 0 else 0.0

        eps = 1e-9
        if denom_cnt > 0:
            if shared_edge_consistency < 1.0 - eps:
                shared_consistency_bad.append((image_id, shared_edge_consistency))
            # Average only over images with valid interior boundaries.
            cnt_shared_considered += 1

        # --------------------------------------------------------------
        # Self-intersection statistics.
        self_intersections_img = 0
        for _, seg_list in per_class_segs.items():
            for seg in seg_list:  # seg = [ext, hole1, ...]
                if not isinstance(seg, list) or len(seg) == 0:
                    continue
                self_intersections_img += count_self_intersections_one_instance(seg, W, H)

        sum_self_intersections += self_intersections_img
        if self_intersections_img > 0:
            self_intersect_image_ids.append((image_id, self_intersections_img))

        # --------------------------------------------------------------
        # Per-image rates.
        total_pix = float(H * W)
        gap_rate = gap_mask.sum() / total_pix
        inter_rate = inter_mask.sum() / total_pix
        intra_rate = intra_mask_total.sum() / total_pix
        intra_by_class_rate = {k: v.sum() / total_pix for k, v in intra_by_class.items()}

        print(f"== image {image_id} ==")
        print(f"  gap_rate                : {gap_rate:.6f}")
        print(f"  inter_overlap_rate      : {inter_rate:.6f}")
        print(f"  intra_overlap_rate (all): {intra_rate:.6f}")
        print(f"  shared_edge_consistency : {shared_edge_consistency:.6f}")  
        print(f"  self_intersections      : {self_intersections_img}")

        if len(intra_by_class_rate) > 0:
            print("  intra_overlap_rate by class:")
            for k in sorted(intra_by_class_rate.keys()):
                print(f"    - {k}: {intra_by_class_rate[k]:.6f}")

        if inter_rate > 0:
            inter_overlap_image_id.append(image_id)
        if intra_rate > 0:
            intra_overlap_image_id.append(image_id)

        # Dataset accumulators.
        n_images += 1
        sum_gap   += gap_rate
        sum_inter += inter_rate
        sum_intra += intra_rate
        sum_shared_consistency += shared_edge_consistency  
        for k in intra_by_class_rate:
            per_class_sum[k] += intra_by_class_rate[k]

        # Optional debug visualization.
        if args.out_dir and args.draw_debug:
            dbg = np.zeros((H, W, 3), dtype=np.uint8); dbg[:] = 255
            rng = np.random.RandomState(42)
            palette = {k: tuple(int(c) for c in rng.randint(160, 230, size=3)) for k in class_union.keys()}
            for k in class_union.keys():
                dbg[class_union[k]] = palette[k]
            dbg[gap_mask] = (255, 0, 0)
            dbg[intra_mask_total] = (255, 0, 255)
            dbg[inter_mask] = (0, 0, 255)
            for class_key, seg_list in per_class_segs.items():
                for seg in seg_list:
                    if not isinstance(seg, list) or len(seg) == 0:
                        continue
                    ext = flatten_to_xylist(seg[0])
                    if ext.shape[0] >= 2:
                        ext = clip_poly(ext, W, H)
                        cv2.polylines(dbg, [ext.reshape(-1, 1, 2)], True,
                                      color=(0, 0, 0), thickness=1, lineType=cv2.LINE_8)
                    for h in seg[1:]:
                        if isinstance(h, list) and len(h) >= 4:
                            hole = flatten_to_xylist(h)
                            if hole.shape[0] >= 2:
                                hole = clip_poly(hole, W, H)
                                cv2.polylines(dbg, [hole.reshape(-1, 1, 2)], True,
                                              color=(0, 0, 0), thickness=1, lineType=cv2.LINE_8)
            os.makedirs(args.out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(args.out_dir, f"{image_id}_debug.png"), dbg)

    # ------------------------------------------------------------------
    # 8. Dataset-level summary
    # ------------------------------------------------------------------
    if n_images > 0:
        print("\n== Averages over images ==")
        print(f"  gap_rate (avg)           : {sum_gap / n_images:.4f}")
        print(f"  inter_overlap_rate (avg) : {sum_inter / n_images:.4f}")
        print(f"  intra_overlap_rate (avg) : {sum_intra / n_images:.4f}")
        
        if cnt_shared_considered > 0:
            print(f"  shared_edge_consistency (avg)  : {sum_shared_consistency / cnt_shared_considered:.6f}")
        else:
            print(f"  shared_edge_consistency (avg)  : N/A (no images with internal polylines)")

        # Kept as reference during debugging.
        # if shared_consistency_bad:
        #     print("\n== Images with shared_edge_consistency < 1 ==")
        #     # Sort from low to high for easier inspection.
        #     shared_consistency_bad.sort(key=lambda x: x[1])
        #     for img_id, val in shared_consistency_bad:
        #         print(f"  image_id={img_id}  shared_edge_consistency={val:.6f}")
        # else:
        #     print("\nAll evaluated images have shared_edge_consistency == 1 (within tolerance).")

        print(f"  self_intersections (sum)       : {sum_self_intersections}") 

        # Kept as reference during debugging.
        # if self_intersect_image_ids:
        #     print("\n== Images with self_intersections > 0 ==")
        #     self_intersect_image_ids.sort(key=lambda x: x[1], reverse=True)
        #     for img_id, cnt in self_intersect_image_ids:
        #         print(f"  image_id={img_id}  self_intersections={cnt}")
        # else:
        #     print("\nNo self-intersections found in any evaluated images.")

        if len(all_class_keys) > 0:
            print("  intra_overlap_rate by class (avg over images):")
            for k in sorted(all_class_keys):
                avg = per_class_sum[k] / n_images if n_images > 0 else 0.0
                print(f"    - {k}: {avg:.4f}")
    else:
        print("[WARN] No images found.")

    print("inter_overlap_image_id: ", inter_overlap_image_id)
    print("intra_overlap_image_id: ", intra_overlap_image_id)


if __name__ == "__main__":
    main()

    """
    Usage (example):

    python eval_overlap_gap_share_self.py \
        --pred_dir <path_to_prediction_jsons> \
        --width 512 \
        --height 512
    """

    """
    # Evaluate image_id=0 only
    python eval_overlap_gap_share_self.py \
    --pred_dir /path/to/categories \
    --width 512 --height 512 \
    --image_id 0    
    
    python eval_overlap_gap_share_self.py \
    --pred_dir ./ACPV-Net/outputs/legacy/deventer_vmamba-small_512_vh_m_ldm_v1.5_simp_gt_b8/ddim/poly_gb_v5_dp-2.5_angle_tol-10_corner_eps-2_replace_thresh-5_nms-3_th-0.5_topk-1k/categories \
    --width 512 --height 512
    == Averages over images ==
    gap_rate (avg)           : 0.000000
    inter_overlap_rate (avg) : 0.000048
    intra_overlap_rate (avg) : 0.000038
    intra_overlap_rate by class (avg over images):
        - artificial_structure: 0.000000
        - building: 0.000000
        - road_bridge: 0.000000
        - unknown: 0.000000
        - unvegetated: 0.000000
        - vegetation: 0.000038
        - water: 0.000000
        inter_overlap_image_id:  [111, 507, 524, 558, 1054, 1081, 1318, 1319, 1792, 1864, 1932, 1944]
        intra_overlap_image_id:  [111, 1932]

    python eval_overlap_gap_share_self.py \
    --pred_dir ./ACPV-Net/outputs/deventer512_vmamba-s_m_vh-ldm_kl4_b8/ddim_original/poly_pslg_dp-2.5_angle_tol-10_corner_eps-2_replace_thresh-5_nms-3_th-0.5_topk-1k/categories \
    --width 512 --height 512

    python scripts/eval_overlap_gap_share_self.py \
    --pred_dir ./outputs/deventer512_vmamba-s_m_vh-ldm_kl4_b8/ddim/poly_pslg_dp-2.5_angle_tol-10_corner_eps-2_replace_thresh-5_nms-3_th-0.5_topk-1k/categories \
    --width 512 --height 512
    
    # GCP: 
    python calculate_overlap_gap_share_self.py \
    --pred_dir ./PS_cluster_baseline_results/GCP/work_dirs/summary_deventer_512 \
    --width 512 --height 512
    
    # HiSup:
    python calculate_overlap_gap_share_self.py \
    --pred_dir ../work3/HiSup-main/outputs/deventer_512_categories \
    --width 512 --height 512

    # DeepSnake:
    python calculate_overlap_gap_share_self.py \
    --pred_dir ./PS_cluster_baseline_results/dance-master/output/deventer_512/dsnake_R50_bs8_ep100/categories_preds_poly \
    --width 512 --height 512

    # TopDiG:
    python calculate_overlap_gap_share_self.py \
    --pred_dir ./PS_cluster_baseline_results/TopDiG/records/Deventer_512_summary \
    --width 512 --height 512

    # FFL:
    python calculate_overlap_gap_share_self.py \
    --pred_dir "../work3/FFL-main/data/mapping_challenge_dataset(Deventer_512)/eval_runs_summary" \
    --width 512 --height 512
    """
