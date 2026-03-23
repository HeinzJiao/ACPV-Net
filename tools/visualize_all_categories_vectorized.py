#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render overview visualizations from per-category polygon JSON files.

This script reads category-wise polygon JSON files, groups annotations by
image, and renders enlarged overview PNGs with optional overlap, gap, and
vertex overlays. Each JSON file may store either prediction entries directly
or COCO-style annotation dictionaries with an `annotations` field.

Example:
    python tools/visualize_all_categories_vectorized.py \
        --pred_dir /path/to/category_jsons \
        --out_dir /path/to/overview_pngs \
        --draw_overlap \
        --draw_gap \
        --draw_vertices
"""

import os
import json
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from polygonize_utils import flatten_to_xylist

# File name stem -> category ID.
NAME2ID = {
    "building": 0,
    "road_bridge": 1,   # Rendered as road.
    "unvegetated": 2,
    "vegetation": 3,
    "water": 4,
}

def infer_cat_id_from_stem(stem: str):
    s = stem.lower()
    matches = [(len(k), NAME2ID[k]) for k in NAME2ID if k in s]
    if not matches:
        return None
    matches.sort(reverse=True)  # Prefer the longest matching class name.
    return matches[0][1]

def main():
    # Rendering strategy:
    # 1. Fill each category without drawing boundaries and track pixel coverage.
    # 2. Mark overlap pixels (covered by multiple categories) and gap pixels
    #    (covered by none).
    # 3. Draw all polygon boundaries and optional vertices on top so shared
    #    edges are not incorrectly shown as overlap.

    import argparse
    parser = argparse.ArgumentParser(
        description="Render per-image overview PNGs from per-category polygon JSON files."
    )
    parser.add_argument('--pred_dir', required=True,
                        help='Directory containing per-category polygon JSON files.')
    parser.add_argument('--out_dir', required=True,
                        help='Directory to save rendered overview PNG files.')
    parser.add_argument('--height', type=int, default=512,
                        help='Canvas height before upscaling.')
    parser.add_argument('--width', type=int, default=512,
                        help='Canvas width before upscaling.')
    parser.add_argument('--scale', type=int, default=2,
                        help='Visualization upscaling factor.')

    # Blending strength for overlap and gap overlays.
    parser.add_argument('--alpha_overlap', type=float, default=1.0,
                        help='Opacity of the overlap overlay.')
    parser.add_argument('--alpha_gap', type=float, default=1.0,
                        help='Opacity of the gap overlay.')

    # Optional overlays.
    parser.add_argument('--draw_overlap', action='store_true', default=False,
                        help='Highlight overlap regions.')
    parser.add_argument('--draw_gap', action='store_true', default=False,
                        help='Highlight gap regions.')
    parser.add_argument('--draw_vertices', action='store_true', default=False,
                        help='Draw polygon vertices on top of the overview.')

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    H, W = args.height, args.width
    SCALE = 2
    H2, W2 = H * SCALE, W * SCALE

    # Category fill colors.
    CATEGORY_COLORS_BGR = {
        4: (237, 174, 120),  # Water       RGB(120,174,237)
        3: (181, 226, 173),  # Vegetation  RGB(173,226,181)
        2: (234, 245, 251),  # Unvegetated RGB(224,214,206)
        1: (193, 167, 141),  # Road        RGB(141,167,193)
        0: (199, 224, 242),  # Building    RGB(242,224,199)
    }

    # Dot pattern for unknown and artificial-structure regions.
    DOT_GRAY_BGR = (200, 200, 200)
    DOT_STEP = 16

    # Boundary line color.
    BOUNDARY_BGR = (0, 0, 0)

    # Solid overlay colors for overlap and gap.
    OVERLAP_FILL_BGR = (0, 0, 255)
    GAP_FILL_BGR = (0, 0, 0)

    # Vertex color and radius on the upscaled canvas.
    VERT_BGR = (0, 128, 255)  # Orange.
    VERT_RADIUS = 4

    def infer_cat_id_from_stem(stem: str):
        s = stem.lower()
        matches = [(len(k), NAME2ID[k]) for k in NAME2ID if k in s]
        if not matches:
            return None
        matches.sort(reverse=True)
        return matches[0][1]

    # Group predictions by image ID.
    per_image = defaultdict(list)
    for fname in sorted(os.listdir(args.pred_dir)):
        if not fname.endswith('.json'):
            continue
        stem = os.path.splitext(fname)[0]
        cat_id = infer_cat_id_from_stem(stem)
        if cat_id is None:
            continue
        with open(os.path.join(args.pred_dir, fname), 'r') as f:
            anns = json.load(f)
            if type(anns) != list:  # GT-style COCO file instead of a plain prediction list.
                anns = anns["annotations"]
        for ann in anns:
            image_id = int(ann["image_id"])
            seg = ann["segmentation"]  # [[ext_flat], [hole1_flat], ...]
            per_image[image_id].append((cat_id, seg))

    for image_id, items in tqdm(sorted(per_image.items()), desc="Rendering overviews"):
        # Render directly on the upscaled canvas.
        canvas_up = np.ones((H2, W2, 3), dtype=np.uint8) * 255
        cover_count_up = np.zeros((H2, W2), dtype=np.uint8)

        all_vertices_up = []   # Vertices in upscaled coordinates.
        all_polylines_up = []  # Exterior and hole boundaries in upscaled coordinates.

        # Fill all categories without boundaries while updating coverage counts.
        for cat_id, seg in items:
            if not isinstance(seg, list) or len(seg) == 0:
                continue

            ext_xy = flatten_to_xylist(seg[0])
            holes_xy = [flatten_to_xylist(r) for r in seg[1:]]
            if len(ext_xy) < 3:
                continue

            # Upscale coordinates to the visualization canvas.
            ext_arr_up = (np.array(ext_xy, dtype=np.float32) * SCALE).astype(np.int32)
            inst_mask_up = np.zeros((H2, W2), dtype=np.uint8)

            # Fill the exterior ring.
            ext_poly = ext_arr_up.reshape(-1, 1, 2)
            # Clip polygon coordinates to the canvas bounds.
            ext_poly[:, 0, 0] = np.clip(ext_poly[:, 0, 0], 0, W2 - 1)
            ext_poly[:, 0, 1] = np.clip(ext_poly[:, 0, 1], 0, H2 - 1)
            cv2.fillPoly(inst_mask_up, [ext_poly], 1)

            # Carve holes back to zero.
            for h in holes_xy:
                if len(h) >= 3:
                    h_arr_up = (np.array(h, dtype=np.float32) * SCALE).astype(np.int32).reshape(-1, 1, 2)
                    h_arr_up[:, 0, 0] = np.clip(h_arr_up[:, 0, 0], 0, W2 - 1)
                    h_arr_up[:, 0, 1] = np.clip(h_arr_up[:, 0, 1], 0, H2 - 1)
                    cv2.fillPoly(inst_mask_up, [h_arr_up], 0)

            # Fill the category color without drawing boundaries yet.
            color = CATEGORY_COLORS_BGR.get(cat_id, (255, 255, 255))
            canvas_up[inst_mask_up == 1] = color

            # Add a gray dot pattern for unknown or artificial-structure regions.
            if cat_id in (6, 0):
                # Use a staggered dot grid to avoid visible striping.
                for y in range(0, H2, DOT_STEP):
                    x_offset = (y // DOT_STEP) % 2 * (DOT_STEP // 2)
                    for x in range(x_offset, W2, DOT_STEP):
                        if inst_mask_up[y, x] == 1:
                            canvas_up[y, x] = DOT_GRAY_BGR

            # Update the coverage count for overlap/gap detection.
            cover_count_up += inst_mask_up

            # Store exterior and hole boundaries for the final outline pass.
            all_polylines_up.append(ext_arr_up)
            for h in holes_xy:
                if len(h) >= 2:
                    all_polylines_up.append((np.array(h, dtype=np.float32) * SCALE).astype(np.int32))

            # Store vertices for the optional vertex overlay.
            all_vertices_up.extend(ext_arr_up.tolist())
            for h in holes_xy:
                if len(h) > 0:
                    all_vertices_up.extend(((np.array(h, dtype=np.float32) * SCALE).astype(np.int32)).tolist())

        # Apply overlap and gap overlays on the upscaled canvas.
        if args.draw_overlap:
            overlap_mask = (cover_count_up > 1)
            if overlap_mask.any():
                canvas_up[overlap_mask] = (
                    (1.0 - args.alpha_overlap) * canvas_up[overlap_mask].astype(np.float32)
                    + args.alpha_overlap * np.array(OVERLAP_FILL_BGR, dtype=np.float32)
                ).astype(np.uint8)

        if args.draw_gap:
            gap_mask = (cover_count_up == 0)
            if gap_mask.any():
                canvas_up[gap_mask] = (
                    (1.0 - args.alpha_gap) * canvas_up[gap_mask].astype(np.float32)
                    + args.alpha_gap * np.array(GAP_FILL_BGR, dtype=np.float32)
                ).astype(np.uint8)

        # Draw all boundaries after the fills and overlays.
        for poly in all_polylines_up:
            if isinstance(poly, np.ndarray) and poly.shape[0] >= 2:
                poly2 = poly.reshape(-1, 1, 2)
                cv2.polylines(canvas_up, [poly2], isClosed=True,
                              color=BOUNDARY_BGR, thickness=2, lineType=cv2.LINE_AA)

        # Draw vertices on top if requested.
        if args.draw_vertices and len(all_vertices_up) > 0:
            for (x, y) in all_vertices_up:
                cx, cy = int(x), int(y)
                if 0 <= cx < W2 and 0 <= cy < H2:
                    cv2.circle(canvas_up, (cx, cy), VERT_RADIUS, VERT_BGR,
                               thickness=-1, lineType=cv2.LINE_AA)

        cv2.imwrite(os.path.join(args.out_dir, f"{image_id}.png"), canvas_up)



if __name__ == "__main__":
    main()

    """
    # ----- GT (previous version) -----
    python visualize_all_categories_vectorized_fancy.py \
    --pred_dir ./ACPV-Net/data/deventer_512/poly_gt_global_boundary/test/d-2_angle_tol_deg-10-5-2_corner_eps-2_min_sep-3_len_px-10-6_v5/categories_coco_ann \
    --out_dir ./ACPV-Net/data/deventer_512/poly_gt_global_boundary/test/d-2_angle_tol_deg-10-5-2_corner_eps-2_min_sep-3_len_px-10-6_v5/overview_fancy_vertex_orange_4px \
    --draw_overlap \
    --draw_gap \
    --draw_vertices 

    # ----- GT -----
    python visualize_all_categories_vectorized_fancy.py \
    --pred_dir ./ACPV-Net/data/deventer_512/test/annotations \
    --out_dir ./ACPV-Net/data/deventer_512/test/overview_fancy \
    --draw_overlap \
    --draw_gap \
    --draw_vertices 
    
    # ----- Ours -----
    # Final
    python tools/visualize_all_categories_vectorized.py \
    --pred_dir ./outputs/deventer512_vmamba-s_m_vh-ldm_kl4_b8/ddim/poly_pslg_dp-2.5_angle_tol-10_corner_eps-2_replace_thresh-5_nms-3_th-0.5_topk-1k/categories \
    --out_dir ./outputs/deventer512_vmamba-s_m_vh-ldm_kl4_b8/ddim/poly_pslg_dp-2.5_angle_tol-10_corner_eps-2_replace_thresh-5_nms-3_th-0.5_topk-1k/overview \
    --draw_overlap \
    --draw_gap \
    --draw_vertices
    
    # ----- GCP -----
    python visualize_all_categories_vectorized_fancy.py \
    --pred_dir ./PS_cluster_baseline_results/GCP/work_dirs/summary_deventer_512/categories \
    --out_dir ./PS_cluster_baseline_results/GCP/work_dirs/summary_deventer_512/overview_fancy_vertex_orange_4px \
    --draw_overlap \
    --draw_gap \
    --draw_vertices 
    
    # DP
    python visualize_all_categories_vectorized_fancy.py \
    --pred_dir ./HiSup-main/outputs/deventer_vmamba-small_512_vh_m_ldm_v1.5_simp_gt_b8/dp_1 \
    --out_dir ./HiSup-main/outputs/deventer_vmamba-small_512_vh_m_ldm_v1.5_simp_gt_b8/dp_1_overview_fancy \
    --draw_vertices \
    --draw_overlap \
    --draw_vertices 
    
    # Ablation: OPC on, VSS off (PSLG + DP)
    python visualize_all_categories_vectorized_fancy.py \
    --pred_dir ./HiSup-main/outputs/deventer_vmamba-small_512_vh_m_ldm_v1.5_simp_gt_b8/ddim/aba_topo_recon/no_vss_dp-2/categories \
    --out_dir ./HiSup-main/outputs/deventer_vmamba-small_512_vh_m_ldm_v1.5_simp_gt_b8/ddim/aba_topo_recon/no_vss_dp-2/overview_fancy \
    --draw_overlap \
    --draw_gap \
    --draw_vertices
    
    # ----- HiSup -----
    python visualize_all_categories_vectorized_fancy.py \
    --pred_dir ../work3/HiSup-main/outputs/deventer_512_categories \
    --out_dir ../work3/HiSup-main/outputs/deventer_512_overview_fancy_vertex_orange_4px \
    --draw_overlap \
    --draw_gap \
    --draw_vertices
    
    # ----- TopDiG -----
    python visualize_all_categories_vectorized_fancy.py \
    --pred_dir ./PS_cluster_baseline_results/TopDiG/records/Deventer_512_summary \
    --out_dir ./PS_cluster_baseline_results/TopDiG/records/Deventer_512_summary/overview_fancy_vertex_orange_4px \
    --draw_overlap \
    --draw_gap \
    --draw_vertices 
    """
