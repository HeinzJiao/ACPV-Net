#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polygonize one semantic prediction with the PSLG-based post-processing pipeline.

This script reconstructs polygons from a multi-class label map and predicted
vertices by building a global boundary PSLG, selecting key points on boundary
polylines, polygonizing faces, and vectorizing category-specific rings.

Example:
    python tools/polygonize_pslg_one_image.py \
        --seg_path /path/to/seg.npy \
        --junction_json /path/to/vertices.json \
        --categories 0 1 2 3 4 \
        --vss --dp_fix
"""

import json
import argparse
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import unary_union, polygonize
import cv2
from scipy.spatial import cKDTree
import os
from scipy.ndimage import distance_transform_edt
from polygonize_utils import (load_pred_vertices, merge_small_regions_by_majority,
                              build_global_segments, polygonize_from_segments, compute_vertex_degrees,
                              structural_corners_from_degrees, collect_boundary_vertices, build_adjacency,
                              is_important_corner, snap_points_to_boundary,
                              snap_points_to_boundary_knn_distance_first_simple, label_of_face,
                              extract_category_faces_rings, vectorize_ring_with_points,
                              collect_category_boundary_vertices, 
                              _to_img_pt, flatten_to_xylist, 
                              make_canvas, draw_segments, draw_points, _paint_points_pixels, _set_points_pixels, draw_polyline, draw_filled_polygon_with_holes_safe,
                              break_segments_into_polylines,
                              )

CATEGORY_COLORS_BGR = {
        4: (237, 174, 120),  # Water       RGB(120,174,237)
        3: (181, 226, 173),  # Vegetation  RGB(173,226,181)
        2: (234, 245, 251),  # Unvegetated RGB(224,214,206)
        1: (193, 167, 141),  # Road        RGB(141,167,193)
        0: (199, 224, 242),  # Building    RGB(242,224,199)
    }

class VisRecorder:
    """
    Collect visualization payloads during processing;
    render them later in one place.
    """

    def __init__(self, enabled, vis_dir=None, vis_scale=1):
        self.enabled = enabled
        self.vis_dir = vis_dir
        self.vis_scale = vis_scale
        self.data = {}

    def put(self, key, value):
        if not self.enabled:
            return
        self.data[key] = value

    def put_cat(self, key, cat, value):
        if not self.enabled:
            return
        if key not in self.data:
            self.data[key] = {}
        self.data[key][int(cat)] = value


def render_all_visualizations(V: VisRecorder):
    """Render all recorded visualization items to files under V.vis_dir."""
    if not V.enabled:
        return

    os.makedirs(V.vis_dir, exist_ok=True)
    s = V.vis_scale
    H, W = V.data["HW"]
    segs = V.data["boundary_segments"]  # overdense PSLG
    pred_xy = V.data["pred_vertices"]  # extracted vertices from predicted vertex heatmaps

    ANCHOR_BGR = (68,78,196)
    PRED_XY_BGR = (43, 152, 22)
    SNAPPED_PTS_BGR = (0, 128, 255)

    # ---------- Step0: colored mask ----------
    if "mask_colored_bgr" in V.data:
        cv2.imwrite(os.path.join(V.vis_dir, "step0_mask_colored.png"), V.data["mask_colored_bgr"])

    # ---------- *** Step1: global boundary segments ----------
    if "overdense" in V.data:
        anchors = V.data["overdense"]["anchors"]

        canvas = make_canvas(H, W, scale=s)
        draw_segments(canvas, segs, color=(0,0,0), thickness=1, scale=s)
        draw_points(canvas, anchors, color=ANCHOR_BGR, radius=6, scale=s)
        cv2.imwrite(os.path.join(V.vis_dir, "step1_boundary.png"), canvas)

    # ---------- Step2: over-dense PSLG (segments + all vertices + anchors) ----------
    if "overdense" in V.data:
        allv = V.data["overdense"]["all_vertices"]
        anchors = V.data["overdense"]["anchors"]

        canvas = make_canvas(H, W, scale=s)
        draw_segments(canvas, segs, color=(0,0,0), thickness=1, scale=s)
        draw_points(canvas, allv, color=(176,104,84), radius=3, scale=s)
        draw_points(canvas, anchors, color=ANCHOR_BGR, radius=6, scale=s)
        cv2.imwrite(os.path.join(V.vis_dir, "step2_overdense_pslg.png"), canvas)

    # ---------- *** Step3: pred vertices on top of step2 ----------
    if "overdense" in V.data:
        canvas = make_canvas(H, W, scale=s)
        draw_points(canvas, pred_xy, color=PRED_XY_BGR, radius=6, scale=s)
        cv2.imwrite(os.path.join(V.vis_dir, "step3_pred_vertices.png"), canvas)

        allv = V.data["overdense"]["all_vertices"]
        anchors = V.data["overdense"]["anchors"]
        snapped_pts = V.data["vss_points"].get("snapped", [])

        canvas = make_canvas(H, W, scale=s)
        draw_segments(canvas, segs, color=(0,0,0), thickness=1, scale=s)
        # draw_points(canvas, allv, color=(176,104,84), radius=max(3, s-1), scale=s)
        draw_points(canvas, pred_xy, color=PRED_XY_BGR, radius=3, scale=s)
        draw_points(canvas, snapped_pts, color=SNAPPED_PTS_BGR, radius=3, scale=s)
        draw_points(canvas, anchors, color=ANCHOR_BGR, radius=3, scale=s)
        cv2.imwrite(os.path.join(V.vis_dir, "step3_pred_vertices_overlay.png"), canvas)

    # ---------- Step4: rings per category ----------
    if "rings_by_cat" in V.data:
        for cat, poly_rings in V.data["rings_by_cat"].items():
            canvas = make_canvas(H, W, scale=s)
            for item in poly_rings:
                draw_polyline(canvas, item["exterior"], color=(0,0,0), thickness=1, closed=True, scale=s)
                for hole in item["interiors"]:
                    draw_polyline(canvas, hole, color=(0,0,0), thickness=1, closed=True, scale=s)
            cv2.imwrite(os.path.join(V.vis_dir, f"step4_rings_cat{cat}.png"), canvas)

    # ---------- Step4b: rings + points per category ----------
    if "rings_points_by_cat" in V.data:
        for cat, payload in V.data["rings_points_by_cat"].items():
            canvas = payload["canvas_rings"].copy()
            draw_points(canvas, payload["snapped_pts"], color=(0,255,0), radius=max(1,s), scale=s)
            draw_points(canvas, payload["struct_pts"], color=(0,0,255), radius=max(1,s), scale=s)
            cv2.imwrite(os.path.join(V.vis_dir, f"step4_rings_points_cat{cat}.png"), canvas)

    # ---------- Step5: vectorized result per category ----------
    if "vectorized_by_cat" in V.data:
        for cat, results in V.data["vectorized_by_cat"].items():
            canvas = make_canvas(H, W, scale=s)
            for inst in results:
                ext_xy = flatten_to_xylist(inst[0])
                draw_polyline(canvas, ext_xy, color=(0,0,0), thickness=1, closed=True, scale=s)
                draw_points(canvas, ext_xy, color=(255,0,255), radius=max(1,s), scale=s)
                for hv in inst[1:]:
                    hole_xy = flatten_to_xylist(hv)
                    draw_polyline(canvas, hole_xy, color=(0,0,0), thickness=1, closed=True, scale=s)
                    draw_points(canvas, hole_xy, color=(255,0,255), radius=max(1,s), scale=s)
            cv2.imwrite(os.path.join(V.vis_dir, f"step5_vectorized_cat{cat}.png"), canvas)

    # ---------- ** Optimized PSLG ----------
    if "optimized_pslg" in V.data:
        opt_segments = V.data["optimized_pslg"]["segments"]
        opt_vertices = V.data["optimized_pslg"]["vertices"]
        anchors_opt = V.data["optimized_pslg"]["anchors"]

        canvas = make_canvas(H, W, scale=s)
        for (a, b) in opt_segments:
            x0, y0 = a; x1, y1 = b
            cv2.line(canvas, _to_img_pt(int(x0), int(y0), s), _to_img_pt(int(x1), int(y1), s),
                     (0,0,0), 1, lineType=cv2.LINE_AA)
        draw_points(canvas, opt_vertices, color=SNAPPED_PTS_BGR, radius=6, scale=s)
        # for (x, y) in anchors_opt:
        #     cx, cy = _to_img_pt(int(x), int(y), s)
        #     cv2.circle(canvas, (cx, cy), 3, ANCHOR_BGR, -1, lineType=cv2.LINE_AA)
        cv2.imwrite(os.path.join(V.vis_dir, "step3_optimized_pslg.png"), canvas)


def process_one_image_categories(
    seg_path,
    junction_json,
    categories,
    dist_thresh=5.0,
    corner_eps=2.0,
    vis_dir=None,
    vis_scale=1,
    vss=True,
    dp_fix=True,
    dp_epsilon_dp=1.0,
    dp_epsilon_fix=2.5,
):
    """
    Topological reconstruction (Overdense PSLG construction + Vertex-guided subset selection / DP) for one image.

    Inputs
    - seg_path: path to predicted multi-class segmentation label map (npy).
    - junction_json: path to predicted junction/vertex coordinates (json).
    - categories: iterable of category IDs to extract polygons for.

    Key hyper-parameters
    - dist_thresh: max distance for snapping predicted vertices onto boundary polylines (pixels).
    - corner_eps: radius to treat a point as "too close to structural corner" (pixels).
    - vss: main path. If True, use vertex-guided subset selection (snap predicted vertices to boundary polylines);
           if False, fall back to pure DP simplification.
        - dp_fix: optional safeguard (effective only when vss=True). After VSS, apply DP to recover
                  missing critical corners (e.g., near-90° turns) and merge them into the keep-point set
                  to avoid large shape distortion.
        - dp_epsilon_dp: RDP epsilon (in pixels) used for pure DP simplification when vss=False.
        - dp_epsilon_fix: RDP epsilon (in pixels) used by the dp_fix safeguard to detect candidate
                          corner-like points missed by VSS.

    Outputs
    - results_by_cat: Dict[int, List[instance]]
        instance := [exterior_flat, hole1_flat, ...]
        each ring is flattened coords [x0,y0,x1,y1,...] and has >= 6 length.
        invalid rings/instances are skipped.

    Visualization (optional)
    - If vis_dir is provided, the following intermediate/final figures will be saved:
      - step0_mask_colored.png: colorized label map after pre-cleaning (palette rendering).
      - step1_boundary.png: global boundary PSLG from label transitions (unit grid segments).
      - step2_overdense_pslg.png: over-dense PSLG view (all segments + all vertices + anchor corners).
      - step3_pred_vertices.png: predicted vertices overlaid on the over-dense PSLG (raw predictions).
      - step4_rings_cat{cat}.png: polygonized faces/rings (exterior + holes) for the given category.
      - step4_rings_points_cat{cat}.png: ring visualization with kept points overlay (snapped points + structural corners).
      - step5_vectorized_cat{cat}.png: final vectorized polygons after point-guided ring simplification.
      - step3_optimized_pslg.png: optimized PSLG reconstructed from final vectorized rings (deduplicated edges/vertices).
    """

    # ---- Vis recorder ----
    V = VisRecorder(enabled=bool(vis_dir), vis_dir=vis_dir, vis_scale=vis_scale)

    # ==========================================
    # --- Step 0. load + pre-clean label map ---
    # ==========================================
    L = np.load(seg_path).astype(np.int32)
    L = merge_small_regions_by_majority(L, min_area=20, connectivity=4, win_sizes=(3, 7))
    H, W = L.shape
    V.put("HW", (H, W))

    if V.enabled:
        vis_mask = colorize_mask_bgr(L, CATEGORY_COLORS_BGR)
        if vis_scale != 1:
            vis_mask = cv2.resize(vis_mask, None, fx=vis_scale, fy=vis_scale, interpolation=cv2.INTER_NEAREST)
        V.put("mask_colored_bgr", vis_mask)

    pred_xy = load_pred_vertices(junction_json)
    V.put("pred_vertices", pred_xy)

    # ===================================================
    # --- Step 1. build global boundary PSLG segments ---
    # ===================================================
    segments = build_global_segments(L, include_frame=True)
    V.put("boundary_segments", segments)

    # === Step 2. extract structural corners (degree != 2) ===
    deg = compute_vertex_degrees(segments)
    struct_corners = structural_corners_from_degrees(deg)  # set of (x, y)

    if V.enabled:
        all_vertices_overdense = _collect_vertices_from_segments(segments)
        V.put("overdense", {"all_vertices": all_vertices_overdense, "anchors": struct_corners})

    # ====================================================================================
    # --- Step 3. Break PSLG into polylines and select keep-points (DP / VSS / VSS + DP-fix) ---
    # ====================================================================================
    # 3.1 Break the global boundary PSLG at structural corners (degree != 2),
    #     yielding a set of polylines for local processing.
    polylines = break_segments_into_polylines(segments, struct_corners)  # List[List[(x, y)]]

    # 3.2 Always include image corners as anchors (must be preserved).
    #     Note: use (0,0), (W,0), (W,H), (0,H) instead of (W-1,H-1),
    #     since the boundary grid is defined on an (H+1) x (W+1) lattice.
    image_corner_anchors = {(0, 0), (W, 0), (W, H), (0, H)}
    struct_corners = set(struct_corners) | image_corner_anchors

    # Collect keep-points selected from all polylines (excluding structural corners)
    snapped_set_global = set()

    for pl in polylines:
        pl_xy = np.array(pl)  # polyline geometry: [[x, y], ...]
        if len(pl_xy) < 2:
            continue

        # --- Case A: no VSS, fall back to pure DP simplification ---
        if not vss:
            # Use DP to extract geometric keypoints directly from the polyline
            dp_idxs = rdp_indices_cv2(pl_xy, epsilon=dp_epsilon_dp, closed=False)
            dp_pts = [tuple(pl_xy[i]) for i in dp_idxs]

            # Discard DP points too close to structural corners
            merged_pts = []
            for px, py in dp_pts:
                if all((px - cx) ** 2 + (py - cy) ** 2 > corner_eps ** 2 for cx, cy in struct_corners):
                    merged_pts.append((px, py))

        # --- Case B: VSS only (snap predicted vertices, no DP safeguard) ---
        elif not dp_fix:
            # Snap predicted vertices onto the current polyline
            _, snapped_set_pl = snap_points_to_boundary(
                pred_xy, pl_xy, dist_thresh=dist_thresh
            )

            # Discard snapped points too close to structural corners
            merged_pts = []
            for px, py in snapped_set_pl:
                if all((px - cx) ** 2 + (py - cy) ** 2 > corner_eps ** 2 for cx, cy in struct_corners):
                    merged_pts.append((px, py))

        # --- Case C: VSS + DP safeguard (recover missing near-90° corners) ---
        else:
            # (a) DP-based keypoints as geometric safeguard
            dp_idxs = rdp_indices_cv2(pl_xy, epsilon=dp_epsilon_fix, closed=False)
            dp_keys_90 = select_right_angle_keys(
                pl_xy, dp_idxs, tol_deg=10.0  # keep near-90° turns only
            )

            # (b) Snap predicted vertices (VSS main path)
            _, snapped_set_pl = snap_points_to_boundary(
                pred_xy, pl_xy, dist_thresh=dist_thresh
            )

            # (c) Merge: snapped points dominate; DP keys fill in missed corners
            merged_pts = merge_snapped_with_dp_keys(
                snapped_pts=list(snapped_set_pl),
                dp_keys_90=dp_keys_90,
                struct_corners=struct_corners,
                corner_eps=corner_eps,
                replace_thresh=5.0
            )

        # Accumulate keep-points from this polyline
        snapped_set_global.update(merged_pts)

    # 3.3 Final keep-point set: structural corners ∪ selected polyline points
    keep_points_set = set(struct_corners) | snapped_set_global

    if V.enabled:
        V.put("vss_points", {
        "anchors": list(struct_corners),              # deg!=2 + image corners
        "snapped": list(snapped_set_global),          # VSS snapped (excluding anchors)
        "keep": list(keep_points_set)                 # anchors ∪ snapped
    })

    # ==============================================================
    # --- Step 4. polygonize -> faces (shared by all categories) ---
    # ==============================================================
    faces = polygonize_from_segments(segments)

    # collect optimized PSLG from final rings
    opt_edge_keys = set()
    opt_vertices = set()

    results_by_cat = {}
    rings_by_cat = {}
    rings_points_by_cat = {}
    vectorized_by_cat = {}

    for cat in categories:
        cat = int(cat)

        # rings for this category
        poly_rings = extract_category_faces_rings(faces, L, target_cat=cat)
        rings_by_cat[cat] = poly_rings

        # ring points overlay payload (optional)
        if V.enabled:
            # Render the ring-only base image first; the renderer overlays points later.
            canvas4 = make_canvas(H, W, scale=vis_scale)
            ring_points_set = set()
            for item in poly_rings:
                draw_polyline(canvas4, item["exterior"], color=(0, 0, 0), thickness=1, closed=True, scale=vis_scale)
                for hole in item["interiors"]:
                    draw_polyline(canvas4, hole, color=(0, 0, 0), thickness=1, closed=True, scale=vis_scale)

                for xy in item["exterior"]:
                    ring_points_set.add(tuple(xy))
                for hole in item["interiors"]:
                    for xy in hole:
                        ring_points_set.add(tuple(xy))

            cat_struct_pts = [p for p in struct_corners if tuple(p) in ring_points_set]
            cat_snapped_pts = [p for p in snapped_set_global if tuple(p) in ring_points_set]
            rings_points_by_cat[cat] = {
                "canvas_rings": canvas4,
                "struct_pts": cat_struct_pts,
                "snapped_pts": cat_snapped_pts
            }

        # vectorize rings with keep_points_set
        results = []
        for item in poly_rings:
            ext_vec = vectorize_ring_with_points(item["exterior"], keep_points_set)
            # merge near-parallel edges
            ext_vec = simplify_polyline_by_angle(ext_vec, angle_thresh_deg=0.1, closed=True,
                                                 protected=struct_corners, max_passes=5)

            # Clip coordinates back to the valid image domain.
            for i in range(0, len(ext_vec), 2):
                ext_vec[i] = max(0, min(W - 1, int(ext_vec[i])))
                ext_vec[i + 1] = max(0, min(H - 1, int(ext_vec[i + 1])))

            # Remove consecutive duplicates, including a repeated start/end point.
            tmp = []
            for i in range(0, len(ext_vec), 2):
                p = (ext_vec[i], ext_vec[i + 1])
                if (not tmp) or p != tmp[-1]:
                    tmp.append(p)
            if len(tmp) >= 2 and tmp[0] == tmp[-1]:
                tmp.pop()
            ext_vec = [v for x, y in tmp for v in (x, y)]

            if len(ext_vec) < 6:
                continue

            inst = [ext_vec]
            for hole in item["interiors"]:
                hv = vectorize_ring_with_points(hole, keep_points_set)
                # merge near-parallel edges
                hv = simplify_polyline_by_angle(hv, angle_thresh_deg=0.1, closed=True,
                                                protected=struct_corners, max_passes=5)

                # Clip coordinates back to the valid image domain.
                for i in range(0, len(hv), 2):
                    hv[i] = max(0, min(W - 1, int(hv[i])))
                    hv[i + 1] = max(0, min(H - 1, int(hv[i + 1])))

                # Remove consecutive duplicates, including a repeated start/end point.
                tmp = []
                for i in range(0, len(hv), 2):
                    p = (hv[i], hv[i + 1])
                    if (not tmp) or p != tmp[-1]:
                        tmp.append(p)
                if len(tmp) >= 2 and tmp[0] == tmp[-1]:
                    tmp.pop()
                hv = [v for x, y in tmp for v in (x, y)]

                if len(hv) >= 6:
                    inst.append(hv)
            results.append(inst)

        # collect optimized PSLG edges from final result
        for inst in results:
            ext_xy = flatten_to_xylist(inst[0])
            for (a, b) in _rings_to_segments(ext_xy, closed=True):
                opt_edge_keys.add(_norm_edge_key(a, b))
                opt_vertices.add(a)
                opt_vertices.add(b)
            for hv in inst[1:]:
                hole_xy = flatten_to_xylist(hv)
                for (a, b) in _rings_to_segments(hole_xy, closed=True):
                    opt_edge_keys.add(_norm_edge_key(a, b))
                    opt_vertices.add(a)
                    opt_vertices.add(b)

        results_by_cat[cat] = results
        vectorized_by_cat[cat] = results

    # ---- dump visualization payloads ----
    V.put("rings_by_cat", rings_by_cat)
    V.put("rings_points_by_cat", rings_points_by_cat)
    V.put("vectorized_by_cat", vectorized_by_cat)

    if V.enabled:
        opt_segments = [(a, b) for (a, b) in opt_edge_keys]
        deg_opt = compute_vertex_degrees(opt_segments)
        struct_corners_opt = structural_corners_from_degrees(deg_opt)
        V.put("optimized_pslg", {"segments": opt_segments, "vertices": opt_vertices, "anchors": struct_corners_opt})

        render_all_visualizations(V)

    return results_by_cat


def simplify_polyline_by_angle(
    flat_coords,
    angle_thresh_deg = 2.0,
    closed = True,          # Keep True for closed rings without duplicated endpoints.
    protected = None,  # {(x, y), ...} points protected on the integer grid
    max_passes = 10,
):
    """
    Merge nearly collinear adjacent edges by removing the middle vertex.

    Args:
        flat_coords: Flattened coordinates for either a ring or an open polyline.
        closed: Treat the input as a closed ring without duplicating endpoints.
        protected: Optional set of integer-grid points that must be preserved.
        max_passes: Safety cap for the iterative angle-based simplification.

    Returns:
        Flattened coordinates in the same format, still without endpoint
        duplication.
    """
    pts = np.asarray(flat_coords, dtype=np.float32).reshape(-1, 2)
    min_n = 3 if closed else 2
    if pts.shape[0] <= min_n:
        return flat_coords

    prot = set() if protected is None else {
        (int(round(x)), int(round(y))) for (x, y) in protected
    }

    cos_thr = float(np.cos(np.deg2rad(angle_thresh_deg)))

    def _unit(v):
        n = float(np.linalg.norm(v))
        return (v / n) if n > 1e-9 else None

    for _ in range(max_passes):
        m = pts.shape[0]
        if m <= min_n:
            break

        keep = np.ones(m, dtype=bool)
        changed = False

        # For closed rings, every point can be tested as a middle vertex.
        idxs = range(m) if closed else range(1, m - 1)
        for i in idxs:
            if not keep[i]:
                continue

            ip = (i - 1) % m if closed else i - 1
            inx = (i + 1) % m if closed else i + 1
            if ip < 0 or inx >= m:     # Preserve open-polyline endpoints.
                continue

            b = pts[i]
            # Never remove protected points.
            if (int(round(b[0])), int(round(b[1]))) in prot:
                continue

            a, c = pts[ip], pts[inx]
            u1, u2 = _unit(b - a), _unit(c - b)
            if u1 is None or u2 is None:
                # Remove the middle point when a zero-length edge appears.
                keep[i] = False
                changed = True
                continue

            # Merge when adjacent edges are almost collinear and point forward.
            if float(np.clip(u1.dot(u2), -1.0, 1.0)) >= cos_thr:
                keep[i] = False
                changed = True

        if not changed:
            break
        pts = pts[keep]

    # Keep the existing convention: no duplicated first/last point.
    return pts.reshape(-1).astype(
        type(flat_coords[0]) if len(flat_coords) > 0 else np.float32
    ).tolist()


def colorize_mask_bgr(L: np.ndarray, palette_bgr: dict, fallback=(220, 210, 205)) -> np.ndarray:
    """
    Colorize an integer label map into a BGR image.

    Any class ID missing from the palette falls back to the provided color.
    """
    H, W = L.shape
    out = np.empty((H, W, 3), dtype=np.uint8)
    # Initialize the canvas with the fallback color.
    out[:] = np.array(fallback, dtype=np.uint8)
    # Assign colors class by class with vectorized masking.
    for k, bgr in palette_bgr.items():
        out[L == k] = bgr
    return out


def _collect_vertices_from_segments(segments):
    """Collect all unique vertices from a segment list."""
    V = set()
    for (p0, p1) in segments:
        x0,y0 = p0; x1,y1 = p1
        V.add((int(x0), int(y0)))
        V.add((int(x1), int(y1)))
    return V


def _rings_to_segments(ring_xy, closed=True):
    """Convert one ring point sequence to segments with integer endpoints."""
    segs = []
    n = len(ring_xy)
    if n >= 2:
        for i in range(n - 1):
            x0,y0 = ring_xy[i]; x1,y1 = ring_xy[i+1]
            a = (int(x0), int(y0)); b = (int(x1), int(y1))
            segs.append((a, b))
        if closed and n >= 3:
            x0,y0 = ring_xy[-1]; x1,y1 = ring_xy[0]
            a = (int(x0), int(y0)); b = (int(x1), int(y1))
            segs.append((a, b))
    return segs


def _norm_edge_key(a, b):
    """Return a canonical key for deduplicating undirected edges."""
    return (a, b) if a <= b else (b, a)


# Visualize the PSLG polylines after splitting them at anchor vertices.
def visualize_pslg_colored_cv2(polylines,
                               anchors,
                               H, W,
                               vis_dir,
                               vis_scale=2,
                               thickness=2,
                               fname="step1_polylines_colored.png",
                               palette_bgr=None,
                               annotate_index=False):
    """
    Visualize PSLG polylines after splitting at anchor vertices.

    Args:
        polylines: `List[List[(x, y)]]` or `List[np.ndarray (Ni, 2)]`.
        anchors: Iterable of `(x, y)` anchor points.
        H, W: Original image height and width.
        vis_dir: Output directory.
        vis_scale: Integer visualization scale factor.
        thickness: Line thickness in output pixels.
        fname: Output filename.
        palette_bgr: Optional BGR color palette.
        annotate_index: Whether to annotate each polyline with its index.
    """
    os.makedirs(vis_dir, exist_ok=True)

    # White canvas.
    Hs, Ws = int(H*vis_scale), int(W*vis_scale)
    canvas = np.full((Hs, Ws, 3), 255, dtype=np.uint8)

    # BGR palette.
    if palette_bgr is None:
        palette_bgr = [
            (31,119,180),(255,127,14),(44,160,44),(214,39,40),(148,103,189),
            (140,86,75),(227,119,194),(127,127,127),(188,189,34),(23,190,207),
            (174,199,232),(255,187,120),(152,223,138),(255,152,150),(197,176,213),
            (196,156,148),(247,182,210),(199,199,199),(219,219,141),(158,218,229)
        ]

    # Draw each polyline with a different color.
    for i, pl in enumerate(polylines):
        arr = np.asarray(pl, dtype=np.float32)
        if arr.shape[0] < 2:
            continue
        pts = (arr * float(vis_scale)).round().astype(np.int32).reshape(-1,1,2)
        color = palette_bgr[i % len(palette_bgr)]
        cv2.polylines(canvas, [pts], isClosed=False, color=color,
                      thickness=thickness, lineType=cv2.LINE_AA)

        if annotate_index:
            mid = arr[len(arr)//2]
            mx, my = int(round(mid[0]*vis_scale)), int(round(mid[1]*vis_scale))
            cv2.putText(canvas, str(i), (mx+3, my-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # Draw anchors in red.
    if anchors:
        for (x, y) in anchors:
            cx, cy = int(round(x*vis_scale)), int(round(y*vis_scale))
            cv2.circle(canvas, (cx, cy), max(3, vis_scale), (0,0,255), thickness=-1, lineType=cv2.LINE_AA)

    # Save the visualization.
    out_path = os.path.join(vis_dir, fname)
    cv2.imwrite(out_path, canvas)
    return out_path


def rdp_indices_cv2(points_np, epsilon, closed=False):
    """
    Extract DP keypoint indices with `cv2.approxPolyDP`.

    Returns ordered indices in the original polyline and always keeps
    endpoints.
    """
    if points_np.shape[0] <= 2 or epsilon <= 0:
        return list(range(points_np.shape[0]))
    cnt = points_np.astype(np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(cnt, epsilon, closed).reshape(-1, 2)

    # Map the DP vertices back to ordered indices in the original sequence.
    idxs = []
    used = set()
    last = -1
    for p in approx:
        d2 = np.sum((points_np - p)**2, axis=1)
        # Prefer the closest index after `last` to keep the order stable.
        d2_masked = d2.copy()
        d2_masked[:last+1] = np.inf
        j = int(np.argmin(d2_masked))
        if not np.isfinite(d2_masked[j]):
            j = int(np.argmin(d2))  # Fallback if the ordered search fails.
        if j not in used:
            idxs.append(j); used.add(j); last = j
    # Always keep both endpoints.
    if 0 not in used:
        idxs = [0] + idxs
    if (points_np.shape[0]-1) not in used:
        idxs = idxs + [points_np.shape[0]-1]
    # Return sorted unique indices.
    return sorted(set(idxs))


def select_right_angle_keys(pl_xy: np.ndarray,
                            dp_idxs,
                            tol_deg = 10.0,
                            closed = False):
    """
    Keep only DP keypoints whose local turn angle is close to 90 degrees.

    The angle is measured on the DP keypoint sequence itself rather than on the
    immediate neighbors in the dense input polyline.
    """
    pts = np.asarray(pl_xy, np.float32)
    if len(dp_idxs) < 3:
        return []

    kept = []
    K = len(dp_idxs)
    # Open polylines skip endpoints, while closed rings test all keypoints.
    if closed:
        it = range(K)  # Cyclic iteration for closed rings.
    else:
        it = range(1, K-1)

    for k in it:
        i_prev = dp_idxs[(k-1) % K]
        i_curr = dp_idxs[k]
        i_next = dp_idxs[(k+1) % K]
        if not closed and (k == 0 or k == K-1):
            continue

        a, b, c = pts[i_prev], pts[i_curr], pts[i_next]
        v1, v2 = a - b, c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        ang = np.degrees(np.arccos(cosang))
        if abs(ang - 90.0) <= tol_deg:
            kept.append((float(b[0]), float(b[1])))

    return kept


def merge_snapped_with_dp_keys(snapped_pts,
                               dp_keys_90,
                               struct_corners,
                               corner_eps = 1.0,
                               replace_thresh = 5.0):
    """
    Merge snapped points with DP-based 90-degree corner candidates.

    The main preference is given to snapped prediction points; DP candidates
    are kept only when they are not already covered by a nearby snapped point.
    """
    # Step 1: remove snapped points that are too close to structural anchors.
    filt_snapped = []
    r2_corner = corner_eps * corner_eps
    for (sx, sy) in snapped_pts:
        near = False
        for (cx, cy) in struct_corners:
            dx, dy = sx - cx, sy - cy
            if dx*dx + dy*dy <= r2_corner:
                near = True; break
        if not near:
            filt_snapped.append((float(sx), float(sy)))

    # Step 2: merge the remaining snapped points with DP candidates.
    remain_keys = list(dp_keys_90)
    r2_rep = replace_thresh * replace_thresh
    out = []

    # Start from snapped points so predictions take priority.
    out.extend(filt_snapped)

    # Keep a DP key only if no snapped point already explains it.
    for kx, ky in dp_keys_90:
        drop = False
        for (sx, sy) in filt_snapped:
            dx, dy = sx - kx, sy - ky
            if dx*dx + dy*dy <= r2_rep:
                drop = True; break
        if not drop:
            out.append((float(kx), float(ky)))

    # Step 3: deduplicate on the integer grid.
    out = list({(int(round(x)), int(round(y))) for (x, y) in out})
    # Return float tuples; downstream code casts them consistently.
    return [(float(x), float(y)) for (x, y) in out]


def main():
    parser = argparse.ArgumentParser(
        description="Vectorize multiple categories with global boundary + per-category snapping."
    )

    # ------------------------------------------------------------------
    # Required I/O
    # ------------------------------------------------------------------
    parser.add_argument('--seg_path', required=True,
                        help='H×W numpy label map (.npy), values in {0..6}')
    parser.add_argument('--junction_json', required=True,
                        help='JSON of predicted vertices: [[x,y], ...]')
    parser.add_argument('--categories', type=int, nargs='+', required=True,
                        help='Target category ids, e.g. --categories 0 1 2')
    parser.add_argument("--vis_dir", default=None,
                        help="Directory to save intermediate and final visualizations (optional)")
    parser.add_argument("--vis_scale", type=int, default=1,
                        help="Visualization scale factor (default: 1)")

    # ------------------------------------------------------------------
    # Core geoemtric hyper-parameters
    # ------------------------------------------------------------------
    parser.add_argument("--dist_thresh", type=float, default=5.0,
                        help="Snap radius for predicted vertices to boundary polylines (pixels)")
    parser.add_argument("--corner_eps", type=float, default=2.0,
                        help="Radius to treat a point as too close to a structural corner (pixels)")

    # ------------------------------------------------------------------
    # Point selection strategy
    # ------------------------------------------------------------------
    parser.add_argument("--vss", action="store_true",
                        help="Enable vertex-guided subset selection (VSS). "
                        "If not set, pure DP simplification is used.")
    parser.add_argument("--dp_fix", action="store_true",
                        help="Enable DP-based safeguard to recover missing critical corners "
                        "(effective only when --vss is set).")
    parser.add_argument("--dp_epsilon_dp", type=float, default=1,
                        help="RDP epsilon for pure DP fallback (used when --vss is not set)")
    parser.add_argument("--dp_epsilon_fix", type=float, default=2.0,
                        help="RDP epsilon for DP safeguard in VSS mode (--dp_fix)")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Run polygonization
    # ------------------------------------------------------------------
    results = process_one_image_categories(
        seg_path=args.seg_path,
        junction_json=args.junction_json,
        categories=args.categories,
        dist_thresh=args.dist_thresh,
        corner_eps=args.corner_eps,
        vis_dir=args.vis_dir,
        vis_scale=args.vis_scale,
        vss=args.vss,
        dp_fix=args.dp_fix,
        dp_epsilon_dp=args.dp_epsilon_dp,
        dp_epsilon_fix=args.dp_epsilon_fix,
    )

if __name__ == "__main__":
    main()
