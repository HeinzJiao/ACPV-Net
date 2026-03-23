#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for PSLG-based polygonization.

This module provides helper routines for label-map cleanup, global boundary
construction, face extraction, vertex snapping, ring vectorization, and
visualization used by the PSLG polygonization scripts.
"""

import json
import argparse
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import unary_union, polygonize
import cv2
import os
from scipy.ndimage import distance_transform_edt


def load_label_map(npy_path):
    L = np.load(npy_path)
    return L.astype(np.int32)


def load_pred_vertices(json_path):
    with open(json_path, 'r') as f:
        arr = json.load(f)
    if len(arr) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(arr, dtype=np.int32)


def merge_small_regions_by_majority(L, min_area=20, connectivity=4, win_sizes=(3, 7)):
    """
    Remove tiny connected components and merge them into nearby majority labels.
    """
    H, W = L.shape
    L_clean = L.copy()
    unknown = -1

    present_labels = np.unique(L_clean)  # e.g., [0..6]

    # Step 1: remove small connected components class by class.
    for c in present_labels:
        mask = (L_clean == c).astype(np.uint8)
        num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                L_clean[lab == i] = unknown

    # Step 2: fill unknown pixels with local majority voting.
    for k in win_sizes:
        U = (L_clean == unknown)
        if not U.any():
            break
        kernel = np.ones((k, k), np.uint8)

        # Count how many pixels of each label appear inside the local window.
        counts = []
        for c in present_labels:
            bin_map = (L_clean == c).astype(np.uint8)
            cnt = cv2.filter2D(bin_map, -1, kernel, borderType=cv2.BORDER_REPLICATE)
            counts.append(cnt)
        counts = np.stack(counts, axis=-1)  # H x W x C
        best_idx = np.argmax(counts, axis=-1)
        assign_map = present_labels[best_idx]

        L_clean[U] = assign_map[U]

    # Step 3: fill any remaining unknown pixels with the nearest label.
    U = (L_clean == unknown)
    if U.any():
        dstack = []
        for c in present_labels:
            # Distance to label `c` is computed by EDT on its complement.
            dstack.append(distance_transform_edt(L_clean != c))
        dstack = np.stack(dstack, axis=-1)  # H x W x C
        best_idx = np.argmin(dstack, axis=-1)
        assign_map = present_labels[best_idx]
        L_clean[U] = assign_map[U]

    return L_clean


# ---------- Step 1: global boundary network from label transitions ----------
def build_global_segments(L, include_frame=True):
    """
    Build unit grid segments from label transitions in a multi-class map.
    """
    H, W = L.shape
    segs = []

    # Vertical segments from horizontal label changes.
    diff_v = (L[:, 1:] != L[:, :-1])
    ys, xs = np.where(diff_v)
    for y, x in zip(ys, xs + 1):
        segs.append(((int(x), int(y)), (int(x), int(y + 1))))

    # Horizontal segments from vertical label changes.
    diff_h = (L[1:, :] != L[:-1, :])
    ys, xs = np.where(diff_h)
    for y, x in zip(ys + 1, xs):
        segs.append(((int(x), int(y)), (int(x + 1), int(y))))

    if include_frame:
        # Top and bottom image frame.
        for x in range(W):
            segs.append(((x, 0), (x + 1, 0)))
            segs.append(((x, H), (x + 1, H)))
        # Left and right image frame.
        for y in range(H):
            segs.append(((0, y), (0, y + 1)))
            segs.append(((W, y), (W, y + 1)))

    return segs


def polygonize_from_segments(segments):
    ls = [LineString([p0, p1]) for (p0, p1) in segments]
    mls = MultiLineString(ls)
    merged = unary_union(mls)
    return list(polygonize(merged))


# ---------- Step 2: structural corners from vertex degree ----------
def compute_vertex_degrees(segments):
    deg = {}
    for (p0, p1) in segments:
        p0 = tuple(p0)
        p1 = tuple(p1)
        deg[p0] = deg.get(p0, 0) + 1
        deg[p1] = deg.get(p1, 0) + 1
    return deg


def structural_corners_from_degrees(deg_dict):
    return {k for k, v in deg_dict.items() if v != 2}


# ---------- Step 3: snap predicted vertices to the boundary network ----------
def collect_boundary_vertices(segments, H, W, include_frame=True):
    vs = set()
    for (p0, p1) in segments:
        vs.add((int(p0[0]), int(p0[1])))
        vs.add((int(p1[0]), int(p1[1])))

    if include_frame:
        for x in range(0, W + 1):
            vs.add((x, 0))
            vs.add((x, H))
        for y in range(0, H + 1):
            vs.add((0, y))
            vs.add((W, y))

    arr = np.array(list(vs), dtype=np.int32)
    return arr, vs


# Build a simple adjacency map for the segment graph.
def build_adjacency(segments):
    adj = {}
    for p0, p1 in segments:
        adj.setdefault(p0, set()).add(p1)
        adj.setdefault(p1, set()).add(p0)
    return adj


# Test whether a boundary vertex is an important turning point.
def is_important_corner(v, adj, tol_deg=15.0):
    nbrs = list(adj.get((int(v[0]), int(v[1])), ()))
    d = len(nbrs)
    if d != 2:
        return True  # Endpoint, T-junction, or X-junction.
    vx, vy = int(v[0]), int(v[1])
    v1 = np.array([nbrs[0][0]-vx, nbrs[0][1]-vy], dtype=float)
    v2 = np.array([nbrs[1][0]-vx, nbrs[1][1]-vy], dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return False
    cosang = np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))  # [0,180]
    return abs(ang - 90.0) <= float(tol_deg)


def snap_points_to_boundary(pred_xy, boundary_xy, dist_thresh=3.0):
    if pred_xy.shape[0] == 0 or boundary_xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int32), set()
    tree = cKDTree(boundary_xy.astype(np.float32))
    d, idx = tree.query(pred_xy.astype(np.float32), k=1)
    keep = d <= float(dist_thresh)
    snapped = boundary_xy[idx[keep]]
    snapped_set = {(int(x), int(y)) for x, y in snapped}
    snapped_arr = np.array(list(snapped_set), dtype=np.int32)
    return snapped_arr, snapped_set


# kNN snapping with distance-first search and on-the-fly corner checks.
def snap_points_to_boundary_knn_distance_first_simple(
    pred_xy, boundary_xy, segments,
    dist_thresh=3.0, k=8, tol_deg=15.0
):
    """
    Snap predicted vertices onto boundary vertices with a distance-first policy.

    For each predicted point, query `k` nearest boundary vertices, search them
    from near to far, prefer the first important corner within `dist_thresh`,
    and otherwise fall back to the nearest valid candidate.
    """
    if pred_xy.shape[0] == 0 or boundary_xy.shape[0] == 0:
        return np.zeros((0,2), dtype=np.int32), set()

    # Build the adjacency map once for online corner checks.
    adj = build_adjacency(segments)

    k = min(int(k), boundary_xy.shape[0])
    tree = cKDTree(boundary_xy.astype(np.float32))
    d, idx = tree.query(pred_xy.astype(np.float32), k=k)
    if k == 1:
        d = d[:, None]; idx = idx[:, None]

    keep = set()
    for i in range(d.shape[0]):
        # The closest boundary candidate decides whether the point is usable.
        order = np.argsort(d[i])
        j0 = idx[i][order[0]]
        d0 = d[i][order[0]]
        if d0 > float(dist_thresh):
            continue

        chosen = None
        # Search candidates from near to far until the threshold is exceeded.
        for j in idx[i][order]:
            if d[i][np.where(idx[i]==j)[0][0]] > float(dist_thresh):
                break
            v = boundary_xy[j]
            if is_important_corner(v, adj, tol_deg=tol_deg):
                chosen = j
                break
        if chosen is None:
            chosen = j0  # No corner was found within the threshold.

        x, y = boundary_xy[chosen]
        keep.add((int(x), int(y)))

    if not keep:
        return np.zeros((0,2), dtype=np.int32), set()
    arr = np.asarray(list(keep), dtype=np.int32)
    return arr, keep


# ---------- Step 4: extract category-specific rings from polygonized faces ----------
def label_of_face(face_poly, L):
    cx, cy = face_poly.representative_point().coords[0]
    H, W = L.shape
    px = int(np.clip(np.floor(cx), 0, W - 1))
    py = int(np.clip(np.floor(cy), 0, H - 1))
    return int(L[py, px])


def extract_category_faces_rings(faces, L, target_cat):
    H, W = L.shape
    polys = []
    for poly in faces:
        if not isinstance(poly, Polygon):
            continue
        if label_of_face(poly, L) != target_cat:
            continue

        ext = list(poly.exterior.coords)
        if len(ext) <= 3:
            continue
        if ext[0] == ext[-1]:
            ext = ext[:-1]
        exterior = np.array(ext, dtype=np.float32).round().astype(np.int32)

        holes = []
        for ring in poly.interiors:
            coords = list(ring.coords)
            if len(coords) <= 3:
                continue
            if coords[0] == coords[-1]:
                coords = coords[:-1]
            holes.append(np.array(coords, dtype=np.float32).round().astype(np.int32))

        polys.append({"exterior": exterior, "interiors": holes})
    return polys


# ---------- Step 5: keep key ring points and flatten the vertex sequence ----------
def vectorize_ring_with_points(ring_xy, keep_points_set):
    kept = []
    last = None
    for x, y in ring_xy:
        if (int(x), int(y)) in keep_points_set:
            if last is None or (x, y) != last:
                kept.append((int(x), int(y)))
                last = (int(x), int(y))
    if len(kept) < 3:
        return []
    flat = []
    for x, y in kept:
        flat.extend([int(x), int(y)])
    return flat


def collect_category_boundary_vertices(poly_rings):
    """
    Collect all boundary vertices from one category, including hole rings.
    """
    vs = set()
    for item in poly_rings:
        for x, y in item["exterior"]:
            vs.add((int(x), int(y)))
        for hole in item["interiors"]:
            for x, y in hole:
                vs.add((int(x), int(y)))
    if len(vs) == 0:
        return np.zeros((0, 2), dtype=np.int32), set()
    arr = np.array(list(vs), dtype=np.int32)
    return arr, vs


# ===== Visualization helpers =====
def make_canvas(H, W, scale=2):
    """Create a white canvas that also covers the grid border at `H` and `W`."""
    canvas = np.ones(( (H+1)*scale, (W+1)*scale, 3 ), dtype=np.uint8) * 255
    return canvas


def _to_img_pt(x, y, scale):
    return (int(x*scale), int(y*scale))


def draw_segments(canvas, segments, color=(0,0,0), thickness=1, scale=2):
    """Draw a list of grid-aligned segments."""
    for (p0, p1) in segments:
        x0,y0 = p0; x1,y1 = p1
        cv2.line(canvas, _to_img_pt(x0,y0,scale), _to_img_pt(x1,y1,scale), color, thickness, lineType=cv2.LINE_AA)


def draw_points(canvas, pts_xy, color, radius=2, scale=2):
    """Draw integer grid points or pixel points."""
    for x,y in pts_xy:
        cv2.circle(canvas, _to_img_pt(int(x), int(y), scale), radius, color, -1, lineType=cv2.LINE_AA)


def _paint_points_pixels(canvas, pts, bgr, s=1):
    """
    Paint points by coloring exactly one pixel per point.
    canvas: uint8 (H*s, W*s, 3)
    pts: iterable of (x, y) in original lattice/image coords
    bgr: tuple(int,int,int)
    s: vis scale
    """
    if pts is None:
        return

    H, W = canvas.shape[:2]
    for (x, y) in pts:
        cx, cy = _to_img_pt(int(x), int(y), s)  # map to scaled image coords
        if 0 <= cy < H and 0 <= cx < W:
            canvas[cy, cx] = bgr


def _set_points_pixels(canvas, pts_xy, color=(255,0,255), scale=1):
    """
    Paint points as single pixels instead of circles.
    """
    Hs, Ws = canvas.shape[:2]
    bgr = (int(color[0]), int(color[1]), int(color[2]))
    for x, y in pts_xy:
        ix, iy = _to_img_pt(int(x), int(y), scale)
        if 0 <= ix < Ws and 0 <= iy < Hs:
            canvas[iy, ix] = bgr


def draw_polyline(canvas, ring_xy, color=(0,0,0), thickness=1, closed=True, scale=1, antialias=False):
    if len(ring_xy) >= 2:
        pts = np.array([_to_img_pt(int(x),int(y),scale) for x,y in ring_xy], dtype=np.int32)
        lt = cv2.LINE_AA if antialias else cv2.LINE_8
        cv2.polylines(canvas, [pts], closed, color, thickness, lineType=lt)


def flatten_to_xylist(flat):
    """[x1,y1,x2,y2,...] -> [(x1,y1), (x2,y2), ...]"""
    a = np.asarray(flat, dtype=np.int32).reshape(-1,2)
    return [(int(x),int(y)) for x,y in a]


def draw_filled_polygon_with_holes_safe(canvas, ext_xy, holes_xy_list,
                                        fill_color=(255,255,255),
                                        edge_color=(0,0,0),
                                        point_color=(255,0,255),
                                        thickness=1, radius=1, scale=1,
                                        overlap_color=(0,128,255)):
    """
    Safely draw one polygon instance with holes on a shared canvas.

    The function avoids overwriting already painted regions, carves holes only
    from the current instance color, and highlights interior overlaps without
    changing the boundary pixels.
    """
    Hs, Ws = canvas.shape[:2]

    # Step 1: build masks for the exterior ring and all holes.
    ext_mask = np.zeros((Hs, Ws), dtype=np.uint8)
    ext_pts  = np.array([_to_img_pt(x, y, scale) for x, y in ext_xy], dtype=np.int32)
    cv2.fillPoly(ext_mask, [ext_pts], 1)

    hole_masks, hole_pts_list = [], []
    for hole_xy in holes_xy_list:
        hm = np.zeros((Hs, Ws), dtype=np.uint8)
        hole_pts = np.array([_to_img_pt(x, y, scale) for x, y in hole_xy], dtype=np.int32)
        cv2.fillPoly(hm, [hole_pts], 1)
        hole_masks.append(hm)
        hole_pts_list.append(hole_pts)

    # The geometric interior equals exterior minus the union of holes.
    if hole_masks:
        holes_union = np.zeros((Hs, Ws), dtype=np.uint8)
        for hm in hole_masks:
            holes_union |= hm
        interior_mask = (ext_mask == 1) & (holes_union == 0)
    else:
        interior_mask = (ext_mask == 1)

    # Step 2: fill only on untouched white background pixels.
    bg_mask   = (canvas[:, :, 0] == 255) & (canvas[:, :, 1] == 255) & (canvas[:, :, 2] == 255)
    fill_mask = (ext_mask == 1) & bg_mask
    if np.any(fill_mask):
        canvas[fill_mask] = np.array(fill_color, dtype=np.uint8)

    # Step 3: carve holes only from pixels belonging to this instance.
    if hole_masks:
        fc = np.array(fill_color, dtype=np.uint8)
        is_self_color = (canvas[:, :, 0] == fc[0]) & (canvas[:, :, 1] == fc[1]) & (canvas[:, :, 2] == fc[2])
        for hm in hole_masks:
            carve_mask = (hm == 1) & is_self_color
            if np.any(carve_mask):
                canvas[carve_mask] = (255, 255, 255)

    # Build a boundary mask so overlap highlighting does not touch edges.
    edge_mask = np.zeros((Hs, Ws), dtype=np.uint8)
    cv2.polylines(edge_mask, [ext_pts], True, 1, thickness=1, lineType=cv2.LINE_8)
    for hole_pts in hole_pts_list:
        cv2.polylines(edge_mask, [hole_pts], True, 1, thickness=1, lineType=cv2.LINE_8)

    # Highlight interior overlaps without changing the boundary pixels.
    fc = np.array(fill_color, dtype=np.uint8)
    is_self_color = (canvas[:, :, 0] == fc[0]) & (canvas[:, :, 1] == fc[1]) & (canvas[:, :, 2] == fc[2])
    colored_mask  = ~((canvas[:, :, 0] == 255) & (canvas[:, :, 1] == 255) & (canvas[:, :, 2] == 255))

    overlap_mask = interior_mask & (~is_self_color) & colored_mask & (edge_mask == 0)
    is_black_edge = (canvas[:, :, 0] == 0) & (canvas[:, :, 1] == 0) & (canvas[:, :, 2] == 0)
    overlap_mask &= (~is_black_edge)
    if np.any(overlap_mask):
        canvas[overlap_mask] = np.array(overlap_color, dtype=np.uint8)

    # Step 4: draw edges and vertices on top.
    draw_polyline(canvas, ext_xy, color=edge_color, thickness=thickness, closed=True, scale=scale)
    for hole_xy in holes_xy_list:
        draw_polyline(canvas, hole_xy, color=edge_color, thickness=thickness, closed=True, scale=scale)

    # _set_points_pixels(canvas, ext_xy, color=point_color, scale=scale)
    draw_points(canvas, ext_xy, color=point_color, radius=radius, scale=scale)
    for hole_xy in holes_xy_list:
        # _set_points_pixels(canvas, hole_xy, color=point_color, scale=scale)
        draw_points(canvas, hole_xy, color=point_color, radius=radius, scale=scale)


def break_segments_into_polylines(segments, struct_corners):
    """
    Split an undirected segment set into polylines at structural corners.
    """
    from collections import defaultdict, deque

    # Build the adjacency map and undirected edge set.
    adj = defaultdict(set)
    edges = set()
    for (x1, y1), (x2, y2) in segments:
        a = (int(x1), int(y1))
        b = (int(x2), int(y2))
        if a == b:
            continue
        adj[a].add(b)
        adj[b].add(a)
        e = frozenset((a, b))
        edges.add(e)

    # Vertex degree table.
    deg = {v: len(nbs) for v, nbs in adj.items()}
    struct_corners = set((int(x), int(y)) for (x, y) in struct_corners)

    visited = set()  # Visited undirected edges.
    polylines = []

    def next_unvisited_neighbor(u, prev=None):
        """Return one unvisited neighbor of `u` other than `prev`, if any."""
        for w in adj[u]:
            if prev is not None and w == prev:
                continue
            if frozenset((u, w)) not in visited:
                return w
        return None

    def walk_path(start, nb):
        """
        Walk along unvisited edges until a structural corner or endpoint is hit.
        """
        path = [start, nb]
        visited.add(frozenset((start, nb)))
        prev, cur = start, nb

        while True:
            # Stop when reaching a structural corner away from the start.
            if (cur in struct_corners or deg.get(cur, 0) != 2) and cur != start:
                break
            # Otherwise continue along the only remaining unvisited edge.
            nxt = next_unvisited_neighbor(cur, prev)
            if nxt is None:
                # No continuation is available.
                break
            visited.add(frozenset((cur, nxt)))
            path.append(nxt)
            prev, cur = cur, nxt

            # Stop when the walk closes a loop.
            if cur == start:
                break

        # Keep only non-trivial paths.
        if len(path) >= 2:
            polylines.append(path)

    # Pass 1: start from structural corners to cover open chains first.
    for v in adj.keys():
        if deg.get(v, 0) != 2 or v in struct_corners:
            for nb in list(adj[v]):
                e = frozenset((v, nb))
                if e in visited:
                    continue
                walk_path(v, nb)

    # Pass 2: process any remaining pure cycles.
    for e in list(edges):
        if e in visited:
            continue
        a, b = tuple(e)
        # Start from one remaining edge and walk around the cycle.
        path = [a, b]
        visited.add(frozenset((a, b)))
        prev, cur = a, b

        while True:
            nxt = next_unvisited_neighbor(cur, prev)
            if nxt is None:
                # Stop if the walk unexpectedly ends before closing the loop.
                break
            visited.add(frozenset((cur, nxt)))
            path.append(nxt)
            prev, cur = cur, nxt
            if cur == a:
                # The cycle is closed.
                break

        if len(path) >= 2:
            polylines.append(path)

    return polylines


def _simplify_snapped_on_polyline_v1(pl_xy, snapped_pts_set, is_closed, angle_tol_deg=5):
    """
    Simplify snapped points on one polyline with an angle threshold.

    Open polylines use the two endpoints as anchors, while closed rings use
    cyclic neighbors. The function returns only the retained snapped points.
    """
    if len(snapped_pts_set) == 0:
        return []

    pl_xy = np.asarray(pl_xy, dtype=np.int32)
    idx_map = {tuple(pl_xy[i]): i for i in range(len(pl_xy))}
    # Map snapped points to ordered indices on the polyline.
    idxs = sorted({idx_map.get(tuple(p)) for p in snapped_pts_set if tuple(p) in idx_map})
    if len(idxs) == 0:
        return []

    def _turn_angle(a, b, c):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        c = np.asarray(c, dtype=np.float32)
        v1 = b - a
        v2 = c - b
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 180.0  # Keep degenerate cases conservatively.
        cosang = float(np.dot(v1, v2) / (n1 * n2))
        cosang = max(-1.0, min(1.0, cosang))
        ang = float(np.degrees(np.arccos(cosang)))
        return ang

    kept = []

    if not is_closed:
        # Open polyline: use endpoints as anchors.
        start_idx = 0
        end_idx = len(pl_xy) - 1
        aug = [start_idx] + idxs + [end_idx]

        for t in range(1, len(aug) - 1):
            i_prev, i_cur, i_next = aug[t - 1], aug[t], aug[t + 1]
            # Evaluate only snapped points, not the anchors.
            if i_cur not in idxs:
                continue
            ang = _turn_angle(pl_xy[i_prev], pl_xy[i_cur], pl_xy[i_next])
            if ang > angle_tol_deg:
                kept.append(i_cur)

        # Return snapped points only, not endpoints.
        kept = sorted(set(kept))

    else:
        # Closed ring: use cyclic neighbors.
        n = len(idxs)
        for t in range(n):
            i_prev = idxs[(t - 1) % n]
            i_cur = idxs[t]
            i_next = idxs[(t + 1) % n]
            ang = _turn_angle(pl_xy[i_prev], pl_xy[i_cur], pl_xy[i_next])
            if ang > angle_tol_deg:
                kept.append(i_cur)
        kept = sorted(set(kept))
        if len(kept) < 3:
            return []

    return [tuple(pl_xy[i]) for i in kept]


def _simplify_snapped_on_polyline_v2(pl_xy, snapped_pts_set, is_closed,
                                     angle_tol_deg=5, min_sep=2.0):
    """
    Version 2: sparsify by arc-length spacing and then apply angle filtering.
    """
    if len(snapped_pts_set) == 0:
        return []

    pl_xy = np.asarray(pl_xy, dtype=np.int32)

    # Step 1: map snapped points to ordered indices on the polyline.
    idx_map = {tuple(pl_xy[i]): i for i in range(len(pl_xy))}
    idxs = sorted({idx_map.get(tuple(p)) for p in snapped_pts_set if tuple(p) in idx_map})
    if len(idxs) == 0:
        return []

    # Step 2: precompute cumulative arc length.
    if len(pl_xy) >= 2:
        d = pl_xy[1:] - pl_xy[:-1]
        seglen = np.sqrt((d[:,0].astype(np.float32)**2 + d[:,1].astype(np.float32)**2))
        cum = np.concatenate([[0.0], np.cumsum(seglen, dtype=np.float32)])
    else:
        cum = np.array([0.0], dtype=np.float32)
    total_len = float(cum[-1]) if len(cum) else 0.0

    # Step 3: greedily enforce a minimum arc-length spacing.
    kept_idx = []
    last_s = None
    for i in idxs:
        s = float(cum[i])
        if last_s is None or (s - last_s) >= float(min_sep):
            kept_idx.append(i)
            last_s = s

    # For closed rings, also enforce spacing across the wrap-around seam.
    if is_closed and len(kept_idx) >= 2 and total_len > 0:
        while len(kept_idx) >= 2:
            s_first = float(cum[kept_idx[0]])
            s_last = float(cum[kept_idx[-1]])
            wrap_gap = (total_len - s_last + s_first)
            if wrap_gap >= float(min_sep):
                break
            kept_idx.pop()

    idxs = kept_idx
    if len(idxs) == 0:
        return []

    # Step 4: apply the same angle-threshold simplification as v1.
    def _turn_angle(a, b, c):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        c = np.asarray(c, dtype=np.float32)
        v1 = b - a; v2 = c - b
        n1 = float(np.linalg.norm(v1)); n2 = float(np.linalg.norm(v2))
        if n1 < 1e-6 or n2 < 1e-6:
            return 180.0
        cosang = float(np.dot(v1, v2) / (n1 * n2))
        cosang = max(-1.0, min(1.0, cosang))
        return float(np.degrees(np.arccos(cosang)))

    kept = []
    if not is_closed:
        # Open polyline: use endpoints as anchors and return snapped points only.
        start_idx, end_idx = 0, len(pl_xy) - 1
        aug = [start_idx] + idxs + [end_idx]
        for t in range(1, len(aug) - 1):
            i_prev, i_cur, i_next = aug[t - 1], aug[t], aug[t + 1]
            if i_cur not in idxs:
                continue
            ang = _turn_angle(pl_xy[i_prev], pl_xy[i_cur], pl_xy[i_next])
            if ang > angle_tol_deg:
                kept.append(i_cur)
        kept = sorted(set(kept))
    else:
        # Closed ring: use cyclic neighbors and require at least three points.
        n = len(idxs)
        for t in range(n):
            i_prev = idxs[(t - 1) % n]
            i_cur  = idxs[t]
            i_next = idxs[(t + 1) % n]
            ang = _turn_angle(pl_xy[i_prev], pl_xy[i_cur], pl_xy[i_next])
            if ang > angle_tol_deg:
                kept.append(i_cur)
        kept = sorted(set(kept))
        if len(kept) < 3:
            return []

    return [tuple(pl_xy[i]) for i in kept]

