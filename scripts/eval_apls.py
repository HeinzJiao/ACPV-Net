#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute APLS from COCO-format polygon annotations by converting masks to
skeleton graphs and comparing shortest-path structure between GT and predictions.

Inputs:
- A COCO-format GT JSON file.
- A COCO-format prediction JSON file.
- The corresponding image folder.
- A cache folder for GT graphs and an output folder for predicted graphs/results.

Outputs:
- Cached `.gpickle` graph files for GT and predictions.
- Optional graph visualizations.
- `apls_results.json` with per-image APLS scores and a dataset-level average.

Example:
    python eval_apls.py \
        --dt_file /path/to/prediction.json \
        --gt_file /path/to/ground_truth.json \
        --gt_folder /path/to/gt_graphs \
        --image_folder /path/to/images \
        --output_folder /path/to/output_dir
"""

import os
import cv2
import json
import argparse
import pickle
from collections import defaultdict

import numpy as np
import networkx as nx
from tqdm import tqdm
from skimage.morphology import skeletonize
from scipy.spatial import KDTree

# Hyperparameters used by the evaluation protocol.
SAMPLE_STEP = 8.0
SNAP_EPS = 0.5
MIN_EDGE_LEN = 2.0
SUBDIVIDE_LONG_EDGES = True
VIS_THICK = 2


def load_image_id_mapping(gt_annotation_file):
    """
    Build mapping from COCO image_id to file_name.
    """
    with open(gt_annotation_file, 'r') as f:
        gt_data = json.load(f)
    return {img['id']: img['file_name'] for img in gt_data['images']}

def extract_mask_from_coco(coco_json, image_id, image_shape):
    """
    Rasterize COCO polygon annotations (exterior + holes) into a binary mask.
    Exterior polygon is filled with 1; interior holes are subtracted.
    """
    with open(coco_json, 'r') as f:
        anns = json.load(f)

    H, W = image_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    for ann in anns:
        if ann['image_id'] != image_id:
            continue

        segmentation = ann.get('segmentation', [])
        for i, poly in enumerate(segmentation):
            poly_coords = np.array(poly, dtype=np.int32).reshape((-1, 2))
            if i == 0:
                cv2.fillPoly(mask, [poly_coords], 1)
            else:
                cv2.fillPoly(mask, [poly_coords], 0)

    return mask


def extract_skeleton(bin_mask):
    """
    Compute a 1-pixel-wide skeleton from a binary mask.
    """
    bin_mask = (bin_mask > 0).astype(np.uint8)
    sk = skeletonize(bin_mask).astype(np.uint8)
    return sk


def skeleton_to_pixel_graph(skeleton):
    """
    Convert a skeleton image into a pixel-level weighted graph.

    Each foreground pixel is a node.
    8-neighborhood connectivity is used:
      - weight = 1 for horizontal/vertical edges
      - weight = sqrt(2) for diagonal edges
    """
    G = nx.Graph()
    ys, xs = np.where(skeleton > 0)
    coords = list(zip(ys, xs))
    coord_set = set(coords)

    for y, x in coords:
        G.add_node((y, x))

    for y, x in coords:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx2 = y + dy, x + dx
                if (ny, nx2) in coord_set:
                    w = np.hypot(dy, dx)  
                    if not G.has_edge((y, x), (ny, nx2)):
                        G.add_edge((y, x), (ny, nx2), weight=w)

    return G


def simplify_graph(G):
    """
    Collapse degree-2 pixel chains into weighted edges.
    Only key nodes (degree != 2) are preserved.
    """
    if G.number_of_nodes() == 0:
        return nx.Graph()

    degree = dict(G.degree())
    key_nodes = {n for n, d in degree.items() if d != 2}

    if not key_nodes:
        nodes = list(G.nodes())
        if len(nodes) <= 2:
            return G.copy()
        ends = [n for n, d in degree.items() if d == 1]
        if len(ends) >= 2:
            key_nodes = set(ends)
        else:
            key_nodes = {nodes[0], nodes[-1]}

    SG = nx.Graph()
    for u in key_nodes:
        SG.add_node(u)

    visited = set()
    for u in key_nodes:
        for v in G.neighbors(u):
            if (u, v) in visited or (v, u) in visited:
                continue
            
            path_len = 0.0
            prev = u
            cur  = v
            
            while True:
                w = G[prev][cur]['weight']
                path_len += w
                visited.add((prev, cur))
                visited.add((cur, prev))

                if cur in key_nodes:
                    if u != cur:
                        if path_len >= MIN_EDGE_LEN:
                            SG.add_node(cur)
                            if SG.has_edge(u, cur):
                                if path_len < SG[u][cur]['weight']:
                                    SG[u][cur]['weight'] = path_len
                            else:
                                SG.add_edge(u, cur, weight=path_len)
                    break

                neighs = [x for x in G.neighbors(cur) if x != prev]
                if not neighs:
                    break
                nxt = neighs[0]
                prev, cur = cur, nxt

    return SG


def subdivide_edges_for_control_nodes(G, step=SAMPLE_STEP):
    """
    Subdivide long edges into shorter segments to explicitly insert control nodes.
    """
    if not SUBDIVIDE_LONG_EDGES:
        return G

    H = nx.Graph()
    for n in G.nodes():
        H.add_node(n)

    for u, v, data in G.edges(data=True):
        L = data.get('weight', 1.0)
        if L <= step:
            H.add_edge(u, v, weight=L)
            continue

        uy, ux = u
        vy, vx = v
        dy, dx = vy - uy, vx - ux
        # Normalize the edge direction while guarding against zero length.
        if L == 0:
            H.add_edge(u, v, weight=L)
            continue
        num_seg = int(np.ceil(L / step))
        last = u

        for k in range(1, num_seg):
            t = min(1.0, k / num_seg)
            ny = uy + dy * t
            nx2 = ux + dx * t
            mid = (float(ny), float(nx2))
            H.add_node(mid)
            seg_len = L / num_seg
            H.add_edge(last, mid, weight=seg_len)
            last = mid

        seg_len = L / num_seg
        H.add_edge(last, v, weight=seg_len)

    return H


def build_control_points(G, step=SAMPLE_STEP):
    """
    Build a set of control points for APLS evaluation.

    Control points include:
      1) All structural nodes (degree != 2), i.e., junctions/endpoints.
      2) (Optional) Equidistant points along long edges:
         - If SUBDIVIDE_LONG_EDGES=True, these points are already inserted
           as explicit nodes by subdivide_edges_for_control_nodes().
         - If SUBDIVIDE_LONG_EDGES=False, only the original endpoints exist.
      3) One anchor node per connected component (to ensure every component
         contributes at least one control point).

    Args:
        G (nx.Graph): Weighted graph.
        step (float): Target spacing for edge subdivision.

    Returns:
        list: A list of nodes (tuples) that exist in G.
    """
    control = set()

    # 1) Structural nodes: degree != 2
    deg = dict(G.degree())
    for n, d in deg.items():
        if d != 2:
            control.add(n)

    # 2) One anchor node per connected component
    for comp in nx.connected_components(G):
        comp_nodes = list(comp)
        if len(comp_nodes) == 0:
            continue
        control.add(comp_nodes[0])

    return list(control)


def kdtree_of_nodes(G):
    """
    Build a KDTree over graph node coordinates for nearest-neighbor queries.
    """
    nodes = np.array([[n[0], n[1]] for n in G.nodes()], dtype=np.float64)
    if len(nodes) == 0:
        return None, None
    tree = KDTree(nodes)
    node_list = list(G.nodes())
    return tree, node_list


def nearest_node_in_graph(tree, node_list, pt):
    """
    Query the nearest node in a graph (KDTree + node_list) for a given point.
    """
    if tree is None or len(node_list) == 0:
        return None
    dist, idx = tree.query([pt[0], pt[1]])
    return node_list[int(idx)], float(dist)


def symmetric_node_injection(G_gt, G_pr, CP_gt, CP_pr, eps=SNAP_EPS):
    """
    Symmetric node injection (SpaceNet APLS idea).

    For path comparison, we need corresponding nodes between GT and Pred graphs.
    This function maps control points from one graph to the nearest node in the
    other graph, in both directions.

    Returns:
        pairs_gt_to_pr (list): [(a, a2), ...] where a ∈ CP_gt, a2 ∈ V_pred
        pairs_pr_to_gt (list): [(b, b2), ...] where b ∈ CP_pr, b2 ∈ V_gt
    """
    tree_pr, list_pr = kdtree_of_nodes(G_pr)
    tree_gt, list_gt = kdtree_of_nodes(G_gt)

    # GT control points mapped onto Pred nodes
    pairs_gt_to_pr = []
    for a in CP_gt:
        a2, d = nearest_node_in_graph(tree_pr, list_pr, a)
        if a2 is not None and d <= eps:
            pairs_gt_to_pr.append((a, a2))
        else:
            if a2 is not None:
                pairs_gt_to_pr.append((a, a2))

    # Pred control points mapped onto GT nodes
    pairs_pr_to_gt = []
    for b in CP_pr:
        b2, d = nearest_node_in_graph(tree_gt, list_gt, b)
        if b2 is not None and d <= eps:
            pairs_pr_to_gt.append((b, b2))
        else:
            if b2 is not None:
                pairs_pr_to_gt.append((b, b2))

    return pairs_gt_to_pr, pairs_pr_to_gt


def all_pairs_from_sources(G, sources):
    """
    For a given set of source nodes `sources`:
    run single_source_dijkstra_path_length once per source node,
    and collect shortest-path distances only between nodes in `sources`.
    Return a dictionary of the form dict[(u, v)] = dist.
    """
    sources = list(sources)
    dmat = {}
    for s in sources:
        lengths = nx.single_source_dijkstra_path_length(G, s, weight='weight')
        for t in sources:
            if t == s:
                continue
            d = lengths.get(t, float('inf'))

            key = (s, t) if s < t else (t, s)
            if key in dmat:
                dmat[key] = min(dmat[key], d)
            else:
                dmat[key] = d
    return dmat


def compute_apls_symmetric(G_gt, G_pr, visualize=False, vis_image=None, vis_out_dir=None, fname=""):
    """
    Compute symmetric APLS using SpaceNet-style symmetric node injection.

    Two directions are computed and averaged implicitly by summing errors:

      Direction 1 (GT → Pred):
        - Use GT control points as anchors.
        - Compare shortest paths between GT node pairs vs. Pred node pairs
          after mapping GT points onto Pred nodes.

      Direction 2 (Pred → GT):
        - Use Pred control points as anchors.
        - Compare shortest paths between Pred node pairs vs. GT node pairs
          after mapping Pred points onto GT nodes.

    Error definition (per node pair):
      - If both paths are infinite: err = 0
      - If only one is infinite: err = 1
      - Otherwise: err = min(1, |L_gt - L_pr| / max(L_gt, eps))

    APLS:
      apls = 1 - mean(err)

    Args:
        G_gt (nx.Graph): GT pixel graph.
        G_pr (nx.Graph): Pred pixel graph.
        visualize: Reserved (not used here; kept for interface compatibility).
        vis_image / vis_out_dir / fname: Reserved.

    Returns:
        float or None:
            - None if GT graph is empty (skip).
            - 0.0 if Pred graph is empty.
            - Otherwise APLS in [0, 1].
    """
    if G_gt.number_of_nodes() == 0:
        return None
    if G_pr.number_of_nodes() == 0:
        return 0.0

    # Simplify graphs by collapsing degree-2 chains.
    G_gt_s = simplify_graph(G_gt)
    G_pr_s = simplify_graph(G_pr)

    # Optionally subdivide long edges to insert control nodes.
    if SUBDIVIDE_LONG_EDGES:
        G_gt_s = subdivide_edges_for_control_nodes(G_gt_s, step=SAMPLE_STEP)
        G_pr_s = subdivide_edges_for_control_nodes(G_pr_s, step=SAMPLE_STEP)

    # Build control points.
    CP_gt = build_control_points(G_gt_s, step=SAMPLE_STEP)
    CP_pr = build_control_points(G_pr_s, step=SAMPLE_STEP)

    # Map control points symmetrically to the other graph.
    pairs_gt_to_pr, pairs_pr_to_gt = symmetric_node_injection(G_gt_s, G_pr_s, CP_gt, CP_pr, eps=SNAP_EPS)

    # Direction 1: GT control points.
    S_gt = [a for a, _ in pairs_gt_to_pr]
    S_gt_mapped = {a: a2 for a, a2 in pairs_gt_to_pr}

    d_gt = all_pairs_from_sources(G_gt_s, S_gt)

    S_pr_from_gt = [S_gt_mapped[a] for a in S_gt]
    d_pr = all_pairs_from_sources(G_pr_s, S_pr_from_gt)

    total_err = 0.0
    n_pairs = 0
    
    # Align control-point pairs by index order: (a, b) <-> (a', b').
    S_gt_list = list(S_gt)
    for i in range(len(S_gt_list)):
        for j in range(i + 1, len(S_gt_list)):
            a, b = S_gt_list[i], S_gt_list[j]
            a2, b2 = S_gt_mapped[a], S_gt_mapped[b]
            key1 = (a, b) if a < b else (b, a)
            key2 = (a2, b2) if a2 < b2 else (b2, a2)
            L_gt = d_gt.get(key1, float('inf'))
            L_pr = d_pr.get(key2, float('inf'))
            if np.isinf(L_gt) and np.isinf(L_pr):
                err = 0.0
            elif np.isinf(L_gt) and not np.isinf(L_pr):
                err = 1.0
            elif np.isinf(L_pr) and not np.isinf(L_gt):
                err = 1.0
            elif L_gt == 0:
                err = 0.0 if L_pr == 0 else 1.0
            else:
                err = min(1.0, abs(L_gt - L_pr) / max(L_gt, 1e-6))
            total_err += err
            n_pairs += 1

    # Direction 2: prediction control points.
    S_pr = [b for b, _ in pairs_pr_to_gt]
    S_pr_mapped = {b: b2 for b, b2 in pairs_pr_to_gt}

    d_pr2 = all_pairs_from_sources(G_pr_s, S_pr)
    S_gt_from_pr = [S_pr_mapped[b] for b in S_pr]
    d_gt2 = all_pairs_from_sources(G_gt_s, S_gt_from_pr)

    for i in range(len(S_pr)):
        for j in range(i + 1, len(S_pr)):
            a, b = S_pr[i], S_pr[j]
            a2, b2 = S_pr_mapped[a], S_pr_mapped[b]

            key1 = (a, b) if a < b else (b, a)
            key2 = (a2, b2) if a2 < b2 else (b2, a2)
            
            L_pr = d_pr2.get(key1, float('inf'))
            L_gt = d_gt2.get(key2, float('inf'))
            
            if np.isinf(L_gt) and np.isinf(L_pr):
                err = 0.0
            elif np.isinf(L_gt) and not np.isinf(L_pr):
                err = 1.0
            elif np.isinf(L_pr) and not np.isinf(L_gt):
                err = 1.0
            elif L_gt == 0:
                err = 0.0 if L_pr == 0 else 1.0
            else:
                err = min(1.0, abs(L_gt - L_pr) / max(L_gt, 1e-6))
                
            total_err += err
            n_pairs += 1

    apls = 1.0 - (total_err / n_pairs if n_pairs > 0 else 1.0)
    return apls


def visualize_graph_on_image(image, graph, color=(255, 0, 0)):
    """
    Overlay a graph on an RGB image for quick qualitative inspection.
    """
    vis = image.copy()
    for u, v in graph.edges():
        pt1 = (int(round(u[1])), int(round(u[0])))
        pt2 = (int(round(v[1])), int(round(v[0])))
        cv2.line(vis, pt1, pt2, color, VIS_THICK)
    return vis


def process_gt_mask_folder(gt_file, image_folder, output_folder, save_mask_folder=None):
    """
    Precompute GT skeleton pixel graphs from COCO polygon annotations.

    For each image:
      1) Rasterize instance polygons into a binary mask (exterior minus holes).
      2) Skeletonize the mask.
      3) Convert skeleton pixels into a weighted pixel graph (8-neighborhood).
      4) Save the pixel graph as .gpickle for later evaluation.

    Notes:
      - We store the *pixel* graph directly; simplification happens during evaluation.
      - Visualization uses a simplified graph only for readability.

    Args:
        gt_file (str): COCO GT JSON (per class).
        image_folder (str): Folder containing the corresponding images.
        output_folder (str): Folder to store GT .gpickle graphs (+ visualizations).
        save_mask_folder (str or None): Optional folder to dump rasterized masks.
    """
    os.makedirs(output_folder, exist_ok=True)
    vis_output_folder = os.path.join(output_folder, "visualizations")
    os.makedirs(vis_output_folder, exist_ok=True)

    if save_mask_folder is not None:
        os.makedirs(save_mask_folder, exist_ok=True)

    with open(gt_file, 'r') as f:
        gt = json.load(f)

    image_id_to_name = {img['id']: img['file_name'] for img in gt['images']}

    # Group polygons by image_id for faster rasterization.
    polygons_by_image = defaultdict(list)
    for ann in gt['annotations']:
        polygons_by_image[ann['image_id']].append(ann['segmentation'])

    for image_id, file_name in tqdm(image_id_to_name.items(), desc="Processing GT"):
        image_path = os.path.join(image_folder, file_name)
        img = cv2.imread(image_path)
        if img is None:
            print(f"[GT] read fail: {image_path}")
            continue

        H, W = img.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)

        # Rasterize all instances into a single binary mask.
        for segmentation in polygons_by_image[image_id]:
            for i, poly in enumerate(segmentation):
                poly_coords = np.array(poly, dtype=np.int32).reshape((-1, 2))
                if i == 0:
                    cv2.fillPoly(mask, [poly_coords], 1)
                else:
                    cv2.fillPoly(mask, [poly_coords], 0)

        if save_mask_folder is not None:
            cv2.imwrite(os.path.join(save_mask_folder, file_name), mask * 255)

        sk = extract_skeleton(mask)
        Gpix = skeleton_to_pixel_graph(sk)

        out_path = os.path.join(output_folder, file_name.rsplit('.', 1)[0] + ".gpickle")
        with open(out_path, "wb") as f:
            pickle.dump(Gpix, f)

        vis = visualize_graph_on_image(img, simplify_graph(Gpix), color=(0, 255, 255))
        cv2.imwrite(os.path.join(vis_output_folder, file_name), vis)


def process_mask_folder(dt_file, gt_file, image_folder, output_folder):
    """
    Precompute prediction skeleton pixel graphs from COCO prediction polygons.

    This mirrors GT preprocessing, but uses `gt_file` only to obtain
    a stable image_id → file_name mapping so we can iterate over the same image set.

    For each image:
      1) Rasterize prediction polygons into a binary mask.
      2) Skeletonize the mask.
      3) Convert skeleton to weighted pixel graph.
      4) Save as .gpickle.

    Args:
        dt_file (str): COCO prediction JSON (per class).
        gt_file (str): COCO GT JSON, used for image id mapping.
        image_folder (str): Folder containing images.
        output_folder (str): Folder to store prediction graphs (+ visualizations).
    """
    os.makedirs(output_folder, exist_ok=True)
    vis_output_folder = os.path.join(output_folder, "visualizations")
    os.makedirs(vis_output_folder, exist_ok=True)

    image_id_mapping = load_image_id_mapping(gt_file)

    for image_id, file_name in tqdm(image_id_mapping.items(), desc="Processing Pred"):
        image_path = os.path.join(image_folder, file_name)
        img = cv2.imread(image_path)
        if img is None:
            print(f"[Pred] read fail: {image_path}")
            continue

        mask = extract_mask_from_coco(dt_file, image_id, img.shape)
        sk = extract_skeleton(mask)
        Gpix = skeleton_to_pixel_graph(sk)

        out_path = os.path.join(output_folder, file_name.rsplit('.', 1)[0] + ".gpickle")
        with open(out_path, "wb") as f:
            pickle.dump(Gpix, f)

        vis = visualize_graph_on_image(img, simplify_graph(Gpix), color=(0, 255, 0))
        cv2.imwrite(os.path.join(vis_output_folder, file_name), vis)


def compute_apls_for_folders(gt_folder, pred_folder):
    """
    Compute dataset-level APLS from precomputed GT/Pred pixel graphs.

    Evaluation is performed per image graph file:
      - If the prediction graph is missing: skip (and report).
      - If the GT graph has no nodes: skip (empty GT).
      - Otherwise compute symmetric APLS.

    Args:
        gt_folder (str): Folder containing GT .gpickle graphs.
        pred_folder (str): Folder containing Pred .gpickle graphs.

    Returns:
        avg (float): Average APLS over valid images.
        per_image (dict): Per-image APLS values (key = filename.gpickle).
        n_valid (int): Number of valid images contributing to the average.
        n_total (int): Total number of GT graph files found.
    """
    graph_files = [f for f in os.listdir(gt_folder) if f.endswith(".gpickle")]

    total = 0.0
    n_valid = 0
    per_image = {}

    for gf in tqdm(graph_files, desc="APLS"):
        gt_path = os.path.join(gt_folder, gf)
        pr_path = os.path.join(pred_folder, gf)

        if not os.path.exists(pr_path):
            print(f"missing pred: {gf}")
            continue

        with open(gt_path, "rb") as f:
            G_gt_pix = pickle.load(f)
        with open(pr_path, "rb") as f:
            G_pr_pix = pickle.load(f)

        if G_gt_pix.number_of_nodes() == 0:
            print(f"skip {gf}: empty GT")
            continue

        apls = compute_apls_symmetric(G_gt_pix, G_pr_pix)
        if apls is None:
            continue

        per_image[gf] = float(apls)
        total += apls
        n_valid += 1

    avg = total / n_valid if n_valid > 0 else 0.0
    return avg, per_image, n_valid, len(graph_files)

def parse_args():
    """Parse command-line arguments for the APLS/PSLG evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate skeleton graphs from COCO polygons and evaluate APLS (symmetric, fast)."
    )

    # Prediction and GT COCO JSONs.
    parser.add_argument(
        "--dt_file", type=str, required=True,
        help="COCO-style prediction JSON (per-class)."
    )
    parser.add_argument(
        "--gt_file", type=str, required=True,
        help="COCO-style GT JSON (per-class)."
    )

    # Images are used for shape lookup and visualization overlays.
    parser.add_argument(
        "--image_folder", type=str, required=True,
        help="Folder containing images."
    )

    # Graph cache folders.
    parser.add_argument(
        "--gt_folder", type=str, required=True,
        help="Folder to dump/load GT skeleton graphs (*.gpickle)."
    )
    parser.add_argument(
        "--output_folder",
        type=str, required=True,
        help="Folder to dump/load predicted skeleton graphs and results."
    )

    return parser.parse_args()


def is_folder_empty(folder):
    """
    Check whether a folder exists and is empty.

    Returns:
        bool: True if the folder is missing or empty, else False.
    """
    if not os.path.exists(folder):
        return True
    return len(os.listdir(folder)) == 0


def main(args):
    """
    Run the full pipeline:
      1) Build predicted skeleton graphs from COCO polygons.
      2) Compute symmetric APLS using cached GT graphs and predicted graphs.
      3) Save per-image APLS results to JSON.
    """
    os.makedirs(args.output_folder, exist_ok=True)

    # Step 1: build predicted pixel-skeleton graphs.
    process_mask_folder(
        dt_file=args.dt_file,
        gt_file=args.gt_file,
        image_folder=args.image_folder,
        output_folder=args.output_folder
    )

    # Step 2: run batch APLS evaluation.
    avg, per_image, n_valid, n_total = compute_apls_for_folders(
        gt_folder=args.gt_folder,
        pred_folder=args.output_folder
    )

    # Step 3: report and save results.
    print("\n=== Final Results ===")
    print(f"Valid APLS computations: {n_valid}/{n_total}")
    print(f"Average APLS: {avg * 100:.2f}")

    out_json = os.path.join(args.output_folder, "apls_results.json")
    with open(out_json, "w") as f:
        json.dump(per_image, f, indent=2)
    print(f"Saved per-image APLS to: {out_json}")


if __name__ == "__main__":
    args = parse_args()

    # Step 0: prepare cached GT skeleton graphs.
    if is_folder_empty(args.gt_folder):
        print(f"[INFO] GT folder '{args.gt_folder}' is empty. Building GT skeleton graphs...")
        os.makedirs(args.gt_folder, exist_ok=True)

        process_gt_mask_folder(
            gt_file=args.gt_file,
            image_folder=args.image_folder,
            output_folder=args.gt_folder
        )
    else:
        print(f"[INFO] GT folder '{args.gt_folder}' already exists and is not empty. Skipping GT preprocessing.")

    # Step 1-3: build prediction graphs and run evaluation.
    main(args)
