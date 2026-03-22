import os
import cv2
import json
import argparse
import numpy as np
from polygonize_pslg_one_image import (
    process_one_image_categories,
    process_one_image_categories_opencv_fast,
    make_canvas,
    flatten_to_xylist, draw_filled_polygon_with_holes_safe
)
from shapely.geometry import Polygon
from tqdm import tqdm
import time


ID2NAME = {
    0: "building",
    1: "road_bridge",
    2: "unvegetated",
    3: "vegetation",
    4: "water",
}


CATEGORY_COLORS_BGR = {
    0: (42, 42, 165),
    1: (0, 255, 255),
    2: (128, 128, 128),
    3: (0, 255, 0),
    4: (255, 0, 0),
}
EDGE_COLOR  = (0, 0, 0)
POINT_COLOR = (255, 0, 255)


def _bbox_from_xy(ext_xy):
    xs = [p[0] for p in ext_xy]; ys = [p[1] for p in ext_xy]
    x0, y0 = float(min(xs)), float(min(ys))
    x1, y1 = float(max(xs)), float(max(ys))
    return [x0, y0, x1 - x0, y1 - y0]


def area_bbox_with_holes(ext_xy, holes_xy):
    """
    ext_xy: [(x,y), ...] 外环
    holes_xy: [ [(x,y),...], ... ] 多个孔洞
    返回 (area, bbox)，bbox 按外环取 [x_min,y_min,w,h]
    """
    # bbox 按外环（按你要求）
    bbox = _bbox_from_xy(ext_xy) if len(ext_xy) >= 3 else [0.0, 0.0, 0.0, 0.0]

    holes_valid = [h for h in holes_xy if len(h) >= 3]
    poly = Polygon(ext_xy, holes=holes_valid)
    if not poly.is_valid:
        # 经典修复，处理自交/重复点/微缝
        poly = poly.buffer(0)
    return float(max(poly.area, 0.0)), [float(b) for b in bbox]


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("1","true","t","yes","y")


def _flatten_xy(cnt):
        """将 Nx1x2 或 Nx2 的 contour 转为 [x1,y1,x2,y2,...]（float）"""
        c = np.squeeze(cnt, axis=1) if cnt.ndim == 3 else cnt
        if c.ndim != 2 or c.shape[1] != 2:
            return []
        return [float(v) for xy in c for v in xy]

def _approx(cnt, eps):
    """对 contour 做 DP 简化（eps 像素），确保至少 3 点"""
    if cnt is None or len(cnt) == 0:
        return None
    approx = cv2.approxPolyDP(cnt, epsilon=eps, closed=True)
    # 去除重复点导致的退化
    if approx is None or len(approx) < 3:
        return None
    return approx


# ablation: no pslg + dp
def polygonize_per_class_no_pslg_dp(
    seg_path: str,
    categories,
    vis_dir: str = None,
    vis_scale: int = 1,
    dp_epsilon: float = 2.5,
):
    """
    No-PSLG baseline:
      - 对每个类别，直接在该类的二值 mask 上用 cv2.findContours 提取边界
      - 用 Douglas–Peucker (cv2.approxPolyDP) 做像素空间简化
      - 将 (外环 + 若干孔洞) 组织成 segmentation: [ext_flat, hole1_flat, ...]
      - 返回结构与 process_one_image_categories_v5 一致: {cat_id: [segmentation_instance, ...], ...}

    注意：
      - junction_json / dist_thresh / corner_eps / vss 等参数在此基线中不使用，但保留以兼容你的原调用。
      - dp_epsilon 以 “像素” 为单位传给 approxPolyDP。
    """
    # 读取多类 mask
    L = np.load(seg_path).astype(np.int32)  # (H, W)
    H, W = L.shape

    # 可选可视化目录
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    # 统一输出结构：每类一个 list，list 内每个实例是 segmentation（外环+孔洞）
    results = {int(c): [] for c in categories}

    # OpenCV 版本差异：RETR_CCOMP 可以给出两层层级（外环与其直接子孔洞）
    # 如需更复杂嵌套，可改用 RETR_TREE; 这里按常见场景够用。
    RETR_MODE = cv2.RETR_TREE
    CHAIN_MODE = cv2.CHAIN_APPROX_NONE  # 保留点以便 DP 简化

    for cat in categories:
        cat = int(cat)
        # 二值化该类
        mask_bin = (L == cat).astype(np.uint8)  # 0/1
        if mask_bin.max() == 0:
            continue

        # 找轮廓和层级
        contours, hierarchy = cv2.findContours(mask_bin, RETR_MODE, CHAIN_MODE)
        if not contours or hierarchy is None:
            continue
        hierarchy = hierarchy[0]  # (N,4): [next, prev, first_child, parent]

        # 遍历所有外环（parent == -1）
        for i, h in enumerate(hierarchy):
            parent = h[3]
            if parent != -1:
                continue  # 只处理外环

            ext = contours[i]
            if ext is None or len(ext) < 3:
                continue

            # DP 简化外环
            ext_dp = _approx(ext, dp_epsilon)
            if ext_dp is None:
                continue

            # 收集直接子孔洞（parent == i）
            holes_dp = []
            child = h[2]
            while child != -1:
                hole = contours[child]
                if hole is not None and len(hole) >= 3:
                    hole_dp = _approx(hole, dp_epsilon)
                    if hole_dp is not None:
                        holes_dp.append(hole_dp)
                child = hierarchy[child][0]  # 下一个兄弟

            # 组织 segmentation（外环 + 孔洞），均为 float 扁平化
            seg = []
            seg.append(_flatten_xy(ext_dp))
            for hdp in holes_dp:
                seg.append(_flatten_xy(hdp))

            # 过滤非法/空分段
            if len(seg[0]) < 6:  # 至少 3 点
                continue

            results[cat].append(seg)

        # 可选：简单可视化（类别级别），只画轮廓，避免引入你项目的绘图依赖
        if vis_dir:
            vis = np.zeros((H * vis_scale, W * vis_scale, 3), dtype=np.uint8)
            color = (0, 255, 255)  # 任意可见色
            for seg in results[cat]:
                # 外环
                ext = np.array(seg[0], dtype=np.float32).reshape(-1, 2)
                ext_i = np.round(ext * vis_scale).astype(np.int32)
                cv2.polylines(vis, [ext_i], isClosed=True, color=color, thickness=1)
                # 孔洞
                for hole in seg[1:]:
                    h = np.array(hole, dtype=np.float32).reshape(-1, 2)
                    h_i = np.round(h * vis_scale).astype(np.int32)
                    cv2.polylines(vis, [h_i], isClosed=True, color=(0, 128, 255), thickness=1)

            cv2.imwrite(os.path.join(vis_dir, f"cat_{cat}.png"), vis)

    return results


def polygonize_per_class_no_pslg_dp_no_cv2(
    seg_path: str,
    categories,
    vis_dir: str = None,
    vis_scale: int = 1,
    dp_epsilon: float = 2.5,
):
    """
    No-PSLG baseline (NO OpenCV contour extraction, NO OpenCV DP):
      - per class: build boundary half-edges on grid (inside on LEFT), trace rings by left-hand rule
      - simplify each ring with pure-python RDP (closed)
      - group rings into (exterior + holes) by containment
      - output: {cat_id: [ [ext_flat, hole1_flat, ...], ... ], ... }

    Notes:
      - This is a *Python* contour extractor; will be slower than cv2.findContours (C++).
      - But compared with PSLG+shapely polygonize/unary_union, it is often closer and avoids huge object creation.
    """
    L = np.load(seg_path).astype(np.int32)
    H, W = L.shape

    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    results = {int(c): [] for c in categories}

    # -----------------------------
    # key helper 1: point-in-poly
    # -----------------------------
    def point_in_poly(px: float, py: float, ring_xy: np.ndarray) -> bool:
        # ray casting; ring_xy (N,2) not closed
        x = ring_xy[:, 0].astype(np.float64)
        y = ring_xy[:, 1].astype(np.float64)
        inside = False
        j = len(ring_xy) - 1
        for i in range(len(ring_xy)):
            xi, yi = x[i], y[i]
            xj, yj = x[j], y[j]
            cond = ((yi > py) != (yj > py))
            if cond:
                x_int = (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi
                if px < x_int:
                    inside = not inside
            j = i
        return inside

    # -----------------------------
    # key helper 2: RDP for closed ring
    # -----------------------------
    def rdp_closed(ring_xy: np.ndarray, eps: float):
        """
        ring_xy: (N,2) int32, not necessarily closed
        return simplified ring (M,2) int32, not closed
        """
        if ring_xy is None or len(ring_xy) < 3:
            return None

        # remove closure dup if exists
        if np.all(ring_xy[0] == ring_xy[-1]):
            ring_xy = ring_xy[:-1]
        if len(ring_xy) < 3:
            return None

        eps2 = float(eps) ** 2

        # choose a stable cut: farthest from centroid
        cen = ring_xy.mean(axis=0)
        cut = int(np.argmax(np.sum((ring_xy - cen) ** 2, axis=1)))

        rot = np.roll(ring_xy, -cut, axis=0)
        open_xy = np.vstack([rot, rot[0:1]])  # make open

        # iterative RDP on open polyline
        n = len(open_xy)
        keep = np.zeros(n, dtype=bool)
        keep[0] = True
        keep[-1] = True
        stack = [(0, n - 1)]

        while stack:
            i0, i1 = stack.pop()
            a = open_xy[i0].astype(np.float64)
            b = open_xy[i1].astype(np.float64)
            ab = b - a
            ab2 = float(ab[0] * ab[0] + ab[1] * ab[1]) + 1e-12

            max_d2 = -1.0
            max_i = -1
            for i in range(i0 + 1, i1):
                p = open_xy[i].astype(np.float64)
                ap = p - a
                t = float((ap[0] * ab[0] + ap[1] * ab[1]) / ab2)
                if t < 0.0:
                    proj = a
                elif t > 1.0:
                    proj = b
                else:
                    proj = a + t * ab
                dx = p[0] - proj[0]
                dy = p[1] - proj[1]
                d2 = dx * dx + dy * dy
                if d2 > max_d2:
                    max_d2 = d2
                    max_i = i

            if max_d2 > eps2 and max_i != -1:
                keep[max_i] = True
                stack.append((i0, max_i))
                stack.append((max_i, i1))

        simp = open_xy[np.flatnonzero(keep)]
        if len(simp) >= 2 and np.all(simp[0] == simp[-1]):
            simp = simp[:-1]
        if len(simp) < 3:
            return None

        # rotate back
        simp_back = np.roll(simp, cut, axis=0)

        # consecutive dedup
        out = [simp_back[0]]
        for p in simp_back[1:]:
            if not np.all(p == out[-1]):
                out.append(p)
        out = np.array(out, dtype=np.int32)

        return out if len(out) >= 3 else None

    # -----------------------------
    # per category processing
    # -----------------------------
    for cat in categories:
        cat = int(cat)
        mask = (L == cat).astype(np.uint8)
        if mask.max() == 0:
            continue

        # Build directed half-edges on (H+1)x(W+1) lattice:
        # key = u*4 + dir, where dir: 0=R,1=D,2=L,3=U
        V = (H + 1) * (W + 1)
        edge_to = np.full(V * 4, -1, dtype=np.int32)
        visited = np.zeros(V * 4, dtype=bool)

        def vid(x, y):
            return y * (W + 1) + x

        # left-hand priority based on incoming direction
        left_pref = {
            0: (3, 0, 1, 2),
            1: (0, 1, 2, 3),
            2: (1, 2, 3, 0),
            3: (2, 3, 0, 1),
        }

        # ---- internal vertical boundaries ----
        left = mask[:, :-1]
        right = mask[:, 1:]
        diff_v = (left != right)
        ys, x0s = np.nonzero(diff_v)
        xs = x0s + 1  # boundary x

        # down edge (x,y)->(x,y+1) keeps inside on LEFT if right==1 and left==0
        sel = (right[ys, x0s] == 1) & (left[ys, x0s] == 0)
        if np.any(sel):
            y = ys[sel].astype(np.int32)
            x = xs[sel].astype(np.int32)
            u = y * (W + 1) + x
            v = (y + 1) * (W + 1) + x
            edge_to[u * 4 + 1] = v

        # up edge (x,y+1)->(x,y) keeps inside on LEFT if left==1 and right==0
        sel = (left[ys, x0s] == 1) & (right[ys, x0s] == 0)
        if np.any(sel):
            y = ys[sel].astype(np.int32)
            x = xs[sel].astype(np.int32)
            u = (y + 1) * (W + 1) + x
            v = y * (W + 1) + x
            edge_to[u * 4 + 3] = v

        # ---- internal horizontal boundaries ----
        top = mask[:-1, :]
        bot = mask[1:, :]
        diff_h = (top != bot)
        y0s, xs2 = np.nonzero(diff_h)
        ys2 = y0s + 1  # boundary y

        # right edge (x,y)->(x+1,y) keeps inside on LEFT if top==1 and bot==0
        sel = (top[y0s, xs2] == 1) & (bot[y0s, xs2] == 0)
        if np.any(sel):
            y = ys2[sel].astype(np.int32)
            x = xs2[sel].astype(np.int32)
            u = y * (W + 1) + x
            v = y * (W + 1) + (x + 1)
            edge_to[u * 4 + 0] = v

        # left edge (x+1,y)->(x,y) keeps inside on LEFT if bot==1 and top==0
        sel = (bot[y0s, xs2] == 1) & (top[y0s, xs2] == 0)
        if np.any(sel):
            y = ys2[sel].astype(np.int32)
            x = xs2[sel].astype(np.int32)
            u = y * (W + 1) + (x + 1)
            v = y * (W + 1) + x
            edge_to[u * 4 + 2] = v

        # ---- frame boundaries (outside treated as 0) ----
        # top border y=0: left edge (x+1,0)->(x,0) if mask[0,x]==1
        xs = np.flatnonzero(mask[0, :] == 1).astype(np.int32)
        if xs.size:
            u = 0 * (W + 1) + (xs + 1)
            v = 0 * (W + 1) + xs
            edge_to[u * 4 + 2] = v

        # bottom border y=H: right edge (x,H)->(x+1,H) if mask[H-1,x]==1
        xs = np.flatnonzero(mask[H - 1, :] == 1).astype(np.int32)
        if xs.size:
            u = H * (W + 1) + xs
            v = H * (W + 1) + (xs + 1)
            edge_to[u * 4 + 0] = v

        # left border x=0: down edge (0,y)->(0,y+1) if mask[y,0]==1
        ys = np.flatnonzero(mask[:, 0] == 1).astype(np.int32)
        if ys.size:
            u = ys * (W + 1) + 0
            v = (ys + 1) * (W + 1) + 0
            edge_to[u * 4 + 1] = v

        # right border x=W: up edge (W,y+1)->(W,y) if mask[y,W-1]==1
        ys = np.flatnonzero(mask[:, W - 1] == 1).astype(np.int32)
        if ys.size:
            u = (ys + 1) * (W + 1) + W
            v = ys * (W + 1) + W
            edge_to[u * 4 + 3] = v

        # ---- trace rings ----
        rings = []

        keys = np.flatnonzero(edge_to >= 0)
        for start_key in keys:
            start_key = int(start_key)
            if visited[start_key]:
                continue

            visited[start_key] = True
            u = start_key // 4
            d_in = start_key % 4
            v = int(edge_to[start_key])

            # start point u
            x0 = u % (W + 1)
            y0 = u // (W + 1)
            ring = [(int(x0), int(y0))]

            # add v
            x1 = v % (W + 1)
            y1 = v // (W + 1)
            ring.append((int(x1), int(y1)))

            cur_v = v
            cur_d = d_in  # direction used to reach cur_v

            guard = 0
            while True:
                guard += 1
                if guard > (V * 4 + 10):
                    break

                base_u = cur_v
                next_key = None
                next_d = None

                # choose next by left-hand preference
                for nd in left_pref[cur_d]:
                    k = base_u * 4 + nd
                    if edge_to[k] >= 0 and not visited[k]:
                        next_key = k
                        next_d = nd
                        break

                # if no unvisited, allow reuse to close
                if next_key is None:
                    for nd in left_pref[cur_d]:
                        k = base_u * 4 + nd
                        if edge_to[k] >= 0:
                            next_key = k
                            next_d = nd
                            break
                    if next_key is None:
                        break

                if int(next_key) == start_key:
                    break

                visited[int(next_key)] = True
                nxt_v = int(edge_to[int(next_key)])

                xx = nxt_v % (W + 1)
                yy = nxt_v // (W + 1)
                ring.append((int(xx), int(yy)))

                cur_v = nxt_v
                cur_d = int(next_d)

            ring_xy = np.array(ring, dtype=np.int32)
            # drop closure dup if any
            if len(ring_xy) >= 2 and np.all(ring_xy[0] == ring_xy[-1]):
                ring_xy = ring_xy[:-1]
            if len(ring_xy) >= 3:
                rings.append(ring_xy)

        # simplify rings by DP
        simp_rings = []
        for r in rings:
            rdp = rdp_closed(r, dp_epsilon)
            if rdp is not None and len(rdp) >= 3:
                simp_rings.append(rdp)

        if not simp_rings:
            continue

        # group rings into exteriors + holes by containment
        # sort by abs area desc
        def abs_area(r):
            x = r[:, 0].astype(np.float64)
            y = r[:, 1].astype(np.float64)
            return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

        simp_rings.sort(key=abs_area, reverse=True)

        exteriors = []
        holes = []
        for r in simp_rings:
            px = float(np.mean(r[:, 0]))
            py = float(np.mean(r[:, 1]))
            inside_any = False
            for ext in exteriors:
                if point_in_poly(px, py, ext["exterior"]):
                    inside_any = True
                    break
            if not inside_any:
                exteriors.append({"exterior": r, "interiors": []})
            else:
                holes.append(r)

        # assign each hole to smallest containing exterior
        for h in holes:
            px = float(np.mean(h[:, 0]))
            py = float(np.mean(h[:, 1]))
            candidates = []
            for ext in exteriors:
                if point_in_poly(px, py, ext["exterior"]):
                    candidates.append((abs_area(ext["exterior"]), ext))
            if candidates:
                candidates.sort(key=lambda t: t[0])
                candidates[0][1]["interiors"].append(h)

        # finalize output format: segmentation instance = [ext_flat, hole1_flat, ...]
        for inst in exteriors:
            ext = inst["exterior"]
            if ext is None or len(ext) < 3:
                continue
            seg = [_flatten_xy(ext)]
            if len(seg[0]) < 6:
                continue
            for h in inst["interiors"]:
                if h is None or len(h) < 3:
                    continue
                flat = _flatten_xy(h)
                if len(flat) >= 6:
                    seg.append(flat)
            results[cat].append(seg)

        # optional visualization
        if vis_dir:
            vis = np.zeros((H * vis_scale, W * vis_scale, 3), dtype=np.uint8)
            for seg in results[cat]:
                ext = np.array(seg[0], dtype=np.float32).reshape(-1, 2)
                ext_i = np.round(ext * vis_scale).astype(np.int32)
                cv2.polylines(vis, [ext_i], isClosed=True, color=(0, 255, 255), thickness=1)
                for hole in seg[1:]:
                    h = np.array(hole, dtype=np.float32).reshape(-1, 2)
                    h_i = np.round(h * vis_scale).astype(np.int32)
                    cv2.polylines(vis, [h_i], isClosed=True, color=(0, 128, 255), thickness=1)
            cv2.imwrite(os.path.join(vis_dir, f"cat_{cat}.png"), vis)

    return results


def instance_score_mean_prob(prob_map, cat_id, ext_xy, holes_xy, H, W):
    """
    prob_map: (C, H, W) float32 in [0,1]
    cat_id: int
    ext_xy: [(x,y),...]
    holes_xy: list of [(x,y),...]
    """
    if prob_map is None:
        return 1.0
    if prob_map.ndim != 3 or cat_id < 0 or cat_id >= prob_map.shape[0]:
        return 1.0

    p = prob_map[cat_id]  # (H, W)

    # exterior mask
    mask_ext = np.zeros((H, W), dtype=np.uint8)
    poly_ext = np.array(ext_xy, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask_ext, [poly_ext], 1)

    # holes mask
    if holes_xy:
        mask_hole = np.zeros((H, W), dtype=np.uint8)
        for h in holes_xy:
            if len(h) < 3:
                continue
            poly_h = np.array(h, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask_hole, [poly_h], 1)
        mask = (mask_ext == 1) & (mask_hole == 0)
    else:
        mask = (mask_ext == 1)

    if not np.any(mask):
        return 0.0

    return float(np.mean(p[mask]))


def main():
    parser = argparse.ArgumentParser(
        description="Batch polygonization: ours (PSLG+VSS) or ablation (no PSLG, pure DP per region)."
    )

    # ------------------------------------------------------------------
    # Required I/O
    # ------------------------------------------------------------------
    parser.add_argument("--seg_dir", required=True,
                        help="Directory of predicted label maps (.npy).")
    parser.add_argument("--junction_dir", required=True,
                        help="Directory of predicted vertices JSONs: [[x,y], ...] (matched by basename).")
    parser.add_argument("--categories", type=int, nargs="+", required=True,
                        help="Target category IDs, e.g. --categories 0 1 2")
    parser.add_argument("--out_dir", required=True,
                        help="Directory to save per-image polygonization results (.json).")
    parser.add_argument("--cats_out_dir", required=True,
                        help="Directory to save per-category COCO prediction JSONs.")
    parser.add_argument("--overview_dir", required=True,
                        help="Directory to save per-image overview visualization PNGs.")
    parser.add_argument("--prob_dir", default=None,
                    help="Directory of predicted probability maps (.npy), shape (C,H,W). Matched by basename.")


    # ------------------------------------------------------------------
    # Run mode: ours vs ablation
    # ------------------------------------------------------------------
    parser.add_argument("--mode", type=str, default="ours",
                        choices=["ours", "ablation_dp"],
                        help=("Run mode: "
                              "'ours' = PSLG + VSS (+ optional DP-fix); "
                              "'ablation_dp' = no PSLG, pure DP per semantic region."))

    # ------------------------------------------------------------------
    # Optional visualization
    # ------------------------------------------------------------------
    parser.add_argument("--vis_dir", default=None,
                        help="Optional root dir for per-image step visualizations: <vis_dir>/<name>/...")
    parser.add_argument("--vis_scale", type=int, default=1,
                        help="Visualization scale factor (default: 1).")

    # ------------------------------------------------------------------
    # Core geometric hyper-parameters (ours-mode)
    # ------------------------------------------------------------------
    parser.add_argument("--dist_thresh", type=float, default=5.0,
                        help="Snap radius for predicted vertices to boundary polylines (pixels).")
    parser.add_argument("--corner_eps", type=float, default=2.0,
                        help="Radius to treat a point as too close to a structural corner (pixels).")

    # ------------------------------------------------------------------
    # Point selection strategy (ours-mode)
    # ------------------------------------------------------------------
    parser.add_argument("--vss", action="store_true",
                        help="Enable vertex-guided subset selection (VSS). "
                             "If not set, pure DP simplification is used.")
    parser.add_argument("--dp_fix", action="store_true",
                        help="Enable DP-based safeguard (effective only when --vss is set).")
    parser.add_argument("--dp_epsilon_dp", type=float, default=1.0,
                        help="RDP epsilon for pure DP fallback (used when vss is OFF in ours-mode)."
    )
    parser.add_argument("--dp_epsilon_fix", type=float, default=2.5,
                        help="RDP epsilon for DP safeguard in VSS mode (used when dp_fix is ON).")

    # ------------------------------------------------------------------
    # Ablation hyper-parameters (ablation_dp-mode)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--ablation_dp_epsilon", type=float, default=2.5,
        help="RDP epsilon for ablation mode (no PSLG, pure DP per region)."
    )

    args = parser.parse_args()

    # Prepare output dirs
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.overview_dir, exist_ok=True)
    os.makedirs(args.cats_out_dir, exist_ok=True)
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)

    # COCO predictions per category
    per_cat_preds = {int(c): [] for c in args.categories}

    # ------------------------------------------------------------------
    # Iterate inputs
    # ------------------------------------------------------------------
    seg_files = sorted([f for f in os.listdir(args.seg_dir) if f.endswith('.npy')])

    total_time = 0.0
    num_done = 0

    for f in tqdm(seg_files, desc=f"Polygonizing ({args.mode})"):
        name = os.path.splitext(f)[0]
        seg_path  = os.path.join(args.seg_dir, f)
        junc_path = os.path.join(args.junction_dir, name + '.json')

        prob_map = None
        if args.prob_dir is not None:
            prob_path = os.path.join(args.prob_dir, name + ".npy")
            if os.path.exists(prob_path):
                prob_map = np.load(prob_path).astype(np.float32)

        # If junction is missing:
        # - ours-mode: skip
        # - ablation: can run without junctions
        if args.mode == "ours" and (not os.path.exists(junc_path)):
            continue

        vis_subdir = os.path.join(args.vis_dir, name) if args.vis_dir else None

        t0 = time.perf_counter()

        # --------------------------------------------------------------
        # Run polygonization
        # --------------------------------------------------------------
        if args.mode == "ours":
            results = process_one_image_categories_opencv_fast(
                seg_path=seg_path,
                junction_json=junc_path,
                categories=args.categories,
                dist_thresh=args.dist_thresh,
                corner_eps=args.corner_eps,
                vis_dir=vis_subdir,
                vis_scale=args.vis_scale,
                vss=args.vss,
                dp_fix=args.dp_fix,
                dp_epsilon_dp=args.dp_epsilon_dp,
                dp_epsilon_fix=args.dp_epsilon_fix,
            )
        else:
            results = polygonize_per_class_no_pslg_dp(
                seg_path=seg_path,
                categories=args.categories,
                vis_dir=vis_subdir,
                vis_scale=args.vis_scale,
                dp_epsilon=args.ablation_dp_epsilon,
            )

        t1 = time.perf_counter()
        dt = t1 - t0
        total_time += dt
        num_done += 1

        if num_done % 50 == 0:
            fps = num_done / total_time
            print(f"[FPS] {fps:.2f} img/s | avg {total_time/num_done:.3f} s/img")

        # Save per-image json
        out_json_path = os.path.join(args.out_dir, name + '.json')
        with open(out_json_path, 'w') as fp:
            json.dump(results, fp)

        # --------------------------------------------------------------
        # Overview visualization (all categories on one canvas)
        # --------------------------------------------------------------
        L = np.load(seg_path).astype(np.int32)
        H, W = L.shape
        canvas_all = make_canvas(H, W, scale=1)

        for cat, instances in results.items():
            fill_color = CATEGORY_COLORS_BGR.get(int(cat), (0, 0, 0))
            for inst in instances:
                if not inst:
                    continue
                ext_xy   = flatten_to_xylist(inst[0])
                holes_xy = [flatten_to_xylist(hv) for hv in inst[1:]]
                draw_filled_polygon_with_holes_safe(
                    canvas_all,
                    ext_xy,
                    holes_xy,
                    fill_color=fill_color,
                    edge_color=EDGE_COLOR,
                    point_color=POINT_COLOR,
                    thickness=1,
                    radius=1,
                    scale=1,
                )

        cv2.imwrite(os.path.join(args.overview_dir, f"{name}.png"), canvas_all)

        # --------------------------------------------------------------
        # COCO predictions (per-category)
        # --------------------------------------------------------------
        image_id = int(name.split('_')[-1])

        for cat in args.categories:
            cat = int(cat)
            inst_list = results.get(cat, [])
            for inst in inst_list:
                if not inst:
                    continue
                ext_xy = flatten_to_xylist(inst[0])
                holes_xy = [flatten_to_xylist(hv) for hv in inst[1:]]
                area, bbox = area_bbox_with_holes(ext_xy, holes_xy)

                score = instance_score_mean_prob(
                    prob_map=prob_map,
                    cat_id=cat,
                    ext_xy=ext_xy,
                    holes_xy=holes_xy,
                    H=H,
                    W=W
                )

                coco_obj = {
                    "image_id": image_id,
                    "category_id": 100,
                    "segmentation": inst,  # [[x1,y1,...], [hole...], ...]
                    "score": score,
                    "iscrowd": 0,
                    "area": area,
                    "bbox": bbox,
                }
                per_cat_preds[cat].append(coco_obj)

    if num_done > 0:
        fps = num_done / total_time
        avg_t = total_time / num_done
        print(f"\n[Polygonization FPS] {fps:.2f} img/s")
        print(f"[Avg time/image] {avg_t:.3f} s/img")
        print(f"[Total images] {num_done}, [Total time] {total_time:.2f} s")

    # ------------------------------------------------------------------
    # Save COCO prediction json per category
    # ------------------------------------------------------------------
    for cat in args.categories:
        cat = int(cat)
        cat_name = ID2NAME.get(cat, f"cat_{cat}")
        save_path = os.path.join(args.cats_out_dir, f"{cat_name}.json")
        with open(save_path, 'w') as fp:
            json.dump(per_cat_preds[cat], fp)

if __name__ == "__main__":
    main()

