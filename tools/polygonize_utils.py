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
    清除多类别标签图中的小面积碎片，并并入周围占比最多的类别。
    - L: HxW int32, 多类别标签
    - min_area: 面积阈值（像素数），小于该面积的连通域会被移除
    - connectivity: 4 或 8 连通
    - win_sizes: 依次使用的滑窗尺寸，用周围多数类填充 (-1) 像素
    返回: L_clean (HxW int32)
    """
    H, W = L.shape
    L_clean = L.copy()
    unknown = -1

    present_labels = np.unique(L_clean)  # e.g., [0..6]

    # 1) 逐类移除小连通域（置为 -1）
    for c in present_labels:
        mask = (L_clean == c).astype(np.uint8)
        num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                L_clean[lab == i] = unknown

    # 2) 多数类滑窗填充 (-1)
    for k in win_sizes:
        U = (L_clean == unknown)
        if not U.any():
            break
        kernel = np.ones((k, k), np.uint8)

        # 统计每个类别在 kxk 邻域内的像素数
        counts = []
        for c in present_labels:
            bin_map = (L_clean == c).astype(np.uint8)
            cnt = cv2.filter2D(bin_map, -1, kernel, borderType=cv2.BORDER_REPLICATE)
            counts.append(cnt)
        counts = np.stack(counts, axis=-1)  # H x W x C
        best_idx = np.argmax(counts, axis=-1)
        assign_map = present_labels[best_idx]

        L_clean[U] = assign_map[U]

    # 3) 若仍有未填充的像素，用“最近类”填充
    U = (L_clean == unknown)
    if U.any():
        dstack = []
        for c in present_labels:
            # 到“类别 c 的像素”的距离：对 (L_clean==c) 取补集做 EDT
            dstack.append(distance_transform_edt(L_clean != c))
        dstack = np.stack(dstack, axis=-1)  # H x W x C
        best_idx = np.argmin(dstack, axis=-1)
        assign_map = present_labels[best_idx]
        L_clean[U] = assign_map[U]

    return L_clean


# ---------- Step 1: 全局边界网（标签跳变→单位网格线段） ----------
def build_global_segments(L, include_frame=True):
    """
    从多类别标签图 L 生成全局边界网的单位网格线段（水平/垂直，端点为整数网格顶点）。
    额外可选：把整张图像的外框四条边也加入线网，确保贴边实例可被闭合 polygonize。
    返回：segments = [((x0,y0),(x1,y1)), ...]  每段长度为 1。
    """
    H, W = L.shape
    segs = []

    # 内部标签跳变 -> 竖直线段
    diff_v = (L[:, 1:] != L[:, :-1])
    ys, xs = np.where(diff_v)
    for y, x in zip(ys, xs + 1):
        segs.append(((int(x), int(y)), (int(x), int(y + 1))))

    # 内部标签跳变 -> 水平线段
    diff_h = (L[1:, :] != L[:-1, :])
    ys, xs = np.where(diff_h)
    for y, x in zip(ys + 1, xs):
        segs.append(((int(x), int(y)), (int(x + 1), int(y))))

    if include_frame:
        # 顶边 y=0 和 底边 y=H
        for x in range(W):
            segs.append(((x, 0), (x + 1, 0)))
            segs.append(((x, H), (x + 1, H)))
        # 左边 x=0 和 右边 x=W
        for y in range(H):
            segs.append(((0, y), (0, y + 1)))
            segs.append(((W, y), (W, y + 1)))

    return segs


def polygonize_from_segments(segments):
    ls = [LineString([p0, p1]) for (p0, p1) in segments]
    mls = MultiLineString(ls)
    merged = unary_union(mls)
    return list(polygonize(merged))


# ---------- Step 2: 结构角点（度≠2） ----------
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


# ---------- Step 3: 预测顶点吸附到边界网 ----------
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


# 邻接表
def build_adjacency(segments):
    adj = {}
    for p0, p1 in segments:
        adj.setdefault(p0, set()).add(p1)
        adj.setdefault(p1, set()).add(p0)
    return adj


# 在线判断“重要转折角点”
def is_important_corner(v, adj, tol_deg=15.0):
    nbrs = list(adj.get((int(v[0]), int(v[1])), ()))
    d = len(nbrs)
    if d != 2:
        return True  # 结构角点：端点 / T / X
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


# 距离优先 + 在线角点判断 的 kNN 吸附
def snap_points_to_boundary_knn_distance_first_simple(
    pred_xy, boundary_xy, segments,
    dist_thresh=3.0, k=8, tol_deg=15.0
):
    """
    将预测顶点吸附到“全局边界网”的顶点（距离优先 + 在线角点判断）。

    策略（对每个预测点 q）：
      1) 以 q 为查询，取边界网顶点的 k 近邻（KDTree）。
      2) 按距离升序遍历候选，忽略距离 > dist_thresh 的候选；
         遇到第一个“重要转折角点”（见 `is_important_corner`）则选之；
         若阈值内不存在角点，则选择最近候选。
      3) 所有吸附结果去重（多个预测点落同一网格点只保留一个）。

    参数
    ----------
    pred_xy : ndarray, shape (N, 2)
        预测顶点集合（像素坐标，int/float 均可）。
    boundary_xy : ndarray, shape (M, 2)
        全局边界网的顶点集合（整数网格坐标，通常来自线段端点）。
    segments : list of ((x0, y0), (x1, y1))
        全局边界网的单位线段集合（用于构建邻接表和角点判断）。
    dist_thresh : float, 默认 3.0
        吸附半径（像素）。距离大于该阈值的候选将被忽略。
    k : int, 默认 8
        KDTree 近邻数量（通常 8~16 即可）。
    tol_deg : float, 默认 15.0
        几何角点的直角容差（度）。

    返回
    -------
    snapped_arr : ndarray, shape (K, 2), dtype int32
        吸附后的（去重）顶点坐标集合。
    snapped_set : set[(x, y)]
        同上，集合形式，元素为整数元组。

    复杂度
    -------
    构树 O(M log M)；查询 O(N log M + N·k)。N=预测点数，M=边界顶点数。
    """
    if pred_xy.shape[0] == 0 or boundary_xy.shape[0] == 0:
        return np.zeros((0,2), dtype=np.int32), set()

    # 构邻接表一次，在线判断角点
    adj = build_adjacency(segments)

    k = min(int(k), boundary_xy.shape[0])
    tree = cKDTree(boundary_xy.astype(np.float32))
    d, idx = tree.query(pred_xy.astype(np.float32), k=k)
    if k == 1:
        d = d[:, None]; idx = idx[:, None]

    keep = set()
    for i in range(d.shape[0]):
        # 最近距离
        order = np.argsort(d[i])
        j0 = idx[i][order[0]]
        d0 = d[i][order[0]]
        if d0 > float(dist_thresh):
            continue  # 离边太远，丢弃该预测点

        chosen = None
        # 按距离从近到远扫描，直到超过阈值
        for j in idx[i][order]:
            if d[i][np.where(idx[i]==j)[0][0]] > float(dist_thresh):
                break
            v = boundary_xy[j]
            if is_important_corner(v, adj, tol_deg=tol_deg):
                chosen = j
                break
        if chosen is None:
            chosen = j0  # 阈值内没有角点，取最近

        x, y = boundary_xy[chosen]
        keep.add((int(x), int(y)))

    if not keep:
        return np.zeros((0,2), dtype=np.int32), set()
    arr = np.asarray(list(keep), dtype=np.int32)
    return arr, keep


# ---------- Step 4: 取目标类别的外环 + 孔洞（来自 polygonize 的 faces） ----------
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


# ---------- Step 5: 在环上保留角点/吸附点，输出扁平顶点序列 ----------
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
    从该类别的所有环（外环+内环）收集边界顶点集合。
    返回：
      - cat_boundary_xy: ndarray (M,2) int32
      - cat_boundary_set: set{(x,y)}
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


# ===== 可视化小工具 =====
def make_canvas(H, W, scale=2):
    """白底画布；注意这里画布大小用 (H+1, W+1)，覆盖到网格边界 H、W。"""
    canvas = np.ones(( (H+1)*scale, (W+1)*scale, 3 ), dtype=np.uint8) * 255
    return canvas


def _to_img_pt(x, y, scale):
    return (int(x*scale), int(y*scale))


def draw_segments(canvas, segments, color=(0,0,0), thickness=1, scale=2):
    """按单位网格线段画黑线。"""
    for (p0, p1) in segments:
        x0,y0 = p0; x1,y1 = p1
        cv2.line(canvas, _to_img_pt(x0,y0,scale), _to_img_pt(x1,y1,scale), color, thickness, lineType=cv2.LINE_AA)


def draw_points(canvas, pts_xy, color, radius=2, scale=2):
    """画整型网格点/像素点。pts_xy: Nx2 或 list[(x,y), ...]"""
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
    将给定点集在画布上标为单像素颜色（不画圆）。
    - canvas: uint8 (Hs, Ws, 3)
    - pts_xy: iterable of (x,y)  网格坐标（整数）
    - color : BGR 三元组
    - scale : 与 _to_img_pt 保持一致
    """
    Hs, Ws = canvas.shape[:2]
    bgr = (int(color[0]), int(color[1]), int(color[2]))
    for x, y in pts_xy:
        ix, iy = _to_img_pt(int(x), int(y), scale)  # 映射到可视化像素坐标
        if 0 <= ix < Ws and 0 <= iy < Hs:
            canvas[iy, ix] = bgr


def draw_polyline(canvas, ring_xy, color=(0,0,0), thickness=1, closed=True, scale=1, antialias=False):
    if len(ring_xy) >= 2:
        pts = np.array([_to_img_pt(int(x),int(y),scale) for x,y in ring_xy], dtype=np.int32)
        lt = cv2.LINE_AA if antialias else cv2.LINE_8  # 默认不用 AA
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
    安全绘制该实例的带孔多边形（不覆盖已上色区域；孔洞只抠自己颜色；高亮几何 overlap，且不改动边像素）。
    - canvas:  uint8 (H*scale, W*scale, 3) 白底
    - ext_xy:  [(x,y), ...] 该实例外环（整数网格）
    - holes_xy_list: [ [(x,y),...], ... ] 该实例内环列表
    - fill_color:     类别填充色 (BGR)
    - edge_color:     边线颜色（统一黑）
    - point_color:    顶点颜色（统一玫红，单像素）
    - overlap_color:  overlap 高亮颜色 (BGR)
    """
    Hs, Ws = canvas.shape[:2]

    # 1) 外环/孔洞掩膜（单通道）
    ext_mask = np.zeros((Hs, Ws), dtype=np.uint8)
    ext_pts  = np.array([_to_img_pt(x, y, scale) for x, y in ext_xy], dtype=np.int32)
    cv2.fillPoly(ext_mask, [ext_pts], 1)  # ext_mask: 黑底，外环覆盖区域白色

    hole_masks, hole_pts_list = [], []
    for hole_xy in holes_xy_list:
        hm = np.zeros((Hs, Ws), dtype=np.uint8)
        hole_pts = np.array([_to_img_pt(x, y, scale) for x, y in hole_xy], dtype=np.int32)
        cv2.fillPoly(hm, [hole_pts], 1)
        hole_masks.append(hm)  # 每个内部孔洞生成一张单通道mask，黑底，孔洞区域白色
        hole_pts_list.append(hole_pts)  # 保存孔洞多边形顶点(像素点)序列，后面做边界保护用

    # —— 本实例几何内部 = 外环 - 孔洞并集
    # interior_mask为True的像素，就是应该属于该实例内部的像素
    # 这张掩膜后面用来找 overlap：若某个像素在 interior_mask 内、但画布上却不是该实例所属类颜色，就说明已经被别的类别占了（重叠）。
    if hole_masks:
        holes_union = np.zeros((Hs, Ws), dtype=np.uint8)
        for hm in hole_masks:
            holes_union |= hm
        interior_mask = (ext_mask == 1) & (holes_union == 0)
    else:
        interior_mask = (ext_mask == 1)

    # 2) 只在“白底”处填充外环颜色（不覆盖已有类别）
    bg_mask   = (canvas[:, :, 0] == 255) & (canvas[:, :, 1] == 255) & (canvas[:, :, 2] == 255)
    fill_mask = (ext_mask == 1) & bg_mask
    if np.any(fill_mask):
        canvas[fill_mask] = np.array(fill_color, dtype=np.uint8)

    # 3) 对孔洞：只在“当前就是本类别填充色”的像素上抠白（避免抠掉其他类别）
    if hole_masks:
        fc = np.array(fill_color, dtype=np.uint8)
        is_self_color = (canvas[:, :, 0] == fc[0]) & (canvas[:, :, 1] == fc[1]) & (canvas[:, :, 2] == fc[2])
        for hm in hole_masks:
            carve_mask = (hm == 1) & is_self_color
            if np.any(carve_mask):
                canvas[carve_mask] = (255, 255, 255)

    # —— 边界蒙版（保护本实例边，避免 overlap 高亮覆盖边像素）
    edge_mask = np.zeros((Hs, Ws), dtype=np.uint8)
    cv2.polylines(edge_mask, [ext_pts], True, 1, thickness=1, lineType=cv2.LINE_8)
    for hole_pts in hole_pts_list:
        cv2.polylines(edge_mask, [hole_pts], True, 1, thickness=1, lineType=cv2.LINE_8)

    # —— 高亮 overlap：在“本实例几何内部”但当前像素不是本类颜色、且不是白底、且不在边上
    fc = np.array(fill_color, dtype=np.uint8)
    is_self_color = (canvas[:, :, 0] == fc[0]) & (canvas[:, :, 1] == fc[1]) & (canvas[:, :, 2] == fc[2])
    colored_mask  = ~((canvas[:, :, 0] == 255) & (canvas[:, :, 1] == 255) & (canvas[:, :, 2] == 255))

    overlap_mask = interior_mask & (~is_self_color) & colored_mask & (edge_mask == 0)
    is_black_edge = (canvas[:, :, 0] == 0) & (canvas[:, :, 1] == 0) & (canvas[:, :, 2] == 0)
    overlap_mask &= (~is_black_edge)
    if np.any(overlap_mask):
        canvas[overlap_mask] = np.array(overlap_color, dtype=np.uint8)

    # 4) 黑色描边 + 单像素玫红顶点（覆盖在上层）
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
    将无向边集合 segments 在结构角点处断开为若干 polyline。
    参数：
        segments: Iterable[ ((x1,y1),(x2,y2)) ]，无向线段列表
        struct_corners: set[(x,y)]，结构角点（度≠2）
    返回：
        polylines: List[List[(x,y)]]
            - 每条 polyline 是按顺序的顶点序列；内部点度=2；端点在结构角点或回到起点形成环
    """
    from collections import defaultdict, deque

    # --- 建立邻接表 & 边集合（无向） ---
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

    # 度字典
    deg = {v: len(nbs) for v, nbs in adj.items()}
    struct_corners = set((int(x), int(y)) for (x, y) in struct_corners)

    visited = set()  # 已访问的边（frozenset）
    polylines = []

    def next_unvisited_neighbor(u, prev=None):
        """返回 u 的一个与 prev 不同且边未访问的邻居；若无则 None。"""
        for w in adj[u]:
            if prev is not None and w == prev:
                continue
            if frozenset((u, w)) not in visited:
                return w
        return None

    def walk_path(start, nb):
        """
        从 (start -> nb) 沿着未访问的边一直走，
        遇到结构角点/度≠2（且非起点）就停止，形成一条 polyline。
        """
        path = [start, nb]
        visited.add(frozenset((start, nb)))
        prev, cur = start, nb

        while True:
            # 若当前点是结构角点（或度≠2），且不是起点，则结束
            if (cur in struct_corners or deg.get(cur, 0) != 2) and cur != start:
                break
            # 否则应有 1 条继续的未访问边
            nxt = next_unvisited_neighbor(cur, prev)
            if nxt is None:
                # 没路可走，停止（可能到达悬挂端点或已无未访问边）
                break
            visited.add(frozenset((cur, nxt)))
            path.append(nxt)
            prev, cur = cur, nxt

            # 防循环：若回到起点，形成闭环
            if cur == start:
                break

        # 去掉长度过短（少于 2 段）的路径
        if len(path) >= 2:
            polylines.append(path)

    # --- 第一遍：从度≠2（结构角点）出发，覆盖所有“开放端”的路径 ---
    # 这样能把“以角为端点的链”优先拆解干净
    for v in adj.keys():
        if deg.get(v, 0) != 2 or v in struct_corners:
            for nb in list(adj[v]):
                e = frozenset((v, nb))
                if e in visited:
                    continue
                walk_path(v, nb)

    # --- 第二遍：处理剩余“纯环”（所有点度=2，无角点）的边 ---
    # 随便挑一个未访问边起步，转一圈回到起点
    for e in list(edges):
        if e in visited:
            continue
        a, b = tuple(e)
        # 从 a->b 开始绕环
        path = [a, b]
        visited.add(frozenset((a, b)))
        prev, cur = a, b

        while True:
            nxt = next_unvisited_neighbor(cur, prev)
            if nxt is None:
                # 非严格环但也断尽了（理论上不该发生），结束
                break
            visited.add(frozenset((cur, nxt)))
            path.append(nxt)
            prev, cur = cur, nxt
            if cur == a:
                # 回到起点，形成闭合环
                break

        if len(path) >= 2:
            polylines.append(path)

    return polylines


def _simplify_snapped_on_polyline_v1(pl_xy, snapped_pts_set, is_closed, angle_tol_deg=5):
    """
    对当前 polyline 上的“吸附后的预测点序列”按折角阈值做简化：
    - 开放 polyline：使用 [首端点] + 吸附点序列 + [末端点] 作为锚点，判断吸附点是否近似共线；
      仅返回被保留的“吸附点”（端点不返回）。
    - 闭合 polyline：按环状邻接判断；若结果 < 3，则丢弃该环的吸附点。
    """
    if len(snapped_pts_set) == 0:
        return []

    pl_xy = np.asarray(pl_xy, dtype=np.int32)
    idx_map = {tuple(pl_xy[i]): i for i in range(len(pl_xy))}
    # 将吸附点转为 polyline 上的有序索引（去重）
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
            return 180.0  # 退化段，保留
        cosang = float(np.dot(v1, v2) / (n1 * n2))
        cosang = max(-1.0, min(1.0, cosang))
        ang = float(np.degrees(np.arccos(cosang)))
        return ang  # 0≈平行同向, 180≈反向

    kept = []

    if not is_closed:
        # —— 开放 polyline：用首末端点作为锚点来判断 —— #
        start_idx = 0
        end_idx = len(pl_xy) - 1
        aug = [start_idx] + idxs + [end_idx]

        for t in range(1, len(aug) - 1):
            i_prev, i_cur, i_next = aug[t - 1], aug[t], aug[t + 1]
            # 只对吸附点做判断（跳过首尾端点）
            if i_cur not in idxs:
                continue
            ang = _turn_angle(pl_xy[i_prev], pl_xy[i_cur], pl_xy[i_next])
            if ang > angle_tol_deg:  # 非近似平行/共线才保留
                kept.append(i_cur)

        # 注意：不返回端点，只返回被保留的吸附点
        kept = sorted(set(kept))

    else:
        # —— 闭合 polyline（环）：按环状邻接判断 —— #
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
            return []  # 丢弃该环的吸附点

    return [tuple(pl_xy[i]) for i in kept]


def _simplify_snapped_on_polyline_v2(pl_xy, snapped_pts_set, is_closed,
                                     angle_tol_deg=5, min_sep=2.0):
    """
    版本2：密度稀疏化(弧长最小间距) + 折角阈值简化
    - 先沿 polyline 的弧长坐标做最小间距合并（近邻簇只留1个）；
    - 再按 v1 的规则：
        * 开放 polyline：用 [首端点]+吸附点+[末端点] 判断是否近似共线；
        * 闭合 polyline：环状邻接判断；若结果 <3 则丢弃该环的吸附点。
    返回：List[tuple(x,y)]（保留的吸附点坐标）
    """
    if len(snapped_pts_set) == 0:
        return []

    pl_xy = np.asarray(pl_xy, dtype=np.int32)

    # 1) 将吸附点映射为 polyline 上的索引（有序去重）
    idx_map = {tuple(pl_xy[i]): i for i in range(len(pl_xy))}
    idxs = sorted({idx_map.get(tuple(p)) for p in snapped_pts_set if tuple(p) in idx_map})
    if len(idxs) == 0:
        return []

    # 2) 预计算弧长（累计长度）
    if len(pl_xy) >= 2:
        d = pl_xy[1:] - pl_xy[:-1]
        seglen = np.sqrt((d[:,0].astype(np.float32)**2 + d[:,1].astype(np.float32)**2))
        cum = np.concatenate([[0.0], np.cumsum(seglen, dtype=np.float32)])
    else:
        cum = np.array([0.0], dtype=np.float32)
    total_len = float(cum[-1]) if len(cum) else 0.0

    # 3) 弧长最小间距稀疏化（贪心，沿顺序保留，间距<min_sep的点丢弃）
    # 以序列 a b c d e 为例，若 ab < min_sep，b 会被跳过，接着判断 c 是看 ac 是否 >= min_sep，以此类推
    kept_idx = []
    last_s = None
    for i in idxs:
        s = float(cum[i])
        if last_s is None or (s - last_s) >= float(min_sep):
            kept_idx.append(i)
            last_s = s

    # 闭环：循环直到首末“跨环”间距 ≥ min_sep（总是从末尾删除一个）
    if is_closed and len(kept_idx) >= 2 and total_len > 0:
        while len(kept_idx) >= 2:
            s_first = float(cum[kept_idx[0]])
            s_last = float(cum[kept_idx[-1]])
            wrap_gap = (total_len - s_last + s_first)
            if wrap_gap >= float(min_sep):
                break
            kept_idx.pop()  # 继续去掉末尾一个

    idxs = kept_idx
    if len(idxs) == 0:
        return []

    # 4) 折角阈值简化（与 v1 一致）
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
        # 开放：用端点作锚，判断每个吸附点是否近似共线；端点不返回
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
        # 闭合：环状邻接判断；<3 则丢弃
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

