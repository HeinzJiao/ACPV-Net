"""
The code is adopted from https://github.com/spgriffin/polis
"""

import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict
from pycocotools import mask as maskUtils
from shapely import geometry
from shapely.geometry import Polygon


def seg_to_rle(seg, H, W):
    """polygon / RLE-dict / RLE-string -> RLE (or RLEs)"""
    if isinstance(seg, list):
        # polygon(s)
        return maskUtils.frPyObjects(seg, H, W)
    elif isinstance(seg, dict):
        rle = seg
        if isinstance(rle.get('counts', None), str):
            rle = {'size': rle['size'], 'counts': rle['counts'].encode('ascii')}
        return rle
    elif isinstance(seg, str):
        return {'size': [H, W], 'counts': seg.encode('ascii')}
    else:
        raise TypeError(f"Unsupported segmentation type: {type(seg)}")

def seg_to_exterior_polygons(seg, H, W):
    """
    This function extracts the exterior boundary polygon of a single COCO instance, 
    ignoring all interior holes, and unifies polygon and RLE representations.
    """
    polys = []

    # Case 1: Polygon-based segmentation
    # COCO polygon format:
    #   seg = [exterior, hole_1, hole_2, ...]
    # We explicitly take seg[0] as the exterior contour.
    if isinstance(seg, list):
        if len(seg) == 0:
            return polys

        exterior = np.asarray(seg[0], dtype=np.float32).reshape(-1, 2)
        if exterior.shape[0] >= 3:
            polys.append(exterior)
        return polys

    # Case 2: RLE-based segmentation
    # Decode RLE to binary mask
    rle = seg_to_rle(seg, H, W)
    m = maskUtils.decode(rle)  # (H, W) or (H, W, K)

    # Merge multiple masks if present
    if m.ndim == 3:
        m = (m > 0).any(axis=2).astype(np.uint8)
    else:
        m = (m > 0).astype(np.uint8)

    # Extract exterior contours from the binary mask
    contours, _ = cv2.findContours(
        m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    for cnt in contours:
        if cnt.shape[0] >= 3:
            cnt = cnt.squeeze(1).astype(np.float32)  # (N, 2)
            polys.append(cnt)

    return polys

def bounding_box(points):
    """returns a list containing the bottom left and the top right 
    points in the sequence
    Here, we traverse the collection of points only once, 
    to find the min and max for x and y
    """
    bot_left_x, bot_left_y = float('inf'), float('inf')
    top_right_x, top_right_y = float('-inf'), float('-inf')
    for x, y in points:
        bot_left_x = min(bot_left_x, x)
        bot_left_y = min(bot_left_y, y)
        top_right_x = max(top_right_x, x)
        top_right_y = max(top_right_y, y)

    return [bot_left_x, bot_left_y, top_right_x - bot_left_x, top_right_y - bot_left_y]

def compare_polys(poly_a, poly_b):
    """Compares two polygons via the "polis" distance metric.
    See "A Metric for Polygon Comparison and Building Extraction
    Evaluation" by J. Avbelj, et al.
    Input:
        poly_a: A Shapely polygon.
        poly_b: Another Shapely polygon.
    Returns:
        The "polis" distance between these two polygons.
    """
    bndry_a, bndry_b = poly_a.exterior, poly_b.exterior
    dist = polis(bndry_a.coords, bndry_b)
    dist += polis(bndry_b.coords, bndry_a)
    return dist

def polis(coords, bndry):
    """Computes one side of the "polis" metric.
    Input:
        coords: A Shapley coordinate sequence (presumably the vertices
                of a polygon).
        bndry: A Shapely linestring (presumably the boundary of
        another polygon).
    
    Returns:
        The "polis" metric for this pair.  You usually compute this in
        both directions to preserve symmetry.
    """
    sum = 0.0
    for pt in (geometry.Point(c) for c in coords[:-1]): # Skip the last point (same as first)
        sum += bndry.distance(pt)
    return sum/float(2*len(coords))


class PolisEval:
    """
    PoLiS evaluation on COCO-style GT and prediction files.

    Matching strategy:
        - For each GT instance, find the prediction with the highest bbox IoU.
        - If IoU > iou_thresh, compute PoLiS between this pair.
        - Unmatched GTs and predictions are ignored (no penalty).
    """

    def __init__(self, cocoGt, cocoDt, iou_thresh=0.5):
        """
        Args:
            cocoGt (COCO): COCO object for ground truth.
            cocoDt (COCO): COCO object for detections (loadRes result).
            iou_thresh (float): IoU threshold for GT–DT matching.
        """
        self.cocoGt = cocoGt
        self.cocoDt = cocoDt
        self.iou_thresh = iou_thresh

        self.evalImgs = defaultdict(list)
        self.eval = {}
        self._gts = defaultdict(list)
        self._dts = defaultdict(list)
        self.stats = []

        self.imgIds = list(sorted(self.cocoGt.imgs.keys()))

    def _prepare(self):
        """Group GT and DT annotations by image_id."""
        gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=self.imgIds))
        dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=self.imgIds))

        self._gts = defaultdict(list)
        self._dts = defaultdict(list)

        for gt in gts:
            self._gts[gt["image_id"]].append(gt)
        for dt in dts:
            self._dts[dt["image_id"]].append(dt)

        self.evalImgs = defaultdict(list)
        self.eval = {}

    def evaluateImg(self, imgId):
        """
        Compute average PoLiS for a single image over matched GT–DT pairs.

        Args:
            imgId (int): Image id.

        Returns:
            float: Average PoLiS for this image (0.0 if no valid match).
        """
        gts = self._gts[imgId]
        dts = self._dts[imgId]
        if len(gts) == 0 or len(dts) == 0:
            return 0

        # Extract polygon and corresponding bbox for each GT/DT
        gt_polygons = []
        dt_polygons = []
        H = self.cocoGt.imgs[imgId]['height']
        W = self.cocoGt.imgs[imgId]['width']
        for gt in gts:
            polys = seg_to_exterior_polygons(gt['segmentation'], H, W)
            gt_polygons.extend(polys)
        for dt in dts:
            polys = seg_to_exterior_polygons(dt['segmentation'], H, W)
            dt_polygons.extend(polys)

        if len(gt_polygons) == 0 or len(dt_polygons) == 0:
            return 0

        gt_bboxs = [bounding_box(p) for p in gt_polygons]
        dt_bboxs = [bounding_box(p) for p in dt_polygons]

        # bbox IoU matrix: rows = dt, cols = gt
        iscrowd = [0] * len(gt_bboxs)
        ious = maskUtils.iou(dt_bboxs, gt_bboxs, iscrowd)

        img_polis_sum = 0.0
        num_matched = 0

        # For each GT, pick the DT with maximum IoU
        for j, gt_poly in enumerate(gt_polygons):
            matched_idx = np.argmax(ious[:, j])
            iou = ious[matched_idx, j]

            if iou > self.iou_thresh:
                polis_val = compare_polys(
                    Polygon(gt_poly), Polygon(dt_polygons[matched_idx])
                )
                img_polis_sum += polis_val
                num_matched += 1

        if num_matched == 0:
            return 0.0
        return img_polis_sum / float(num_matched)


    def evaluate(self, verbose=True):
        """
        Compute dataset-level average PoLiS.

        Returns:
            float: Average PoLiS over all images with at least one matched pair.
        """
        self._prepare()
        polis_tot = 0.0
        num_valid_imgs = 0

        for imgId in tqdm(self.imgIds, desc="Evaluating PoLiS"):
            img_polis_avg = self.evaluateImg(imgId)
            if img_polis_avg == 0.0:
                continue
            polis_tot += img_polis_avg
            num_valid_imgs += 1

        polis_avg = polis_tot / float(num_valid_imgs) if num_valid_imgs > 0 else 0.0

        if verbose:
            print("Average PoLiS: %.2f" % polis_avg)

        return polis_avg


    



    