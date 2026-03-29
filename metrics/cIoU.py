"""
This is the code from https://github.com/zorzi-s/PolyWorldPretrainedNetwork.
@article{zorzi2021polyworld,
  title={PolyWorld: Polygonal Building Extraction with Graph Neural Networks in Satellite Images},
  author={Zorzi, Stefano and Bazrafkan, Shabab and Habenschuss, Stefan and Fraundorfer, Friedrich},
  journal={arXiv preprint arXiv:2111.15491},
  year={2021}
}
"""

from pycocotools.coco import COCO
#from .coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import json
import argparse
from tqdm import tqdm
import os
import cv2
# from boundary_iou.utils import compute_boundary_iou

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1

    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode

def calc_IoU(a, b):
    i = np.logical_and(a, b)  # intersection
    u = np.logical_or(a, b)  # union
    I = np.sum(i)
    U = np.sum(u)

    iou = I/(U + 1e-9)

    is_void = U == 0
    if is_void:
        return 1.0
    else:
        return iou

def iou_score(pred, targs):
    # pred[pred > 0] = 1
    # targs[targs > 0] = 1
    # pred = (pred > 0.5).astype(np.float32)
    # targs = (targs > 0.5).astype(np.float32)
    if pred.sum() == 0 and targs.sum() == 0:
        return 1
    intersection = (pred * targs).sum()
    union = pred.sum() + targs.sum() - intersection
    # return intersection, union
    return intersection / (union + 1e-10)

def compute_n_ratio(pred_vertices, gt_vertices):
    # Handle cases where the ground truth has no vertices
    if gt_vertices == 0:
        if pred_vertices == 0:
            return 1  # Both are zero, return a ratio of 1
        else:
            return float('NaN')  # If no ground truth vertices but there are predicted, return NaN
    else:
        return pred_vertices / gt_vertices  # Normal case, compute ratio


def seg_to_rle(seg, H, W):
    """Handle polygon / RLE-dict / RLE-string uniformly -> RLE dict/array."""
    if isinstance(seg, list):
        # polygon(s)
        return cocomask.frPyObjects(seg, H, W)
    elif isinstance(seg, dict):
        # RLE dict
        rle = seg
        if isinstance(rle.get('counts', None), str):
            rle = {'size': rle['size'], 'counts': rle['counts'].encode('ascii')}
        return rle
    elif isinstance(seg, str):
        # counts-only compressed RLE string
        return {'size': [H, W], 'counts': seg.encode('ascii')}
    else:
        raise TypeError(f"Unsupported segmentation type: {type(seg)}")


def compute_IoU_cIoU(input_json, gti_annotations, output_dir=None):
    """
    Compute IoU, Boundary-IoU, C-IoU, S-IoU, N-ratio metrics between predictions and ground truth.
    
    Arguments:
        input_json: str, path to COCO-style prediction file
        gti_annotations: str, path to COCO-style ground truth annotation file
        output_dir: str, where to save per-image evaluation results

    Notes / assumptions:
        1) If an image has no GT polygons, its GT mask is all zeros.
           - If the prediction is also empty => IoU is treated as 1.0 (void-void).
           - If the prediction has false positives => IoU becomes 0 and is penalized.
        2) Vertex counts include all polygon rings in ann["segmentation"]
           (exterior + interior holes), which is more accurate than using only
           ann["segmentation"][0].

    Returns:
        None. Prints dataset-level mean metrics and optionally saves per-image results.
    """
    # -------------------------------------------------------------------------
    # 0) Load GT / DT COCO APIs
    # -------------------------------------------------------------------------
    # Load ground truth annotations
    coco_gt = COCO(gti_annotations)

    coco_gt.dataset.setdefault('info', {"description": "tmp", "version": "1.0"})
    coco_gt.dataset.setdefault('licenses', [])

    # Load predictions
    submission_file = json.loads(open(input_json).read())
    coco_dt = coco_gt.loadRes(submission_file)

    # Extract filename without extension
    annotation_name = os.path.splitext(os.path.basename(input_json))[0]

    # -------------------------------------------------------------------------
    # 1) Determine which images to evaluate
    # -------------------------------------------------------------------------
    # Get image ids from the dataset
    all_image_ids = coco_gt.getImgIds()
    # Get image ids from the prediction
    pred_image_ids = set(coco_dt.getImgIds(catIds=coco_dt.getCatIds()))

    if not pred_image_ids.issubset(all_image_ids):
        missing_ids = pred_image_ids - set(all_image_ids)
        raise ValueError(f"Error: Predictions contain image IDs not in GT: {missing_ids}")

    # -------------------------------------------------------------------------
    # 2) Accumulators
    # -------------------------------------------------------------------------
    results = []  # per-image outputs (optional saving)
    list_iou = []
    list_ciou = []
    list_boundary_iou = []
    list_n_ratio = []

    # -------------------------------------------------------------------------
    # 3) Loop over images and compute metrics
    # -------------------------------------------------------------------------
    bar = tqdm(all_image_ids)
    for image_id in bar:
        img = coco_gt.loadImgs(image_id)[0]
        image_name = os.path.basename(img['file_name'])
        H, W = img["height"], img["width"]

        # ---------------------------------------------------------------------
        # 3.1) Build GT mask (union of all GT instances) and count GT vertices
        # ---------------------------------------------------------------------
        gt_ann_ids = coco_gt.getAnnIds(imgIds=image_id)
        mask_gti = np.zeros((H, W), dtype=np.uint8)
        N_GT = 0

        if gt_ann_ids:
            gt_anns = coco_gt.loadAnns(gt_ann_ids)
            for ann in gt_anns:
                rle = seg_to_rle(ann['segmentation'], H, W)
                m = cocomask.decode(rle)

                # Exterior + holes handling:
                # - m[..., 0] treated as exterior
                # - m[..., 1:] treated as holes (subtracted)
                if m.ndim > 2:
                    # Initialize the final mask as the exterior mask (first mask)
                    final_mask = m[:, :, 0].copy()
                    # Subtract all the interior holes (second mask onwards) from the exterior mask
                    for i in range(1, m.shape[-1]):
                        final_mask = np.logical_and(final_mask, np.logical_not(m[:, :, i]))
                else:
                    final_mask = m

                # Union into image-level GT mask.
                mask_gti |= final_mask.astype(np.uint8)
                
                # Vertex count: include all rings (exterior + holes).
                N_GT += sum(len(poly) // 2 for poly in ann['segmentation'])

        # ---------------------------------------------------------------------
        # 3.2) Build Pred mask (union of all predicted instances) and count vertices
        # ---------------------------------------------------------------------
        if image_id in pred_image_ids:
            dt_ann_ids = coco_dt.getAnnIds(imgIds=image_id)
            dt_anns = coco_dt.loadAnns(dt_ann_ids)

            mask = np.zeros((H, W), dtype=np.uint8)
            N = 0

            for ann in dt_anns:
                rle = seg_to_rle(ann['segmentation'], H, W)
                m = cocomask.decode(rle)

                # Exterior + holes handling:
                # - m[..., 0] treated as exterior
                # - m[..., 1:] treated as holes (subtracted)
                if m.ndim > 2:
                    # Initialize the final mask as the exterior mask (first mask)
                    final_mask = m[:, :, 0].copy()
                    # Subtract all the interior holes (second mask onwards) from the exterior mask
                    for i in range(1, m.shape[-1]):
                        final_mask = np.logical_and(final_mask, np.logical_not(m[:, :, i]))
                else:
                    final_mask = m

                mask |= final_mask.astype(np.uint8)
                N += sum(len(poly) // 2 for poly in ann['segmentation'])
        else:
            # No predictions for this image: empty mask and zero vertices.
            mask = np.zeros((img['height'], img['width']), dtype=np.uint8)
            N = 0

        # ---------------------------------------------------------------------
        # 3.3) Compute metrics for this image
        # ---------------------------------------------------------------------
        # (a) IoU on full masks
        iou = calc_IoU(mask, mask_gti)

        # (b) Boundary IoU (boundary extracted via erosion)
        boundary_mask = mask_to_boundary(mask, dilation_ratio=0.02)
        boundary_mask_gti = mask_to_boundary(mask_gti, dilation_ratio=0.02)
        boundary_iou = calc_IoU(boundary_mask, boundary_mask_gti)

        # (c) Vertex-count agreement term for C-IoU
        # ps ∈ [0, 1], higher is better; penalizes vertex under/over-prediction.
        ps = 1 - np.abs(N - N_GT) / (N + N_GT + 1e-9)

        # (d) N-ratio
        n_ratio = compute_n_ratio(N, N_GT)

        # ---------------------------------------------------------------------
        # 3.4) Accumulate
        # ---------------------------------------------------------------------
        list_iou.append(iou)
        list_ciou.append(iou * ps)
        list_boundary_iou.append(boundary_iou)
        list_n_ratio.append(n_ratio)

        results.append({
            "image_name": image_name,
            "n_ratio": round(n_ratio, 2),
            "iou": round(iou * 100, 2),
            "c_iou": round(iou * ps * 100, 2),
            "boundary_iou": round(boundary_iou * 100, 2),
        })

        # Progress display
        bar.set_description(
            f"iou:{np.mean(list_iou) * 100:.2f}, "
            f"boundary-iou:{np.mean(list_boundary_iou) * 100:.2f}, "
            f"n-ratio:{np.nanmean(list_n_ratio):.2f}, "
            f"c-iou:{np.mean(list_ciou) * 100:.2f}, "
        )

    # -------------------------------------------------------------------------
    # 4) Optional: Save per-image results (JSONL)
    # -------------------------------------------------------------------------
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, annotation_name + "_iou_biou_ciou_n-ratio.json")
        with open(output_file, 'w') as f:
            for result in sorted(results, key=lambda x: x['image_name']):
                json.dump(result, f)
                f.write('\n')

    # -------------------------------------------------------------------------
    # 5) Summary
    # -------------------------------------------------------------------------
    print("Done!")
    print("Mean IoU:", f"iou:{np.mean(list_iou) * 100:.2f}")
    print("Mean Boundary IoU:", f"boundary-iou:{np.mean(list_boundary_iou) * 100:.2f}")
    print("Mean n_ratio:", f"n-ratio:{np.nanmean(list_n_ratio):.2f}")
    print("Mean C-IoU:", f"c-iou:{np.mean(list_ciou) * 100:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", default="")
    parser.add_argument("--dt-file", default="")
    args = parser.parse_args()

    gt_file = args.gt_file
    dt_file = args.dt_file
    compute_IoU_cIoU(input_json=dt_file,
                    gti_annotations=gt_file)
