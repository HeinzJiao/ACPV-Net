"""
This is the code from https://github.com/zorzi-s/PolyWorldPretrainedNetwork.
@article{zorzi2021polyworld,
  title={PolyWorld: Polygonal Building Extraction with Graph Neural Networks in Satellite Images},
  author={Zorzi, Stefano and Bazrafkan, Shabab and Habenschuss, Stefan and Fraundorfer, Friedrich},
  journal={arXiv preprint arXiv:2111.15491},
  year={2021}
}
"""
"""

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


def compute_IoU_cIoU(input_json, gti_annotations, output_dir):
    """
    Compute IoU, Boundary-IoU, C-IoU, S-IoU, N-ratio metrics between predictions and ground truth.
    Arguments:
        input_json: str, path to COCO-style prediction file
        gti_annotations: str, path to COCO-style ground truth annotation file
        output_dir: str, where to save per-image evaluation results
    """
    # Ground truth annotations
    coco_gt = COCO(gti_annotations)

    # Predictions annotations
    submission_file = json.loads(open(input_json).read())
    coco_dt = coco_gt.loadRes(submission_file)

    # Extract filename without extension
    annotation_name = os.path.splitext(os.path.basename(input_json))[0]

    # Get image ids from the dataset
    all_image_ids = coco_gt.getImgIds()
    # Get image ids from the prediction
    pred_image_ids = set(coco_dt.getImgIds(catIds=coco_dt.getCatIds()))

    if not pred_image_ids.issubset(all_image_ids):
        missing_ids = pred_image_ids - set(all_image_ids)
        raise ValueError(f"Error: Predictions contain image IDs not in GT: {missing_ids}")

    results = []
    list_iou, list_ciou, list_siou_sigma = [], [], []
    list_siou_2sigma, list_siou_3sigma, list_siou = [], [], []
    list_n_ratio, list_boundary_iou = [], []
    pss, sps, sps_sigma, sps_2sigma, sps_3sigma = [], [], [], [], []

    bar = tqdm(all_image_ids)
    for image_id in bar:
        img = coco_gt.loadImgs(image_id)[0]
        image_name = os.path.basename(img['file_name'])

        # === Load GT mask ===
        annotation_ids = coco_gt.getAnnIds(imgIds=image_id)
        mask_gti = np.zeros((img['height'], img['width']), dtype=np.uint8)
        N_GT = 0
        if annotation_ids:
            annotations = coco_gt.loadAnns(annotation_ids)
            for ann in annotations:
                rle = cocomask.frPyObjects(ann['segmentation'], img['height'], img['width'])
                m = cocomask.decode(rle)

                # If there are multiple masks (e.g., exterior and interior contours)
                if m.ndim > 2:
                    # Initialize the final mask as the exterior mask (first mask)
                    final_mask = m[:, :, 0].copy()
                    # Subtract all the interior holes (second mask onwards) from the exterior mask
                    for i in range(1, m.shape[-1]):
                        final_mask = np.logical_and(final_mask, np.logical_not(m[:, :, i]))
                else:
                    final_mask = m
                # 将当前 annotation 的 final_mask 融合到整图的 mask_gti 中（按位或操作）；
                # 如果该图像上没有多边形标注（final_mask 全为 0），则 mask_gti 保持全 0 不变。
                mask_gti |= final_mask.astype(np.uint8)
                N_GT += sum(len(poly) // 2 for poly in ann['segmentation'])

        # === Load predicted mask ===
        if image_id in pred_image_ids:
            annotation_ids = coco_dt.getAnnIds(imgIds=image_id)
            annotations = coco_dt.loadAnns(annotation_ids)
            mask = np.zeros((img['height'], img['width']), dtype=np.uint8)
            N = 0
            for ann in annotations:
                rle = cocomask.frPyObjects(ann['segmentation'], img['height'], img['width'])
                m = cocomask.decode(rle)

                # If there are multiple masks (e.g., exterior and interior contours)
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
            mask = np.zeros((img['height'], img['width']), dtype=np.uint8)
            N = 0

        # === Compute metrics ===
        iou = calc_IoU(mask, mask_gti)
        boundary_mask = mask_to_boundary(mask, dilation_ratio=0.02)
        boundary_mask_gti = mask_to_boundary(mask_gti, dilation_ratio=0.02)
        boundary_iou = calc_IoU(boundary_mask, boundary_mask_gti)

        ps = 1 - np.abs(N - N_GT) / (N + N_GT + 1e-9)

        N0_sigma, N0_2sigma, N0_3sigma = 50, 90, 500
        sp_sigma = (1 + np.exp(0.1 * (3 - N0_sigma))) / (1 + np.exp(0.1 * (N - N0_sigma)))
        sp_2sigma = (1 + np.exp(0.1 * (3 - N0_2sigma))) / (1 + np.exp(0.1 * (N - N0_2sigma)))
        sp_3sigma = (1 + np.exp(0.1 * (3 - N0_3sigma))) / (1 + np.exp(0.1 * (N - N0_3sigma)))

        siou_sigma = iou * sp_sigma
        siou_2sigma = iou * sp_2sigma
        siou_3sigma = iou * sp_3sigma
        siou = (siou_sigma + siou_2sigma + siou_3sigma) / 3

        n_ratio = compute_n_ratio(N, N_GT)

        # === Accumulate results ===
        list_iou.append(iou)
        list_ciou.append(iou * ps)
        list_siou_sigma.append(siou_sigma)
        list_siou_2sigma.append(siou_2sigma)
        list_siou_3sigma.append(siou_3sigma)
        list_siou.append(siou)
        list_boundary_iou.append(boundary_iou)
        list_n_ratio.append(n_ratio)
        pss.append(ps)
        sps.append((sp_sigma + sp_2sigma + sp_3sigma) / 3)
        sps_sigma.append(sp_sigma)
        sps_2sigma.append(sp_2sigma)
        sps_3sigma.append(sp_3sigma)

        results.append({
            "image_name": image_name,
            "n_ratio": round(n_ratio, 2),
            "iou": round(iou, 2),
            "c_iou": round(iou * ps, 2),
            "boundary_iou": round(boundary_iou, 2),
            "s_iou_sigma": round(siou_sigma, 2),
            "s_iou_2sigma": round(siou_2sigma, 2),
            "s_iou_3sigma": round(siou_3sigma, 2),
            "s_iou": round(siou, 2),
            "ps": round(ps, 2),
            "sp_sigma": round(sp_sigma, 2),
            "sp_2sigma": round(sp_2sigma, 2),
            "sp_3sigma": round(sp_3sigma, 2)
        })

        bar.set_description(
            f"iou:{np.mean(list_iou):.4f}, n-ratio:{np.mean(list_n_ratio):.4f}, c-iou:{np.mean(list_ciou):.4f}, "
            f"boundary-iou:{np.mean(list_boundary_iou):.4f}, s-iou:{np.mean(list_siou):.4f}")

    # === Save per-image results ===
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, annotation_name + "_iou_ciou_metrics.json")
    with open(output_file, 'w') as f:
        for result in sorted(results, key=lambda x: x['image_name']):
            json.dump(result, f)
            f.write('\n')

    # === Print mean results ===
    print("Done!")
    print("Mean IoU:", np.mean(list_iou))
    print("Mean Boundary IoU:", np.mean(list_boundary_iou))
    print("Mean C-IoU:", np.mean(list_ciou))
    print("Mean S-IoU:", np.mean(list_siou))
    print("Mean S-IoU Sigma/2Sigma/3Sigma:", np.mean(list_siou_sigma), np.mean(list_siou_2sigma), np.mean(list_siou_3sigma))
    print("Mean ps:", np.mean(pss))
    print("Mean sp (avg):", np.mean(sps))
    print("Mean n_ratio:", np.nanmean(list_n_ratio))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", default="")
    parser.add_argument("--dt-file", default="")
    args = parser.parse_args()

    gt_file = args.gt_file
    dt_file = args.dt_file
    compute_IoU_cIoU(input_json=dt_file,
                    gti_annotations=gt_file)
