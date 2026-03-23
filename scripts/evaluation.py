#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run polygon evaluation utilities for COCO-format GT and prediction files.

Supported modes:
- `coco_iou`: standard COCO segmentation AP / AR
- `boundary_iou`: boundary IoU
- `polis`: polygon similarity
- `angle`: mean max tangent angle error
- `ciou`: contour IoU / cIoU

Example:
    python evaluation.py \
        --gt-file /path/to/gt.json \
        --dt-file /path/to/pred.json \
        --eval-type coco_iou
"""

import argparse

from multiprocess import Pool
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from boundary_iou.coco_instance_api.coco import COCO as BCOCO
from boundary_iou.coco_instance_api.cocoeval import COCOeval as BCOCOeval
from metrics.polis import PolisEval
from metrics.angle_eval import ContourEval
from metrics.cIoU import compute_IoU_cIoU

def coco_eval(annFile, resFile):
    eval_type_idx = 1
    annType = ['bbox', 'segm']
    print('Running evaluation for *%s* results.' % (annType[eval_type_idx]))

    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(resFile)

    imgIds = cocoGt.getImgIds()
    imgIds = imgIds[:]

    cocoEval = COCOeval(cocoGt, cocoDt, annType[eval_type_idx])
    cocoEval.params.imgIds = imgIds
    cocoEval.params.catIds = [100]
    cocoEval.evaluate()
    cocoEval.accumulate()

    cocoEval.summarize()
    stats = cocoEval.stats

    print("Average Precision  (AP) @[ IoU=0.50:0.95 | area=all   | maxDets=100 ] = %.3f" % stats[0])
    print("Average Precision  (AP) @[ IoU=0.50      | area=all   | maxDets=100 ] = %.3f" % stats[1])
    print("Average Precision  (AP) @[ IoU=0.75      | area=all   | maxDets=100 ] = %.3f" % stats[2])
    print("Average Precision  (AP) @[ IoU=0.50:0.95 | area=small | maxDets=100 ] = %.3f" % stats[3])
    print("Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium| maxDets=100 ] = %.3f" % stats[4])
    print("Average Precision  (AP) @[ IoU=0.50:0.95 | area=large | maxDets=100 ] = %.3f" % stats[5])

    print("Average Recall     (AR) @[ IoU=0.50:0.95 | area=all   | maxDets=100 ] = %.3f" % stats[8])
    # print("Average Recall     (AR) @[ IoU=0.50:0.95 | area=small | maxDets=100 ] = %.3f" % stats[9])
    # print("Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium| maxDets=100 ] = %.3f" % stats[10])
    # print("Average Recall     (AR) @[ IoU=0.50:0.95 | area=large | maxDets=100 ] = %.3f" % stats[11])

    return stats

def boundary_eval(annFile, resFile):
    dilation_ratio = 0.02
    cocoGt = BCOCO(annFile, get_boundary=True, dilation_ratio=dilation_ratio)
    cocoDt = cocoGt.loadRes(resFile)
    cocoEval = BCOCOeval(cocoGt, cocoDt, iouType="boundary", dilation_ratio=dilation_ratio)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

def polis_eval(annFile, resFile):
    gt_coco = COCO(annFile)
    dt_coco = gt_coco.loadRes(resFile)
    polisEval = PolisEval(gt_coco, dt_coco)
    polisEval.evaluate()

def max_angle_error_eval(annFile, resFile):
    gt_coco = COCO(annFile)
    dt_coco = gt_coco.loadRes(resFile)
    contour_eval = ContourEval(gt_coco, dt_coco)
    pool = Pool(processes=20)
    max_angle_diffs = contour_eval.evaluate(pool=pool)
    print('Mean max tangent angle error(MTA): ', f"{max_angle_diffs.mean():.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run polygon evaluation metrics for COCO-format GT and prediction files."
    )
    parser.add_argument("--gt-file", default="", help="Path to the COCO-format ground-truth JSON file.")
    parser.add_argument("--dt-file", default="", help="Path to the COCO-format prediction JSON file.")
    parser.add_argument("--output", default=None, metavar="DIR", help="Directory to save evaluation outputs when required.")
    parser.add_argument(
        "--eval-type",
        default="coco_iou",
        choices=["coco_iou", "boundary_iou", "polis", "angle", "ciou"],
        help="Evaluation mode to run."
    )
    args = parser.parse_args()

    eval_type = args.eval_type
    gt_file = args.gt_file
    dt_file = args.dt_file
    if eval_type == 'coco_iou':
        # eval_coco(gt_file, dt_file, args.output)
        coco_eval(gt_file, dt_file)
    elif eval_type == 'boundary_iou':
        boundary_eval(gt_file, dt_file)
    elif eval_type == 'polis':
        polis_eval(gt_file, dt_file)
    elif eval_type == 'angle':
        max_angle_error_eval(gt_file, dt_file)
    elif eval_type == 'ciou':
        compute_IoU_cIoU(dt_file, gt_file, args.output)
    else:
        raise RuntimeError(
            'Please choose a valid eval type from '
            '["coco_iou", "boundary_iou", "polis", "angle", "ciou"].'
        )


    """
    python3 evaluation.py \
        --gt-file ./HiSup-main/data/deventer_512/vectorized_polygons_split/coco_annotations_test/coco_artificial_structure.json \
        --dt-file ./UNetFormer/fig_results/unetformer_deventer_512_split_by_big_image/predictions_artificial_structure.json \
        --eval-type ciou
        
    # ----- SegFormer, OCRNet (PS cluster) -----
    python3 evaluation.py \
        --gt-file ../model4/data/deventer_512/test/road_bridge_with_info.json \
        --dt-file ./multi_class_seg_baselines/work_dirs/segformer_mit-b2_1x16-160k_deventer512/dp/predictions_road_bridge.json \
        --eval-type ciou

    # ----- Ours -----
    ## Shanghai
    python3 evaluation.py \
        --gt-file ../dataset3/shanghai/dataset/annotations/test_updated_bboxes.json \
        --dt-file ./ACPV-Net/outputs/shanghai_building_vmamba-s_m_vh-ldm_kl4_b8/ddim/poly_pslg_dp-2.5_angle_tol-10_corner_eps-2_replace_thresh-5_nms-3_th-0.5_topk-1k/categories/building.json \
        --eval-type ciou
        
    # ------------------------- DeepSnake (PS cluster) -------------------------
    ## Deventer-512
    python3 evaluation.py \
        --gt-file ./ACPV-Net/data/deventer_512/test/annotations/water.json \
        --dt-file ./PS_cluster/dance-master/output/deventer_512/dsnake_R50_bs8_ep100/categories_preds_poly/water.json \
        --eval-type ciou
        
    ## WHU-Building
    python3 evaluation.py \
        --gt-file ./HiSup-main/data/WHU_Building/annotation/test_512.json \
        --dt-file ./PS_cluster_baseline_results/dance-master/output/whu_building/dsnake_R50_bs8_ep100/dsnake_whu_outer_holes.json \
        --eval-type coco_iou

    # ------------------------- FFL (OTP) -------------------------
    python3 evaluation.py \
    --gt-file ./ACPV-Net/data/deventer_512/test/annotations/road_bridge.json \
    --dt-file "../work3/FFL-main/data/mapping_challenge_dataset(Deventer_512)/eval_runs_summary/road_bridge.json" \
    --eval-type ciou
    
    ## WHU-Building
    python3 evaluation.py \
        --gt-file "../work3/FFL-main/data/mapping_challenge_dataset(whu_building)/raw/val/annotation.json" \
        --dt-file "../work3/FFL-main/data/mapping_challenge_dataset(whu_building)/eval_runs/mapping_dataset.unet_resnet101_pretrained_2025-10-03_14_22_54/test.annotation.poly.acm.tol_1.json" \
        --eval-type coco_iou  # whu building
    
    # ------------------------- TopDiG (PS cluster) -------------------------
    ## Deventer-512
    python3 evaluation.py \
        --gt-file ./ACPV-Net/data/deventer_512/test/annotations/water.json \
        --dt-file ./PS_cluster/TopDiG/records/Deventer_512_summary/water.json \
        --eval-type ciou
    
    ## WHU-Building
    python3 evaluation.py \
        --gt-file ./HiSup-main/data/WHU_Building/annotation/test_512.json \
        --dt-file ./PS_cluster_baseline_results/TopDiG/records/WHU_Building_TCND_300pt/coco_predictions.json \
        --eval-type coco_iou

    ## zero shot
    python3 evaluation.py  \
    --gt-file ./ACPV-Net/data/giethoorn_512/test/annotations/water.json  \
    --dt-file ./PS_cluster/TopDiG/records/Deventer512_to_Giethoorn512_water_TCND_200pt_zeroshot/coco_predictions.json  \
    --eval-type ciou

    # ------------------------- HiSup (OTP) -------------------------
    ## Deventer-512
    python3 evaluation.py \
        --gt-file ./ACPV-Net/data/deventer_512/test/annotations/road_bridge.json \
        --dt-file ../work3/HiSup-main/outputs/deventer_512_summary/road_bridge.json \
        --eval-type ciou

    ## WHU-Building  
    python3 evaluation.py \
        --gt-file ./ACPV-Net/data/WHU_Building/annotation/test_512.json \
        --dt-file ../work3/HiSup-main/outputs/whu_building_512_hrnet48/30e/predictions.json \
        --eval-type mta

    ## zero-shot
    python3 evaluation.py \
        --gt-file ./ACPV-Net/data/giethoorn_512/test/annotations/building.json \
        --dt-file ../work3/HiSup-main/outputs/zero_shot/deventer2giethoorn/building_hrnet48/predictions.json \
        --eval-type ciou
        
    # ------------------------- GCP (PS cluster) -------------------------
    python3 evaluation.py \
        --gt-file ./ACPV-Net/data/deventer_512/test/annotations/water.json \
        --dt-file ./PS_cluster/GCP/work_dirs/deventer_512_summary/categories/water.json \
        --eval-type ciou

    ## zero-shot
    python3 evaluation.py  \
    --gt-file ./ACPV-Net/data/giethoorn_512/test/annotations/water.json  \
    --dt-file ./PS_cluster/GCP/work_dirs/gcp_deventer512_to_giethoorn512_water_zeroshot/coco_preds.segm.poly.json  \
    --eval-type ciou
    
    ## WHU-Building
    python3 evaluation.py \
        --gt-file ./ACPV-Net/data/WHU_Building/annotation/test_512.json \
        --dt-file ./PS_cluster/GCP/work_dirs/summary_whu_building/coco_preds.segm.poly.json \
        --eval-type coco_iou
        

    # ----- Agricultural parcel extraction -----
    python3 evaluation.py \
        --gt-file ../agricultural_parcel_extraction/AI4SmallFarms/annotations/test.json \
        --dt-file ../agricultural_parcel_extraction/AI4SmallFarms_baseline_results/e2evap_dp-1.0.json \
        --eval-type coco_iou
    """
