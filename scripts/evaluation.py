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
