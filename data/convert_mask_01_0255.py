#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert binary masks between {0,1} and {0,255}.

For near-binary inputs such as JPEG-compressed masks, the script can also
apply threshold-based binarization before conversion.

Usage example:
  python data/convert_mask_01_0255.py \
    --root /path/to/input_masks \
    --out_root /path/to/output_masks \
    --mode 01_to_0255
"""
import argparse
import os
from os import path as osp
from PIL import Image
import numpy as np


def is_image(p, exts):
    return osp.isfile(p) and p.lower().split('.')[-1] in exts


def load_gray(path):
    img = Image.open(path).convert('L')
    arr = np.array(img, dtype=np.uint8)
    return arr


def save_gray(path, arr):
    img = Image.fromarray(arr.astype(np.uint8), mode='L')
    img.save(path)


def convert_mask(arr, mode, threshold=127):
    uniq = np.unique(arr)
    uniq_set = set(uniq.tolist())

    if mode == "0255_to_01":
        if uniq_set.issubset({0, 1}):
            return arr, "kept"
        if uniq_set.issubset({0, 255}):
            return (arr // 255).astype(np.uint8), "fixed"
        arr_bin = (arr > threshold).astype(np.uint8)
        return arr_bin, f"thresholded:{uniq.tolist()}"

    if mode == "01_to_0255":
        if uniq_set.issubset({0, 255}):
            return arr, "kept"
        if uniq_set.issubset({0, 1}):
            return (arr * 255).astype(np.uint8), "fixed"
        arr_bin = ((arr > threshold).astype(np.uint8) * 255).astype(np.uint8)
        return arr_bin, f"thresholded:{uniq.tolist()}"

    raise ValueError(f"Unsupported mode: {mode}")


def force_png_path(path):
    root, _ = osp.splitext(path)
    return root + ".png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="input masks root directory")
    ap.add_argument("--out_root", type=str, required=True, help="output masks root directory")
    ap.add_argument("--mode", type=str, required=True, choices=["0255_to_01", "01_to_0255"],
                    help="Conversion direction")
    ap.add_argument("--exts", type=str, default="png,jpg,jpeg,tif,tiff,bmp",
                    help="Matched extensions, comma-separated")
    ap.add_argument("--threshold", type=int, default=127,
                    help="Threshold used to binarize non-strict binary inputs")
    ap.add_argument("--dry-run", action="store_true", help="Print only, do not write files")
    args = ap.parse_args()

    exts = set([e.strip().lower() for e in args.exts.split(",") if e.strip()])
    os.makedirs(args.out_root, exist_ok=True)

    total = fixed = kept = warned = 0

    for dirpath, _, filenames in os.walk(args.root):
        for fn in filenames:
            if not is_image(osp.join(dirpath, fn), exts):
                continue
            fp = osp.join(dirpath, fn)
            total += 1
            arr = load_gray(fp)
            arr_out, status = convert_mask(arr, args.mode, threshold=args.threshold)
            rel_path = osp.relpath(fp, args.root)
            out_path = force_png_path(osp.join(args.out_root, rel_path))
            os.makedirs(osp.dirname(out_path), exist_ok=True)

            if status == "kept":
                arr_to_save = arr
                if args.mode == "0255_to_01":
                    arr_to_save = (arr > 0).astype(np.uint8)
                elif args.mode == "01_to_0255":
                    arr_to_save = ((arr > 0).astype(np.uint8) * 255).astype(np.uint8)
                kept += 1
                if not args.dry_run:
                    save_gray(out_path, arr_to_save)
                continue

            if status == "fixed":
                arr_to_save = arr_out
                if args.mode == "0255_to_01":
                    arr_to_save = (arr_out > 0).astype(np.uint8)
                elif args.mode == "01_to_0255":
                    arr_to_save = ((arr_out > 0).astype(np.uint8) * 255).astype(np.uint8)
                if not args.dry_run:
                    save_gray(out_path, arr_to_save)
                fixed += 1
                print(f"[FIX] {fp} -> {out_path}: mode={args.mode}")
                continue

            if status.startswith("thresholded:"):
                arr_to_save = arr_out
                if args.mode == "0255_to_01":
                    arr_to_save = (arr_out > 0).astype(np.uint8)
                elif args.mode == "01_to_0255":
                    arr_to_save = ((arr_out > 0).astype(np.uint8) * 255).astype(np.uint8)
                if not args.dry_run:
                    save_gray(out_path, arr_to_save)
                fixed += 1
                values = status.split("thresholded:", 1)[1]
                print(f"[FIX] {fp} -> {out_path}: thresholded unexpected values {values} with threshold={args.threshold}")
                continue

            warned += 1
            print(f"[WARN] {fp}: conversion skipped")

    print("\n== Done ==")
    print(f"Total: {total}")
    print(f"Kept: {kept}")
    print(f"Converted: {fixed}")
    print(f"Warnings: {warned}")
    if args.dry_run:
        print("(dry-run: no files were written)")
    else:
        print(f"Saved to: {args.out_root}")


if __name__ == "__main__":
    main()

    """
    python ./data/convert_mask_01_0255.py \
    --root ../AI4SmallFarms/train/edge_masks_w2_0255 \
    --out_root ../AI4SmallFarms/train/masks \
    --mode 0255_to_01
    """
