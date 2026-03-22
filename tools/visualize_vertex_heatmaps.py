import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse

def visualize_heatmaps(input_dir, output_dir, mode='gray'):
    """
    遍历输入文件夹中的所有顶点 heatmap（.npy），将其可视化为图像，并保存。

    Args:
        input_dir (str): 存放 heatmap 的文件夹路径，格式为 .npy，每个 shape = (H, W)，值域为 [0, 1]
        output_dir (str): 输出文件夹路径
        mode (str): 可视化模式，可选 "gray"（灰度图）或 "jet"（彩色热力图）
    """
    os.makedirs(output_dir, exist_ok=True)

    heatmap_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

    for fname in tqdm(heatmap_files, desc="Visualizing heatmaps"):
        path = os.path.join(input_dir, fname)
        heatmap = np.load(path)

        # 映射到 [0, 255]
        heatmap_uint8 = (heatmap * 255).clip(0, 255).astype(np.uint8)

        # 可视化
        if mode == 'gray':
            vis_img = heatmap_uint8  # 单通道灰度图
        elif mode == 'jet':
            vis_img = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # 彩色热力图
        else:
            raise ValueError(f"Unsupported mode: {mode}, choose 'gray' or 'jet'.")

        # 保存图像
        out_path = os.path.join(output_dir, fname.replace('.npy', '.png'))
        cv2.imwrite(out_path, vis_img)

    print(f"共处理 {len(heatmap_files)} 个 heatmap 文件，已保存到：\n{output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize junction heatmaps.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing .npy heatmap files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save visualized heatmap images.')
    parser.add_argument('--resize', type=int, nargs=2, default=None, help='Resize heatmap to given size (H, W).')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    visualize_heatmaps(args.input_dir, args.output_dir, mode='jet')

    """
    python ./tools/visualize_vertex_heatmaps.py \
    --input_dir ./outputs/deventer_vmamba-small_512_v_h_vitpose_simpler_decoder_multisteplr_cf/80k/junction_prob_npy \
    --output_dir ./outputs/deventer_vmamba-small_512_v_h_vitpose_simpler_decoder_multisteplr_cf/80k/junction_prob_npy_vis_jet
    
    python ./tools/visualize_vertex_heatmaps.py \
    --input_dir ./data/deventer_512/poly_gt_global_boundary/train/d-2_angle_tol_deg-10-5-2_corner_eps-2_min_sep-3_len_px-10-6_v5/heatmap_augmented/flip_v \
    --output_dir ./data/deventer_512/poly_gt_global_boundary/train/d-2_angle_tol_deg-10-5-2_corner_eps-2_min_sep-3_len_px-10-6_v5/heatmap_augmented/flip_v_vis_jet
    
    python ./tools/visualize_vertex_heatmaps.py \
    --input_dir ../../agricultural_parcel_extraction/AI4SmallFarms_split_0.05/train_vertex_heatmaps_sigma-3 \
    --output_dir ../../agricultural_parcel_extraction/AI4SmallFarms_split_0.05/train_vertex_heatmaps_sigma-3_vis_jet
    """
