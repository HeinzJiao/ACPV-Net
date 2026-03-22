from ldm.models.diffusion.ddpm_seg_vertex import LatentDiffusion
import torch
import tqdm
from tqdm import tqdm
from scipy.ndimage import zoom
from einops import rearrange, repeat
import numpy as np
import os
from torch import autocast
from scripts.slice2seg import prepare_for_first_stage, dice_score, iou_score
from PIL import Image
from torch.utils.data import DataLoader
from scipy.spatial import cKDTree
import torch.nn.functional as F
import cv2
import json
import torchvision
from torchvision.utils import make_grid
from scripts.extract_vertices_from_heatmap import extract_vertices_from_heatmap

class ExtendedLatentDiffusion(LatentDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def log_images_loop(
            self,
            data,
            save_results,
            save_dir,
            used_sampler="ddpm",
            ddim_steps=20,
            save_samples_seg_logits_npy=True,
            save_samples_heat_logits_npy=True,
    ):
        """
        Run inference over a dataloader, optionally compute metrics, and save outputs.

        This function supports both:
            - label-available evaluation (segmentation + vertex metrics), and
            - image-only generalization testing (no GT vertices / masks).

        If no ground-truth vertices are present, vertex precision/recall are skipped.
        """

        # Initialize accumulators
        num_fg_classes = self.num_classes - 1  # assuming class 0 = background
        dice_sum = np.zeros(num_fg_classes, dtype=np.float64)
        iou_sum = np.zeros(num_fg_classes, dtype=np.float64)
        precision_sum = np.zeros(num_fg_classes, dtype=np.float64)
        recall_sum = np.zeros(num_fg_classes, dtype=np.float64)

        num_seg_batches = 0
        num_vertex_batches = 0

        pbar = tqdm(data, desc="Validating Segmentation")
        for batch_idx, batch in enumerate(pbar):
            image_file_name = batch["file_path_"][0].split("/")[-1]

            # Ground-truth segmentation (may be dummy zeros in image-only mode)
            label = batch.get("segmentation", None)

            # Ground-truth vertices (may be absent or empty in image-only mode)
            has_vertex_gt = (
                    "vertex_locations" in batch
                    and batch["vertex_locations"] is not None
                    and batch["vertex_locations"].numel() > 0
            )

            # ------------------------------------------------------------------
            # Forward pass & sampling
            # ------------------------------------------------------------------
            images = self.log_images(
                batch,
                sampler=used_sampler,
                ddim_steps=ddim_steps,
                plot_denoise_rows=False,
                plot_diffusion_rows=False,
                return_first_stage_outputs=False,
                plot_conditioning_latent=False,
            )
            for k in images:
                N = min(images[k].shape[0], 4)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    images[k] = torch.clamp(images[k], -1.0, 1.0)

            # ------------------------------------------------------------------
            # Reconstructed segmentation probability map & mask
            # ------------------------------------------------------------------
            x_samples_ddpm = images[f"samples_seg_{used_sampler}"]  # [-1, 1]
            x_samples_ddpm = (x_samples_ddpm + 1.0) / 2.0  # [0, 1]
            x_samples_ddpm = x_samples_ddpm.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            x_out_p = rearrange(x_samples_ddpm.squeeze(0).numpy(), "c h w -> h w c")  # HxWx3, probability map
            x_out = (x_out_p > 0.5)
            x_prediction = x_out[:, :, 0]  # HxW, binary 0/1, mask

            # Save segmentation probability map as .npy
            if save_samples_seg_logits_npy:
                os.makedirs(
                    os.path.join(save_dir, f"samples_seg_{used_sampler}_logits_npy"),
                    exist_ok=True,
                )
                path = os.path.join(
                    save_dir,
                    f"samples_seg_{used_sampler}_logits_npy",
                    ".".join([image_file_name.split(".")[0], "npy"]),
                )
                np.save(path, x_out_p[:, :, 0].astype(np.float32))

            # ------------------------------------------------------------------
            # Reconstructed vertex heatmap
            # ------------------------------------------------------------------
            h_samples_ddpm = images[f"samples_heat_{used_sampler}"]  # [-1, 1]
            h_samples_ddpm = (h_samples_ddpm + 1.0) / 2.0  # [0, 1]
            h_samples_ddpm = h_samples_ddpm.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            h_out_p = rearrange(h_samples_ddpm.squeeze(0).numpy(), "c h w -> h w c")  # HxWx3
            h_prediction = h_out_p[:, :, 0]

            # Save vertex heatmap as .npy
            if save_samples_heat_logits_npy:
                os.makedirs(
                    os.path.join(save_dir, f"samples_heat_{used_sampler}_npy"),
                    exist_ok=True,
                )
                path = os.path.join(
                    save_dir,
                    f"samples_heat_{used_sampler}_npy",
                    ".".join([image_file_name.split(".")[0], "npy"]),
                )
                np.save(path, h_out_p[:, :, 0].astype(np.float32))

            # ------------------------------------------------------------------
            # Segmentation metrics: Dice & IoU (only if label is provided)
            # ------------------------------------------------------------------
            if label is not None:
                lbl = label.squeeze(0).numpy()
                lbl = lbl[:, :, 0]
                lbl = lbl.round().astype(int)

                metrics_list_x = [[], []]  # dice, iou
                for cls_idx in range(1, self.num_classes):
                    pred_mask = (x_prediction == cls_idx)
                    gt_mask = (lbl == cls_idx)
                    metrics_list_x[0].append(dice_score(pred_mask, gt_mask))
                    metrics_list_x[1].append(iou_score(pred_mask, gt_mask))

                dice_sum += np.array(metrics_list_x[0], dtype=np.float64)
                iou_sum += np.array(metrics_list_x[1], dtype=np.float64)
                num_seg_batches += 1

            # ------------------------------------------------------------------
            # Vertex metrics: precision & recall (only if GT vertices exist)
            # ------------------------------------------------------------------
            if has_vertex_gt:
                v_locs = batch["vertex_locations"].squeeze(0).numpy()  # Nx2, int32
                metrics_list_h = [[], []]  # precision, recall
                th = 0.1

                for cls_idx in range(1, self.num_classes):
                    extracted_vertices, scores = extract_vertices_from_heatmap(
                        h_prediction, th, kernel_size=3, topk=300
                    )
                    precision, recall = calculate_precision_recall(extracted_vertices, v_locs)
                    metrics_list_h[0].append(precision)
                    metrics_list_h[1].append(recall)

                precision_sum += np.array(metrics_list_h[0], dtype=np.float64)
                recall_sum += np.array(metrics_list_h[1], dtype=np.float64)
                num_vertex_batches += 1

            # ------------------------------------------------------------------
            # Save visualizations
            # ------------------------------------------------------------------
            if save_results:
                self.log_local(save_dir, images, image_file_name)

        pbar.close()

        # ----------------------------------------------------------------------
        # Aggregate metrics and print
        # ----------------------------------------------------------------------
        if num_seg_batches > 0:
            avg_dice = dice_sum / num_seg_batches
            avg_iou = iou_sum / num_seg_batches
            for cls_idx in range(1, self.num_classes):
                print(f"\033[31m[Mean Dice][cls {cls_idx}]: {avg_dice[cls_idx - 1]}\033[0m")
            for cls_idx in range(1, self.num_classes):
                print(f"\033[31m[Mean  IoU][cls {cls_idx}]: {avg_iou[cls_idx - 1]}\033[0m")
        else:
            avg_dice = None
            avg_iou = None
            print("\033[31m[Info]: No ground-truth segmentation available; Dice/IoU not computed.\033[0m")

        if num_vertex_batches > 0:
            avg_precision = precision_sum / num_vertex_batches
            avg_recall = recall_sum / num_vertex_batches
            for cls_idx in range(1, self.num_classes):
                print(f"\033[31m[Mean Precision][cls {cls_idx}]: {avg_precision[cls_idx - 1]}\033[0m")
            for cls_idx in range(1, self.num_classes):
                print(f"\033[31m[Mean  Recall][cls {cls_idx}]: {avg_recall[cls_idx - 1]}\033[0m")
        else:
            avg_precision = None
            avg_recall = None
            print("\033[31m[Info]: No ground-truth vertices available; Precision/Recall not computed.\033[0m")

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4,
                   sampler="ddpm", ddim_steps=20, ddim_eta=1.0,
                   plot_denoise_rows=False, plot_diffusion_rows=False, return_first_stage_outputs=False,
                   plot_conditioning_latent=False,
                   **kwargs,
    ):
        """
        Generate visualization tensors for logging / inspection.

        This function:
            1) Encodes segmentation mask + vertex heatmap into latent space;
            2) Runs one of {direct, DDIM, DDPM} samplers (seg + heat jointly);
            3) Decodes final latents back to pixel space;
            4) Optionally:
               - logs autoencoder reconstructions,
               - logs intermediate diffusion steps (forward / reverse),
               - logs conditioning latents.

        Returned keys (subset, depending on flags and sampler):
            - "ground_truth_heat"                                  : input vertex heatmap (pre-encoded)
            - "reconstruction_seg", "reconstruction_heat"
            - "conditioning_latent"
            - "diffusion_row_*" / "diffusion_row_latent_*"
            - "samples_seg_<sampler>", "samples_heat_<sampler>"
            - "samples_latent_seg_<sampler>", "samples_latent_heat_<sampler>"
            - "denoise_row_ddim"
            - "progressive_row_*_ddpm"

        Args:
            batch (dict): Batch from the dataloader (must contain "image", "segmentation", "heatmap").
            N (int): Maximum batch size to visualize.
            n_row (int): Number of rows for grid visualizations.
            sampler (str): "direct", "ddim", or "ddpm".
            ddim_steps (int): DDIM steps (if sampler == "ddim").
            ddim_eta (float): DDIM eta (0.0 → deterministic).
            plot_denoise_rows (bool): If True, visualize reverse-time denoising trajectory.
            plot_diffusion_rows (bool): If True, visualize forward diffusion (q-sample) trajectory.
            return_first_stage_outputs (bool): If True, log autoencoder reconstructions.
            plot_conditioning_latent (bool): If True, log conditioning latent.

        Returns:
            dict: A dictionary of tensors suitable for logging (e.g. in TensorBoard / WandB).
        """
        log = {}

        # ------------------------------------------------------------------
        # 1) Prepare inputs and encode to latent space
        # ------------------------------------------------------------------
        # Make sure segmentation & heatmap are in [-1, 1] (same range as training)
        batch["segmentation"] = batch["segmentation"] * 2.0 - 1.0
        batch["heatmap"] = batch["heatmap"] * 2.0 - 1.0

        if return_first_stage_outputs:
            # z, zh:   latents for seg / vertex heatmap
            # c:       image condition (already encoded), range from -1 to 1
            # x, h:    original seg / heatmap in pixel space, range from -1 to 1
            # xrec, hrec: autoencoder reconstructions
            # xc: input image
            z, zh, c, x, h, cls_id, xrec, hrec, xc = self.get_input(
                batch,
                self.first_stage_key,
                return_first_stage_outputs=True,
                force_c_encode=True,
                return_original_cond=False,
                bs=N,
            )
        else:
            z, zh, c, x, h, cls_id = self.get_input(
                batch,
                self.first_stage_key,
                return_first_stage_outputs=False,
                force_c_encode=True,
                return_original_cond=False,
                bs=N,
            )

        # ------------------------------------------------------------------
        # 2) Build conditioning dictionary
        # ------------------------------------------------------------------
        if self.model.conditioning_key == "concat":
            c = {"c_concat": [c]}
            cond_key = "c_concat"
        elif self.model.conditioning_key == "crossattn":
            c = {"c_crossattn": [c]}
            cond_key = "c_crossattn"
        else:  # hybrid
            c = {"c_concat": [c], "c_crossattn": [c]}
            cond_key = "c_crossattn"

        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)

        # For reference: input vertex heatmap (after preprocessing)
        log["ground_truth_heat"] = h

        # Optionally log autoencoder reconstructions
        if return_first_stage_outputs:
            log["reconstruction_seg"] = xrec
            log["reconstruction_heat"] = hrec

        # Optionally log conditioning latent (image condition)
        if plot_conditioning_latent and self.model.conditioning_key is not None:
            log["conditioning_latent"] = self.prepare_latent_to_log(c[cond_key][0])

        # ------------------------------------------------------------------
        # 3) Forward diffusion visualization (q-sample)
        # ------------------------------------------------------------------
        if plot_diffusion_rows:
            diff_seg, diff_latent_seg = self.plot_diffusion_rows(z, n_row, ztype="segmentation")
            log["diffusion_row_seg"] = diff_seg
            log["diffusion_row_latent_seg"] = diff_latent_seg

            diff_heat, diff_latent_heat = self.plot_diffusion_rows(zh, n_row, ztype="heatmap")
            log["diffusion_row_heat"] = diff_heat
            log["diffusion_row_latent_heat"] = diff_latent_heat

        # ------------------------------------------------------------------
        # 4) Sampling: direct / DDIM / DDPM
        # ------------------------------------------------------------------

        # 4.1) "direct" mode: take model output at final timestep as x0 / h0
        if sampler == "direct":
            with self.ema_scope():
                noise_x = torch.randn_like(z)  # x_T
                noise_h = torch.randn_like(z)  # h_T (same shape as z)
                final_t = torch.tensor([self.num_timesteps - 1], device=self.device).long()

                model_output = self.apply_model(noise_x, noise_h, final_t, c)
                _, d, _, _ = model_output.shape
                split_size = d // 2

                # First half: seg branch; second half: heatmap branch
                eps_x = model_output[:, :split_size, :, :]
                eps_h = model_output[:, split_size:, :, :]

                samples_z = self.predict_start_from_noise(noise_x, final_t, noise=eps_x)  # seg latent x0
                samples_zh = self.predict_start_from_noise(noise_h, final_t, noise=eps_h)  # heat latent h0

            x_samples = self.decode_first_stage(samples_z, ztype="segmentation")
            h_samples = self.decode_first_stage(samples_zh, ztype="heatmap")

            log["samples_seg_direct"] = x_samples
            log["samples_heat_direct"] = h_samples
            log["samples_latent_seg_direct"] = samples_z
            log["samples_latent_heat_direct"] = samples_zh

        # 4.2) DDIM sampling
        elif sampler == "ddim":
            with self.ema_scope("Plotting"):
                samples_z, samples_zh, z_denoise_row = self.sample_log(
                    cond=c,
                    batch_size=N,
                    ddim=True,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                )
                # z_denoise_row contains:
                #   - "x_inter": noisy latents for seg at intermediate steps
                #   - "pred_x0": predicted x0 for seg
                #   - "h_inter": noisy latents for heatmap
                #   - "pred_h0": predicted h0 for heatmap

            x_samples = self.decode_first_stage(samples_z, ztype="segmentation")
            h_samples = self.decode_first_stage(samples_zh, ztype="heatmap")

            log["samples_seg_ddim"] = x_samples
            log["samples_heat_ddim"] = h_samples
            log["samples_latent_seg_ddim"] = samples_z
            log["samples_latent_heat_ddim"] = samples_zh

            if plot_denoise_rows:
                denoise_grid = self.get_denoise_row_from_list(z_denoise_row)
                log["denoise_row_ddim"] = denoise_grid

        # 4.3) DDPM sampling (progressive denoising)
        elif sampler == "ddpm":
            with self.ema_scope("Plotting Progressives"):
                # img, img_h: final latents for seg / heat at t=0
                # progressives, progressives_h: sequences of partial x0 predictions
                img_z, img_zh, progressives, progressives_h = self.progressive_denoising(
                    c,
                    shape=(self.channels, self.image_size, self.image_size),
                    batch_size=N,
                )

            # seg branch
            prog_row_seg, prog_row_latent_seg = self.get_denoise_row_from_list(
                progressives,
                desc="Progressive Generation",
                ztype="segmentation",
            )
            log["progressive_row_seg_ddpm"] = prog_row_seg
            log["progressive_row_latent_seg_ddpm"] = prog_row_latent_seg

            x_samples = self.decode_first_stage(img_z, ztype="segmentation")
            h_samples = self.decode_first_stage(img_zh, ztype="heatmap")

            log["samples_seg_ddpm"] = x_samples
            log["samples_heat_ddpm"] = h_samples

            # heat branch
            prog_row_heat, prog_row_latent_heat = self.get_denoise_row_from_list(
                progressives_h,
                desc="Progressive Generation",
                ztype="heatmap",
            )
            log["progressive_row_heat_ddpm"] = prog_row_heat
            log["progressive_row_latent_heat_ddpm"] = prog_row_latent_heat

        else:
            raise ValueError(f"Unknown sampler type: {sampler}")

        return log

    def plot_diffusion_rows(self, z, n_row: int, ztype: str):
        """
        Visualize the *forward* diffusion process (q_sample) at a few timesteps.

        For a given latent batch z, this function:
            1) Takes the first `n_row` samples as starting latents;
            2) For t in {0, log_every_t, 2*log_every_t, ..., T-1}:
                 - samples a noisy latent z_t ~ q(z_t | z_0, t);
                 - decodes z_t back to pixel space;
            3) Stacks them into a single grid for logging.

        Args:
            z (Tensor): Latent tensor of shape [B, C, H, W].
            n_row (int): Number of examples (rows) to visualize.
            ztype (str): "segmentation" or "heatmap" (controls decoding branch).

        Returns:
            diffusion_grid (Tensor): Grid of decoded images across timesteps.
            diffusion_grid_latent (Tensor): Grid of noisy latents across timesteps.
        """
        diffusion_row = []
        diffusion_row_latent = []

        # Use only the first n_row samples for visualization
        z_start = z[:n_row]

        for t in range(self.num_timesteps):
            # Log only at a fixed interval and the final step
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t_batch = repeat(torch.tensor([t]), "1 -> b", b=n_row).to(self.device).long()
                noise = torch.randn_like(z_start)

                # Forward diffusion: q(z_t | z_0, t)
                z_noisy = self.q_sample(x_start=z_start, t=t_batch, noise=noise)

                diffusion_row_latent.append(z_noisy)
                diffusion_row.append(self.decode_first_stage(z_noisy, ztype=ztype))

        # Stack decoded images: [n_log, n_row, C, H, W] → grid
        diffusion_row = torch.stack(diffusion_row)  # [n_log, n_row, C, H, W]
        diffusion_grid = rearrange(diffusion_row, "n b c h w -> (b n) c h w")
        diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])

        # Stack noisy latents: same arrangement, but in latent space
        diffusion_row_latent = torch.stack(diffusion_row_latent)  # [n_log, n_row, C, H, W]
        diffusion_grid_latent = rearrange(diffusion_row_latent, "n b c h w -> (b n) c h w")
        diffusion_grid_latent = make_grid(diffusion_grid_latent, nrow=diffusion_row_latent.shape[0])

        return diffusion_grid, diffusion_grid_latent

    def log_local(self, save_dir, images, image_file_name):
        """
        Save image grids for each entry in `images` to disk.

        Args:
            save_dir (str): Root directory where visualizations are stored.
            images (dict): Dict of tensors, each of shape [B, C, H, W] and in [-1, 1].
            image_file_name (str): Original filename (e.g. "xxx.png" or "xxx.jpg");
                                   only the stem is kept for saving.
        """
        root = save_dir
        stem = os.path.splitext(image_file_name)[0]

        for k, img_batch in images.items():
            # 1) Make grid: (B, C, H, W) -> (C, H_grid, W_grid)
            grid = torchvision.utils.make_grid(img_batch, nrow=4)

            # 2) Rescale from [-1, 1] to [0, 1]
            grid = torch.clamp(grid, -1.0, 1.0)
            grid = (grid + 1.0) / 2.0

            # 3) Convert to HWC uint8
            grid = grid.detach().cpu().numpy()  # C, H, W
            grid = np.transpose(grid, (1, 2, 0))  # H, W, C
            grid = (grid * 255.0).astype(np.uint8)

            # 4) Save to `<root>/<k>/<stem>.png`
            out_dir = os.path.join(root, k)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{stem}.png")
            Image.fromarray(grid).save(out_path)


# different from RoomFormer/s3d_floorplan_eval/Evaluator/Evaluator.py
def calculate_precision_recall(pred_coords, gt_coords, threshold=10):
    # 如果pred_coords和gt_coords都为空，表示没有预测和真值，返回完美结果
    if len(pred_coords) == 0 and len(gt_coords) == 0:
        return 1.0, 1.0  # 完美的Precision和Recall

    # 如果其中一个为空，返回Precision或Recall为0
    if len(pred_coords) == 0:
        return 0.0, 0.0  # 没有预测顶点
    if len(gt_coords) == 0:
        return 0.0, 0.0  # 没有真值顶点

    # 使用cKDTree来加速最近邻搜索
    gt_tree = cKDTree(gt_coords)
    pred_tree = cKDTree(pred_coords)

    # 查找每个pred坐标的最近的gt坐标
    pred_distances, _ = gt_tree.query(pred_coords, distance_upper_bound=threshold)
    # 查找每个gt坐标的最近的pred坐标
    gt_distances, _ = pred_tree.query(gt_coords, distance_upper_bound=threshold)

    # True Positives (TP) - gt点与pred点的最近距离在阈值以内 (pred点中与gt匹配上的点)
    TP = np.sum(pred_distances <= threshold)

    # False Positives (FP) - pred点中没有匹配到任何gt点的点
    FP = np.sum(pred_distances > threshold)

    # False Negatives (FN) - gt点中没有匹配到任何pred点的点
    FN = np.sum(gt_distances > threshold)

    # 计算Precision和Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return precision, recall
