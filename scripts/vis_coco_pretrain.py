"""Visualise COCO pretrain synthetic pseudo-video sequences.

Generates a grid showing the reference frame + all query frames with their masks,
demonstrating coherent motion trajectories (progressive pan/rotation/zoom).
"""

import os
import torch
import matplotlib.pyplot as plt
from data.coco_pretrain import COCOPretrainDataset


def save_visualisation(output_dir="vis_coco_pretrain", seq_len=6, n_samples=3):
    os.makedirs(output_dir, exist_ok=True)

    dataset = COCOPretrainDataset(
        data_dir="./data/coco",
        img_size=448,
        seq_len=seq_len,
        n_id=10,
        split="validation",
        max_samples=50,
    )

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def to_img(t):
        t = t * std + mean
        return t.permute(1, 2, 0).numpy().clip(0, 1)

    cmap = plt.cm.get_cmap("tab20", 11)

    for sample_idx in range(n_samples):
        ref_img, ref_mask, q_imgs, q_masks, obj_present, meta = dataset[sample_idx]
        n_query = q_imgs.shape[0]
        n_cols = 1 + n_query  # ref + queries

        fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7))

        # Reference
        axes[0, 0].imshow(to_img(ref_img))
        axes[0, 0].set_title("Ref (t=0)", fontsize=10)
        axes[1, 0].imshow(ref_mask.numpy(), cmap=cmap, interpolation="nearest")
        axes[1, 0].set_title("Ref Mask", fontsize=10)

        # Query frames
        for i in range(n_query):
            axes[0, i + 1].imshow(to_img(q_imgs[i]))
            axes[0, i + 1].set_title(f"Query t={i+1}", fontsize=10)
            axes[1, i + 1].imshow(
                q_masks[i].numpy(), cmap=cmap, interpolation="nearest"
            )
            axes[1, i + 1].set_title(f"Mask t={i+1}", fontsize=10)

        for ax in axes.flatten():
            ax.axis("off")

        fig.suptitle(
            f"Coherent Motion Trajectory — {meta['video_id']}",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        fname = f"coco_pretrain_coherent_{sample_idx}.png"
        plt.savefig(
            os.path.join(output_dir, fname), bbox_inches="tight", dpi=150
        )
        plt.close(fig)
        print(f"Saved {output_dir}/{fname}")


if __name__ == "__main__":
    save_visualisation()
