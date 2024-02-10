import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from epicfields import CacheReader, CacheReader2D, EPICFields
from utils import calc_psnr


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dir_cache",
        type=str,
    )

    parser.add_argument(
        "--dir_output",
        type=str,
    )

    parser.add_argument(
        "--model",
        choices=["neuraldiff", "nerf-w", "t-nerf"],
        type=str,
    )

    parser.add_argument(
        "--split",
        default="test",
        type=str,
    )

    parser.add_argument(
        "--action_type",
        choices=["within_action", "outside_action", "outside_action_easy"],
        type=str,
    )

    # jupyter compatibility
    parser.add_argument(
        "-f",
        default="test",
        type=str,
    )

    args = parser.parse_args()

    return args


def compare_psnr(sample, mask_targ, psnr, psnr_f, psnr_b, action_type, vid, frame):
    f, ax = plt.subplots(1, 3, figsize=(8, 5))

    ax[0].imshow(sample["im_targ"])
    ax[0].set_title("GT", fontsize=10)
    ax[1].imshow(sample["im_pred"])
    ax[1].set_title(f"Prediction. PSNR: {psnr:.2f}", fontsize=10)
    ax[2].imshow(mask_targ, cmap="gray")
    ax[2].set_title(f"PSNR_f/b: {psnr_f:.2f}/{psnr_b:.2f}", fontsize=10)
    for i in range(3):
        ax[i].axis("off")
    plt.tight_layout()
    plt.suptitle(f"{vid}: {frame} - Action: {action_type}", x=0.51, y=0.73, fontsize=10)
    return f


def task_1(
    efields, dir_output, model_type, action_type, dir_cache, split="test", creader=None
):

    if creader is None:
        creader = CacheReader(
            src_dir=dir_cache,
            src_tar=None,
        )

    motion = "dynamic"
    dir_results = f"{dir_output}/{split}/{action_type}"
    valid_vids = set(
        [x.split("-")[0] for x in creader.exp2path.keys() if model_type in x]
    )

    mean_psnr_scores = {}
    mean_psnr_f_scores = {}
    mean_psnr_b_scores = {}

    for vid in tqdm(valid_vids):

        exp = f"{vid}-{model_type}"

        frame2cacheid = creader.frame2cacheid_mapping(exp)
        results = creader[exp]
        psnr_scores = []
        psnr_f_scores = []
        psnr_b_scores = []

        dir_results_per_exp = os.path.join(dir_results, model_type, vid)
        os.makedirs(dir_results_per_exp, exist_ok=True)

        frames = efields.splits[split][vid][action_type]

        if len(frames) == 0:
            continue

        for frame in frames:
            sample = results["out"][frame2cacheid[frame]]

            im_targ = np.asarray(sample["im_targ"])
            if action_type in ["outside_action_easy", "outside_action"]:
                mask_targ = np.random.randn(*list(im_targ.shape[:2]))
            elif action_type == "within_action":
                mask_targ = efields.load(vid, split, motion, frame)
            if action_type == "within_action":
                if mask_targ.sum() == 0:
                    continue

            mask_f = mask_targ.astype("bool")
            mask_b = ~mask_f

            if True:
                sample["im_pred"] = sample["im_pred"].float().clip(0, 1)
                sample["im_targ"] = sample["im_targ"].float().clip(0, 1)
            else:
                sample["im_pred"] = (
                    np.asarray(sample["im_pred"]).astype(float) / 255
                ).clip(0, 1)
                sample["im_targ"] = (
                    np.asarray(sample["im_targ"]).astype(float) / 255
                ).clip(0, 1)

            psnr = calc_psnr(sample["im_pred"], sample["im_targ"])
            psnr_f = calc_psnr(sample["im_pred"][mask_f], sample["im_targ"][mask_f])
            psnr_b = calc_psnr(sample["im_pred"][mask_b], sample["im_targ"][mask_b])

            psnr_scores.append(psnr)
            psnr_f_scores.append(psnr_f)
            psnr_b_scores.append(psnr_b)

            f = compare_psnr(
                sample, mask_targ, psnr, psnr_f, psnr_b, action_type, vid, frame
            )
            f.savefig(
                os.path.join(dir_results_per_exp, frame),
                bbox_inches="tight",
                pad_inches=0.1,
            )
            plt.close()

        mean_psnr_scores[vid] = np.mean(psnr_scores)
        mean_psnr_f_scores[vid] = np.mean(psnr_f_scores)
        mean_psnr_b_scores[vid] = np.mean(psnr_b_scores)

    dst_results = os.path.join(dir_results, model_type, "results.txt")
    with open(dst_results, mode="w") as f:
        f.write(f"PSNR \t {np.mean(list(mean_psnr_scores.values())):.2f}\n")
        if action_type == "within_action":
            f.write(f"PSNR_f \t {np.mean(list(mean_psnr_f_scores.values())):.2f}\n")
            f.write(f"PSNR_b \t {np.mean(list(mean_psnr_b_scores.values())):.2f}\n")
        else:
            f.write(f"PSNR_f \t -----\n")
            f.write(f"PSNR_b \t -----\n")


def main():

    args = parse_args()
    efields = EPICFields(root="annotations")
    task_1(
        efields,
        args.dir_output,
        args.model,
        args.action_type,
        args.dir_cache,
        split="test",
        creader=None,
    )


if __name__ == "__main__":
    main()
