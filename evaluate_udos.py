import argparse

from epicfields import CacheReader, CacheReader2D, EPICFields

from sklearn.metrics import average_precision_score
import torch
import os
from utils import blend_mask
import numpy as np
import matplotlib.pyplot as plt


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dir_cache",
        type=str,
    )

    parser.add_argument(
        '--dir_output',
        type=str,
    )

    parser.add_argument(
        '--vid',
        type=str,
    )

    parser.add_argument(
        '--model',
        choices=['neuraldiff', 'nerf-w', 't-nerf', 'mg'],
        type=str,
    )

    parser.add_argument(
        '--split',
        default='test',
        type=str,
    )

    # jupyter compatibility
    parser.add_argument(
        '-f',
        default='test',
        type=str,
    )

    args = parser.parse_args()

    return args


def calc_ap(mask_targ, mask_pred):
    mask_pred = np.asarray(mask_pred)
    average_precision = 100 * average_precision_score(
        mask_targ.reshape(-1), mask_pred.reshape(-1)
    )
    return average_precision


def compare_ap(vid, frame, im_targ, mask_targ, mask_pred, ap_score):
    f, ax = plt.subplots(1, 2)
    plt.tight_layout()
    plt.suptitle(f'{vid}: {frame} - AP: {ap_score:.2f}', x=0.51, y=0.73, fontsize=10)
    ax[0].imshow(blend_mask(im_targ, mask_targ))
    ax[0].axis('off')
    ax[1].imshow(blend_mask(gray2threech(mask_pred), mask_targ, alpha=0.1))
    ax[1].axis('off')
    return f


def gray2threech(x):
    return np.tile(x[:, :, None], [1, 1, 3])


def task_2(dir_outputs, split, motion, model_type, action_type, vid, efields, CacheReader, CacheReader2D, dir_cache, is_debug=False):

    dir_results = f'{dir_outputs}/{split}/{motion}/{action_type}/{model_type}'
    if model_type == 'mg':
        creader = CacheReader2D(split, 'soft', 'mg', is_debug=False, vid_selected=vid)
    else:
        creader = CacheReader(
            src_dir=dir_cache,
            src_tar=None
        )


    exp = f'{vid}-{model_type}'

    results = creader[exp]
    frame2cacheid = creader.frame2cacheid_mapping(exp)

    frame2score = {}
    ap_scores = []

    samples = []
    masks = []
    ims = []

    dir_results_per_exp = os.path.join(dir_results, vid)
    os.makedirs(dir_results_per_exp, exist_ok=True)

    if is_debug:
        each_nth = 10
    else:
        each_nth = 1

    for frame in efields.splits[split][vid][action_type][::each_nth]:
        sample = results['out'][frame2cacheid[frame]]

        im_targ = np.asarray(sample['im_targ'])

        if action_type == 'within_action':
            mask_targ = efields.load(vid, split, motion, frame)

            if motion in ['dynamic', 'dynamic_no_body_parts', 'body_parts_only']:
                # t-nerf and nerf-w can only segment all dynamic (with semi-static)
                if model_type in ['t-nerf', 'nerf-w', 'mg']:
                    mask_pred = sample['mask_pred']
                else:
                    mask_pred = sample['mask_pers']
            elif motion in ['dynamic_semistatic', 'dynamic_semistatic_no_body_parts']:
                mask_pred = sample['mask_pred']
            else:
                del mask_pred
        else:
            im_pred = np.zeros_like(im_targ).copy()
            mask_targ = np.ones(im_targ.shape[:2])

            # ignore empty masks
            if mask_targ.sum() == 0:
                continue

        if mask_pred.sum() == 0:
            ap_score = 0
        else:
            ap_score = calc_ap(mask_targ, mask_pred)
        ap_scores.append(ap_score)
        frame2score[frame] = ap_score

        if type(mask_pred) != np.ndarray:
            mask_pred = mask_pred.float()
        if im_targ.dtype == np.float16:
            im_targ = im_targ.astype(float)

        f = compare_ap(vid, frame, im_targ, mask_targ, mask_pred, ap_score)
        f.savefig(os.path.join(dir_results_per_exp, frame), bbox_inches = 'tight', pad_inches = 0.1)
        if 'im_pred' not in sample:
            im_pred = np.zeros_like(im_targ)
        else:
            im_pred = sample['im_pred'].numpy()
        frame_name_base = frame.split('.jpg')[0]
        frame_name_pred = frame_name_base + '-pred.jpg'
        frame_name_targ = frame_name_base + '-targ.jpg'
        plt.imsave(os.path.join(dir_results_per_exp, frame_name_pred), im_pred.clip(0, 1))
        plt.imsave(os.path.join(dir_results_per_exp, frame_name_targ), im_targ)
        plt.imsave(os.path.join(dir_results_per_exp, frame_name_base + '-maskgt.jpg'), blend_mask(im_targ, mask_targ))
        plt.imsave(os.path.join(dir_results_per_exp, frame_name_base + '-maskpred.jpg'), mask_pred, cmap='gray')
        plt.close()

        samples.append(sample)
        masks.append(mask_targ)
        ims.append(im_targ)

    return np.mean(ap_scores), frame2score, samples, masks, ims, dir_results_per_exp


def main(args):
    motion_types = ['dynamic_semistatic', 'dynamic_semistatic_no_body_parts', 'dynamic']
    action_type = "within_action"
    efields = EPICFields(root='annotations')

    for motion in motion_types:

        mAP, frame2score, samples, masks, ims, dir_results_exp = task_2(
            args.dir_output, args.split, motion,
            args.model, action_type, args.vid, efields, CacheReader,
            CacheReader2D, args.dir_cache,
            is_debug=False
        )

        predictions = [samples[i]['mask_pred'] for i in range(len(samples))]
        ground_truths = masks

        torch.save(
            {
                'frame2score': frame2score,
                'map': mAP,
            },
            os.path.join(dir_results_exp, 'results.pt')
        )


if __name__ == '__main__':

    args = parse_args()
    main(args)
    print('done')
