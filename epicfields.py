import io
import os
import tarfile
from collections import defaultdict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from utils import ImageReader


def id2frame(x):
    return f"frame_{x:0>10}.jpg"


def split_exp(x):
    idx = x.find("-")
    vid = x[:idx]
    model = x[idx + 1 :]
    assert model in ["t-nerf", "nerf-w", "neuraldiff"]
    return vid, model


def frame2id(frame):
    return int("".join(list(filter(str.isdigit, frame))))


class EPICFields:
    def __init__(self, root):
        """
        root: 'data/annotations/epic-fields/'
        split: 'test' or 'val'
        motion: 'dynamic' or 'dynamic_semistatic'
        """
        self.root = root

        self.splits = {
            "test": pd.read_json("./split/test.json"),
            "val": pd.read_json("./split/val.json"),
        }
        self.imreader = {}

        self.cache_dir = os.path.join(self.root, "cache")

    def imread(self, vid, frame):

        pid = vid.split("_")[0]
        if os.path.isfile(os.path.join(self.cache_dir, vid, frame)):
            im = Image.open(os.path.join(self.cache_dir, vid, frame))
        else:
            src = f"ek100/{pid}/rgb_frames/{vid}.tar"

            if vid not in self.imreader:
                self.imreader[vid] = ImageReader(src)

            im = self.imreader[vid]["./" + frame]
            self.imreader[vid].save("./" + frame, os.path.join(self.cache_dir, vid))
        return np.array(im).copy()

    def load(self, vid, split, motion, frame):
        dir_frames = os.path.join(self.root, f"{split}_{motion}", vid)
        frame = frame.split(".")[0] + ".png"
        path = os.path.join(dir_frames, frame)
        im = np.array(Image.open(path)).astype(float)
        if im.max() == 255:
            im = im / 255

        mask_targ = im

        if motion == "dynamic":
            # includes dynamic objects only
            mask_valid = mask_targ
            mask_ignore_eval = np.zeros_like(mask_targ)
        elif motion == "semistatic":
            # includes semi static objects only
            # 2: during evaluation, we ignore semistatic objects that are ambiguous
            mask_valid = mask_targ == 1
            mask_ignore_eval = mask_targ == 2
        elif motion == "dynamic_semistatic":
            # includes dynamic and semistatic objects
            # 2: during evaluation, we ignore semistatic objects that are ambiguous
            # 3: body parts (belong to dynamic, but split in case someone is
            #    interested in evaluating without body parts) "dynamics" typically
            #    include moving objects and the person (constantly in motion)
            mask_valid = np.logical_or(mask_targ == 1, mask_targ == 3)
            mask_ignore_eval = mask_targ == 2

            # for visualisation, merge labels 1 and 3 to 1
            mask_targ[mask_valid] = 1
        else:
            input("Unknown motion label selected...")

        mask_valid = mask_valid.astype("bool")
        mask_ignore_eval = mask_ignore_eval.astype("bool")
        mask_eval = ~mask_ignore_eval

        return mask_targ, mask_valid, mask_eval


class CacheReader:

    def __init__(self, src_dir, src_tar, skip_cntd=True, new_frame_mapping=True):
        """
        src_dir: outputs/results
        src_tar: outputs/cache_nle/results_nle_neuraldiff.tar
        """

        exp2path = {}
        self.new_frame_mapping = new_frame_mapping
        self.tfile = {}

        if type(src_tar) != list:
            src_tar = [src_tar]

        if src_tar != [None]:
            print(src_tar, "here")

            for src_tar_ in src_tar:
                print(src_tar_)
                tfile = tarfile.TarFile(src_tar_)
                for x in [x for x in tfile.getnames() if "cache.pt" in x]:
                    if "cntd" in x and skip_cntd:
                        print("skipping")
                        continue
                    exp = "P" + x.split("/P")[1].split("/")[0]
                    assert exp not in exp2path
                    exp2path[exp] = "tar:" + x
                    self.tfile[exp] = tfile

        for x in glob(f"{src_dir}/**/*.pt", recursive=True):
            exp = x.split("/cache.pt")[0].split("/")[-1]
            exp2path[exp] = "dir:" + x

        self.exp2path = exp2path
        self.experiments = exp2path.keys()

        subset_vids = set([split_exp(x)[0] for x in self.exp2path])
        subset_model_types = set([split_exp(x)[1] for x in self.exp2path])
        exp2path_intersection = {}
        for vid in subset_vids:
            exps = []
            for model_type in subset_model_types:
                exp = f"{vid}-{model_type}"
                if exp in self.exp2path:
                    exps.append(exp)
            if len(exps) == len(subset_model_types):
                for exp in exps:
                    exp2path_intersection[exp] = self.exp2path[exp]

        self.exp2path_intersection = exp2path_intersection

        # only for cache mapping
        self.split_te_all = pd.read_json("split/test.json", orient="index")

    def load(self, exp):
        path_type = self.exp2path[exp][:4]
        path = self.exp2path[exp][4:]
        if path_type == "tar:":
            member = self.tfile[exp].getmember(path)
            cache_file = self.tfile[exp].extractfile(member)
            cache_bytes = io.BytesIO(cache_file.read())
            cache = torch.load(cache_bytes)
        elif path_type == "dir:":
            cache = torch.load(path)
        return cache

    def frame2cacheid_mapping(self, exp):

        if self.new_frame_mapping:
            results = self[exp]
            return {id2frame(x["sample_id"]): i for i, x in enumerate(results["out"])}

        vid = split_exp(exp)[0]

        split_te = self.split_te_all.loc[vid]
        frames_te = [k_ for k in split_te for k_ in k]
        frame2cacheid = dict([[frame, i] for i, frame in enumerate(sorted(frames_te))])

        return frame2cacheid

    def __getitem__(self, exp):
        return self.load(exp)


class CacheReader2D:
    def __init__(self, split, scores_type, model_type, vid_selected=None):
        """
        scores_type: soft, binary
        split: test, val
        """
        self.efields = EPICFields(root="annotations")
        self.valid_vids = []

        assert model_type in ["mg"]

        self.cache = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        src_dir = f"cache/task2-2d/{split}/soft_scores/"

        for x in glob(f"{src_dir}/**/*.png"):
            vid = x.split("frame_")[0][:-1].split("/")[-1]
            if vid_selected is not None and vid != vid_selected:
                continue
            self.valid_vids += [vid]
            exp = f"{vid}-{model_type}"
            frame_ = x.split(vid)[1][1:]
            mask_pred = np.array(Image.open(x))
            assert mask_pred.max() > 1
            self.cache[exp]["out"][frame_]["mask_pred"] = mask_pred / 255
            self.cache[exp]["out"][frame_]["mask_targ"] = self.efields.load(
                vid, split, "dynamic", frame_
            )
            im_targ = self.efields.imread(vid, frame_.replace("png", "jpg"))
            self.cache[exp]["out"][frame_]["im_targ"] = im_targ.astype(float) / 255

        class Lambda:
            def __getitem__(self, x):
                return x.replace("jpg", "png")

        self.frame2cacheid = Lambda()

    def __getitem__(self, exp):
        return self.cache[exp]

    def frame2cacheid_mapping(self, vid):
        return self.frame2cacheid
