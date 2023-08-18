
import torch
import matplotlib.pyplot as plt
import shutil
import tarfile
import cv2 as cv
import numpy as np
import io
import os


def blend_mask(im, mask, colour=[1, 0, 0], alpha=0.5, show_im=False):
    """Blend an image with a mask (colourised via `colour` and `alpha`)."""
    if type(im) == torch.Tensor:
        im = im.numpy()
    im = im.copy()
    if im.max() > 1:
        im = im.astype(np.float) / 255
    for ch, rgb_v in zip([0, 1, 2], colour):
        im[:, :, ch][mask == 1] = im[:, :, ch][mask == 1] * (1 - alpha) + rgb_v * alpha
    if show_im:
        plt.imshow(im)
        plt.axis("off")
        plt.show()
    return im


def mse(image_pred, image_gt):
    if type(image_pred) != torch.Tensor:
        image_pred = torch.Tensor(image_pred)
    if type(image_gt) != torch.Tensor:
        image_gt = torch.Tensor(image_gt)
    value = (image_pred - image_gt) ** 2
    return torch.mean(value)


def calc_psnr(image_pred, image_gt):
    return -10 * torch.log10(mse(image_pred, image_gt))


def tar2bytearr(tar_member):
    return np.asarray(
        bytearray(
            tar_member.read()
        ),
        dtype=np.uint8
    )


class ImageReader:
    def __init__(self, src, scale=1, cv_flag=cv.IMREAD_UNCHANGED):
        # src can be directory or tar file

        self.scale = 1
        self.cv_flag = cv_flag

        if os.path.isdir(src):
            self.src_type = 'dir'
            self.fpaths = sorted(glob(os.path.join(src, '*')))
        elif os.path.isfile(src) and os.path.splitext(src)[1] == '.tar':
            self.tar = tarfile.open(src)
            self.src_type = 'tar'
            self.fpaths = sorted([x for x in self.tar.getnames() if 'frame_' in x and '.jpg' in x])
        else:
            print('Source has unknown format.')
            exit()

    def __getitem__(self, k):
        if self.src_type == 'dir':
            im = cv.imread(k, self.cv_flag)
        elif self.src_type == 'tar':
            member = self.tar.getmember(k)
            tarfile = self.tar.extractfile(member)
            byte_array = tar2bytearr(tarfile)
            im = cv.imdecode(byte_array, self.cv_flag)
        if self.scale != 1:
            im = cv.resize(
                im, dsize=[im.shape[0] // self.scale, im.shape[1] // self.scale]
            )
        if self.cv_flag != cv.IMREAD_GRAYSCALE:
            im = im[..., [2, 1, 0]]
        return im

    def save(self, k, dst):
        fn = os.path.split(k)[-1]
        if self.src_type == 'dir':
            shutil.copy(fn, os.path.join(dst, fn))
        elif self.src_type == 'tar':
            self.tar.extract(self.tar.getmember(k), dst)
