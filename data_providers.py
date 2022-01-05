import os
import PIL
from PIL import Image
from tensorflow import keras
import numpy as np


def get_geometric_blur_patch(tensor_small, midpoint, patchsize, coeff):
    midpoint = midpoint // coeff
    hs = patchsize // 2
    hn = max(0, midpoint[0] - hs)
    hx = min(midpoint[0] + hs, tensor_small.size()[1] - 1)
    xn = max(0, midpoint[1] - hs)
    xx = min(midpoint[1] + hs, tensor_small.size()[2] - 1)

    p = tensor_small[:, hn:hx, xn:xx]
    if p.size()[1] != patchsize or p.size()[2] != patchsize:
        r = torch.zeros((3, patchsize, patchsize))
        r[:, 0:p.size()[1], 0:p.size()[2]] = p
        p = r
    return p


#####
# Default "patch" dataset, used for training
#####
class PatchedDataProvider(object):
    def __init__(self, x_dir, y_dir, patch_size):
        self.patch_size = patch_size

        self.paths_x = sorted(os.listdir(x_dir))
        self.paths_y = sorted(os.listdir(y_dir))

        # self.transform = build_transform()
        # self.mask_transform = build_mask_transform()

        self.images = []
        self.images_stylized = []

        for p in self.paths_x:
            if p == "Thumbs.db":
                continue

            p_png = os.path.splitext(p)[0] + '.png'
            image = PIL.Image.open(os.path.join(x_dir, p))
            image_stylized = PIL.Image.open(os.path.join(y_dir, p_png))

            image = keras.preprocessing.image.img_to_array(image)
            image_stylized = keras.preprocessing.image.img_to_array(image_stylized)

            self.images.append(image)
            self.images_stylized.append(image_stylized)

        self.valid_midpoints = []
        for im_index in range(len(self.images)):
            self.valid_midpoints.append([])
            for i in range(patch_size // 2, self.images[im_index].shape[0] - patch_size // 2):
                for j in range(patch_size // 2, self.images[im_index].shape[1] - patch_size // 2):
                    self.valid_midpoints[im_index].append((i, j))

        self.valid_indices = [list(range(len(self.valid_midpoints[im_index]))) for im_index in
                              range(len(self.images))]

    def cut_patch(self, im, midpoint, size):
        hs = size // 2
        hn = max(0, midpoint[0] - hs)
        hx = min(midpoint[0] + hs, im.shape[0] - 1)
        xn = max(0, midpoint[1] - hs)
        xx = min(midpoint[1] + hs, im.shape[1] - 1)

        p = im[hn:hx, xn:xx, :]
        if p.shape[0] != size or p.shape[1] != size:
            r = np.zeros((size, size, 3))
            r[0:p.shape[0], 0:p.shape[1], :] = p
            p = r

        return p

    def cut_patches(self, im_index, midpoint, midpoint_r, size):
        patch_input = self.cut_patch(self.images[im_index], midpoint, size)
        patch_stylized = self.cut_patch(self.images_stylized[im_index], midpoint, size)
        patch_random = self.cut_patch(self.images_stylized[im_index], midpoint_r, size)

        return patch_input, patch_stylized, patch_random

    def __getitem__(self, item):
        im_index = item % len(self.images)
        midpoint_id = int(np.random.randint(0, len(self.valid_indices[im_index])))
        midpoint_index = self.valid_indices[im_index][midpoint_id]
        midpoint_r_id = np.random.randint(0, len(self.valid_midpoints[im_index]))
        midpoint_r = self.valid_midpoints[im_index][midpoint_r_id]
        midpoint = self.valid_midpoints[im_index][midpoint_index]

        del self.valid_indices[im_index][midpoint_id]
        if len(self.valid_indices[im_index]) < 1:
            self.valid_indices[im_index] = list(range(0, len(self.valid_midpoints[im_index])))

        patch_pre, patch_post, patch_random = self.cut_patches(im_index, midpoint, midpoint_r, self.patch_size)

        return patch_pre, patch_post, patch_random

    def __len__(self):
        return sum([(len(n) // 2) for n in self.valid_midpoints]) * 5  # dont need to restart

    def get_full_image(self):
        return self.images[0]
