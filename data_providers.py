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

    def cut_patch(self, im, midpoint, size):
        hs = size // 2
        hn = max(0, midpoint[0] - hs)
        hx = min(midpoint[0] + hs, im.size()[1] - 1)
        xn = max(0, midpoint[1] - hs)
        xx = min(midpoint[1] + hs, im.size()[2] - 1)

        p = im[:, hn:hx, xn:xx]
        if p.size()[1] != size or p.size()[2] != size:
            r = np.zeros((3, size, size))
            r[:, 0:p.size()[1], 0:p.size()[2]] = p
            p = r

        return p


    def cut_patches(self, im_index, midpoint, midpoint_r, size):
        patch_input = self.cut_patch(self.images[im_index], midpoint, size)

        patch_stylized = self.cut_patch(self.images_stylized[im_index], midpoint, size)
        patch_random = self.cut_patch(self.images_stylized[im_index], midpoint_r, size)

        return patch_input, patch_stylized, patch_random

    def __getitem__(self, item):
        im_index = item % len(self.images)
        midpoint_id = np.random.randint(0, len(self.valid_indices_left[im_index]))
        midpoint_r_id = np.random.randint(0, len(self.valid_indices[im_index]))
        midpoint = self.valid_indices[im_index][self.valid_indices_left[im_index][midpoint_id], :].squeeze()
        midpoint_r = self.valid_indices[im_index][midpoint_r_id, :].squeeze()

        del self.valid_indices_left[im_index][midpoint_id]
        if len(self.valid_indices_left[im_index]) < 1:
            self.valid_indices_left[im_index] = list(range(0, len(self.valid_indices[im_index])))

        result = {}

        for i in range(0, 1):  # range(im_index - self.temporal_frames, im_index + self.temporal_frames + 1):
            is_curr_item = True  # if i == im_index else False
            # i = max(0, i)
            # i = min(len(self.images_pre)-1, i)

            patch_pre, patch_post, patch_random = self.cut_patches(im_index, midpoint, midpoint_r, self.patch_size)

            if "pre" not in result:
                result['pre'] = patch_pre
            else:
                result['pre'] = torch.cat((result['pre'], patch_pre), dim=0)

            if is_curr_item:
                result['post'] = patch_post
            if is_curr_item:
                result['already'] = patch_random

        return result

    def __len__(self):
        return sum([(n.numel() // 2) for n in self.valid_indices]) * 5  # dont need to restart
