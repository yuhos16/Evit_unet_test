import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from torchvision.transforms import functional as VF
from batchgenerators.augmentations.spatial_transformations import augment_spatial_2


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(0, 1))
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.uniform(-20, 20)
    image = ndimage.rotate(image, angle, order=1, reshape=False, mode='reflect')
    label = ndimage.rotate(label, angle, order=0, reshape=False, mode='nearest')
    return image, label


def get_random_elastic_params(alpha, sigma, size):
    dx = torch.rand([1, 1] + size) * 2 - 1
    if sigma > 0:
        kx = int(8 * sigma + 1)
        if kx % 2 == 0:
            kx += 1
        dx = VF.gaussian_blur(dx, [kx, kx], sigma)
    dx = dx * alpha / size[0]

    dy = torch.rand([1, 1] + size) * 2 - 1
    if sigma > 0:
        ky = int(8 * sigma + 1)
        if ky % 2 == 0:
            ky += 1
        dy = VF.gaussian_blur(dy, [ky, ky], sigma)
    dy = dy * alpha / size[1]

    return torch.cat([dx, dy], dim=1).permute(0, 2, 3, 1)


class RandomGenerator(object):
    def __init__(self, output_size=(256, 256)):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']  # image: (H, W, 3), label: (H, W)

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        h, w = image.shape[:2]
        if (h, w) != self.output_size:
            zoom_factor = (self.output_size[0] / h, self.output_size[1] / w, 1)
            image = zoom(image, zoom_factor, order=3)
            label = zoom(label, (self.output_size[0] / h, self.output_size[1] / w), order=0)

        # Transpose to [C, H, W]
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        label = torch.from_numpy(label.astype(np.int64))

        return {'image': image, 'label': label}


class RandomGenerator_DINO_Deform(object):
    def __init__(self, output_size=(256, 256), alpha=20., sigma=5.):
        self.output_size = output_size
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        h, w = image.shape[:2]
        if (h, w) != self.output_size:
            zoom_factor = (self.output_size[0] / h, self.output_size[1] / w, 1)
            image = zoom(image, zoom_factor, order=3)
            label = zoom(label, (self.output_size[0] / h, self.output_size[1] / w), order=0)

        image_tensor = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        label_tensor = torch.from_numpy(label.astype(np.int64))

        _, _, H, W = image_tensor.shape
        displacement = get_random_elastic_params(self.alpha, self.sigma, [H, W])

        image_dino = VF.elastic_transform(image_tensor, displacement, VF.InterpolationMode.BILINEAR, 0)[0]

        return {
            'image': image_tensor[0],  # [3, H, W]
            'label': label_tensor,
            'image_dino': image_dino,
            'disp': displacement
        }


class ham_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split='train', transform=None):
        self.transform = transform
        self.split = split
        list_path = os.path.join(list_dir, f"{self.split}.txt")
        assert os.path.exists(list_path), f"{list_path} 不存在！请先运行列表划分函数生成。"

        self.sample_list = open(list_path).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx].strip()
        data_path = os.path.join(self.data_dir, slice_name + '.npz')
        data = np.load(data_path)

        image = data['imgs']  # expected shape: (H, W, 3)
        label = data['gts']   # expected shape: (H, W)

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = slice_name
        return sample
