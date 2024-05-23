import torch
import numpy as np
from math import sqrt, ceil
from torchvision.transforms.functional import rotate
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.functional import center_crop

def radon_transform(slices, nb_angles, mode=InterpolationMode.BILINEAR):
    '''
        Compute sinogram [N, ANGLES_NUMBER, L],
        for given slices [N, MATERIALS_NUMBER, L, L]

        This version does NOT center_crop to diagonal length
        i.e. each slice support is assumed to lie within the
        inscribed circle (diameter = L)
    '''
    angles = np.linspace(0, 180, nb_angles, endpoint=False)

    return torch.cat([rotate(slices, angle, mode).sum(2) for angle in angles], 1)

def center_radon_transform(slices, nb_angles, mode=InterpolationMode.BILINEAR):
    '''
        Compute sinogram [N, ANGLES_NUMBER, L],
        for given slices [N, MATERIALS_NUMBER, L, L]
    '''
    angles = np.linspace(0, 180, nb_angles, endpoint=False)
    diagonal_length = ceil(sqrt(2) * slices.shape[-1])
    centered_slices = center_crop(slices, [diagonal_length]*2)

    return torch.cat(
        [rotate(centered_slices, angle, mode).sum(2) for angle in angles], 1
    )

