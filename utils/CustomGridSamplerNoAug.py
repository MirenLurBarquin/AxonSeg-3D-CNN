# Adapted from TorchIo's Grid Sampler: https://torchio.readthedocs.io/_modules/torchio/data/sampler/grid.html#GridSampler

# Copyright (c) TorchIO developers
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from itertools import permutations
from typing import Generator, Optional, Union

import numpy as np
import torch
from torchio.data.image import Image
from torchio.data.sampler.sampler import PatchSampler
from torchio.data.subject import Subject
from torchio.typing import TypeSpatialShape, TypeTripletInt
from torchio.utils import to_tuple


class CustomGridSampler(PatchSampler):
    """
    Custom: GridSampler but just taking 'legal' patches, i.e. inside the sample cylindric volume where the information is.

    For training without augmentation.
    """

    r"""Extract patches across a whole volume.

    Args:
        subject: Instance of :class:`~torchio.data.Subject`
            from which patches will be extracted.
        patch_size: Tuple of integers :math:`(w, h, d)` to generate patches
            of size :math:`w \times h \times d`.
            If a single number :math:`n` is provided,
            :math:`w = h = d = n`.
        patch_overlap: Tuple of even integers :math:`(w_o, h_o, d_o)`
            specifying the overlap between patches for dense inference. If a
            single number :math:`n` is provided, :math:`w_o = h_o = d_o = n`.
        padding_mode: Same as :attr:`padding_mode` in
            :class:`~torchio.transforms.Pad`. If ``None``, the volume will not
            be padded before sampling and patches at the border will not be
            cropped by the aggregator.
            Otherwise, the volume will be padded with
            :math:`\left(\frac{w_o}{2}, \frac{h_o}{2}, \frac{d_o}{2} \right)`
            on each side before sampling. If the sampler is passed to a
            :class:`~torchio.data.GridAggregator`, it will crop the output
            to its original size.
    """

    def __init__(
        self,
        patch_size: TypeSpatialShape,
        mask: Optional[str],
        train: str,
        patch_overlap: TypeSpatialShape = (0, 0, 0),
        padding_mode: Union[str, float, None] = None,
    ):
        super().__init__(patch_size)
        self.patchsize = patch_size
        self.mask_name = mask
        self.padding_mode = padding_mode
        self.patch_overlap = np.array(to_tuple(patch_overlap, length=3))
        self.train = train

    def __getitem__(self, index, subject: Subject):

        # Assume 3D
        location = self._compute_locations(subject)[index]
        index_ini = location[:3]
        cropped_subject = self.crop(subject, index_ini, self.patch_size)
        return cropped_subject

    def _pad(self, subject: Subject) -> Subject:
        if self.padding_mode is not None:
            from torchio.transforms import Pad

            border = self.patch_overlap // 2
            padding = border.repeat(2)
            pad = Pad(padding, padding_mode=self.padding_mode)  # type: ignore[arg-type]  # noqa: B950
            subject = pad(subject)  # type: ignore[assignment]
        return subject

    def _compute_locations(self, subject: Subject):
        sizes = subject.spatial_shape, self.patch_size, self.patch_overlap
        self._parse_sizes(*sizes)  # type: ignore[arg-type]
        return self.generate_patch_corners(*sizes, subject=subject)  # type: ignore[arg-type]

    def _generate_patches(  # type: ignore[override]
        self,
        subject: Subject,
        num_patches: Optional[int] = None,
    ) -> Generator[Subject, None, None]:
        subject = self._pad(subject)
        sizes = subject.spatial_shape, self.patch_size, self.patch_overlap
        self._parse_sizes(*sizes)  # type: ignore[arg-type]
        locations = self.generate_patch_corners(imsize=sizes[0], patchsize=sizes[1], patch_overlap=sizes[2], subject=subject)  # type: ignore[arg-type]  # noqa: B950
        if self.train == 'train':
            locations_aux = locations[:int(len(locations) * 0.7)]
        elif self.train == 'val':
            locations_aux = locations[int(len(locations) * 0.7):int(len(locations) * 0.9)]
        elif self.train == 'test':
            locations_aux = locations[int(len(locations) * 0.9):]

        if len(locations_aux) == 0:
            warnings.warn("Not enough patches")
        else:
            patches_left = num_patches if num_patches is not None else True
            while patches_left:
                rand_loc = np.random.randint(low=0, high=locations_aux.shape[0])
                index_ini = locations_aux[rand_loc][:3]
                yield self.extract_patch(subject, index_ini)
                if num_patches is not None:
                    patches_left -= 1

    def extract_patch(
        self,
        subject: Subject,
        index_ini: TypeTripletInt,
    ) -> Subject:
        self.location = index_ini
        cropped_subject = self.crop(subject, index_ini, self.patch_size)  # type: ignore[arg-type]  # noqa: B950
        return cropped_subject

    @staticmethod
    def _parse_sizes(
        image_size: TypeTripletInt,
        patch_size: TypeTripletInt,
        patch_overlap: TypeTripletInt,
    ) -> None:
        image_size_array = np.array(image_size)
        patch_size_array = np.array(patch_size)
        patch_overlap_array = np.array(patch_overlap)
        if np.any(patch_size_array > image_size_array):
            message = (
                f'Patch size {tuple(patch_size_array)} cannot be'
                f' larger than image size {tuple(image_size_array)}'
            )
            raise ValueError(message)
        if np.any(patch_overlap_array >= patch_size_array):
            message = (
                f'Patch overlap {tuple(patch_overlap_array)} must be smaller'
                f' than patch size {tuple(patch_size_array)}'
            )
            raise ValueError(message)
        if np.any(patch_overlap_array % 2):
            message = (
                'Patch overlap must be a tuple of even integers,'
                f' not {tuple(patch_overlap_array)}'
            )
            raise ValueError(message)

    def get_mask_image(self, subject: Subject) -> Image:
        assert self.mask_name is not None
        if self.mask_name in subject:
            return subject[self.mask_name]
        else:
            message = (
                f'Image "{self.mask_name}" not found in subject: {subject}'
            )
            raise KeyError(message)

    def get_mask(self, subject: Subject) -> torch.Tensor:
        data = self.get_mask_image(subject).data
        if torch.any(data < 0):
            message = (
                'Negative values found'
                f' in probability map "{self.probability_map_name}"'
            )
            raise ValueError(message)
        return data

    # @staticmethod
    def generate_patch_corners(self, imsize, patchsize, patch_overlap, subject: Subject):

        cylindric_transf_mask = self.get_mask(subject).squeeze()

        stride = -(patch_overlap - patchsize)
        stride_perms = np.asarray(list(set(permutations(stride))))

        mask = cylindric_transf_mask
        xyz_perms = np.empty((0, 6))

        for stride_perm in stride_perms:
            Y, X = np.ogrid[:imsize[1] + 1, :imsize[0] + 1]  # it looks like: [array([[0],[1],[2],[3],[4]]), array([[0, 1, 2, 3, 4]])] first one column (Y), second one row (X)

            legal_patches = np.zeros(imsize)  # Every pixel in the image could be a possible down-left corner for a patch
            x_range = list(range(0, imsize[0] - patchsize[0], stride_perm[0]))  # - patchsize because we do not want corners that do not fit a whole patch on the left of the img
            y_range = list(range(0, imsize[1] - patchsize[1], stride_perm[1]))
            z_range = list(range(0, imsize[2] - patchsize[2], stride_perm[2]))

            for x in x_range:
                for y in y_range:
                    for z in z_range:
                        legal_patches[x, y, z] = mask[x, y, z] * mask[x + patchsize[0], y, z] * mask[x, y + patchsize[0], z] * mask[x, y, z + patchsize[0]] * mask[x + patchsize[0], y + patchsize[0], z] * mask[x + patchsize[0], y, z + patchsize[0]] * mask[x, y + patchsize[0], z + patchsize[0]] * mask[x + patchsize[0], y + patchsize[0], z + patchsize[0]]  # all corners of the patch should be =1 in the mask

            xyz_range = np.where(legal_patches == 1)  # patches inside the radius and according to the stride
            xyz = np.asarray(list(zip(xyz_range[0], xyz_range[1], xyz_range[2])))  # saves x, y values as tuples

            if len(xyz_range[0]) == 0:
                return xyz.astype(int)

            xyz_ext = np.hstack((xyz, xyz + patchsize[0]))
            xyz_perms = np.vstack((xyz_perms, xyz_ext))
        return xyz_perms.astype(int)
