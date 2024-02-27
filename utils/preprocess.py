import os

import numpy as np
import torch
import torchio as tio
from skimage.morphology import ball, binary_closing, dilation
import config
import nibabel as nib


class VolumeXNH():
    def __init__(self, img_path: str = config.IMAGE_PATH, seg_path: str = config.SEG_PATH, segx_path: str = config.SEGX_PATH, n_class: int = config.NUM_CLASSES, normalize_data: bool = True):

        '''
        :param img_path (str):  Path to the .nii image volume file
        :param seg_path (str):  Path to the .nii axon segmentation volume file
        :param segx_path (str):  Path to the .nii extra-axonal segmentation volume file
        :param n_class (int):  No. of classes we want to segment (=1 in the axonal case, =3 for extra axonal elements (blod vessels, vacuoles, cell clusters))
        :param normalize_data (bool):  Normalization on the whole image volume
        :return: None
        '''

        self.img_path = img_path
        self.seg_path = seg_path
        self.segx_path = segx_path
        self.n_class = n_class
        self.normalize_data = normalize_data

        # Data loading
        self.img, self.seg, self.segx, self.myelin_seg, self.fg = self.load_data()

        # Merge axonal, extra-axonal and myelin segmentations:
        self.seg_all = self.merge_segmentations(self.seg, self.segx, self.myelin_seg)

        self.subject = tio.Subject(img=self.img, seg=self.seg_all, fg=self.fg)

    def load_data(self):

        '''
        Load and preprocess the data
        '''

        img = tio.ScalarImage(self.img_path)
        seg = tio.LabelMap(self.seg_path)  # Intensity transforms are not applied to label images.
        segx = tio.LabelMap(self.segx_path)
        segx.set_data(segx.data[:, :, :, :config.INPUT_IMAGE_DEPTH])
        fg = tio.LabelMap(tensor=self.foreground_mask(img), affine=img.affine)

        # Get the binary segmentation for axons bc. in raw it labels each individual axon 1 to 54.
        seg.numpy()[seg.numpy() != 0] = 1

        # Get the myelin mask
        myelin_seg = self.get_myelin(seg)

        # Normalize the data
        if self.normalize_data:
            subject_norm = tio.Subject(img_norm=img, fg=fg)
            rescale = tio.RescaleIntensity((0, 255), masking_method='fg')
            znorm = tio.ZNormalization(masking_method='fg')
            img = rescale(znorm(subject_norm)).img_norm

        img.numpy()[fg.numpy() == 0] = 0
        img.data = img.data.to(torch.float32)  # the image has to be float32 to be able to pass it into torch.nn.conv3d.
        seg.data = seg.data.to(torch.uint8)
        segx.data = segx.data.to(torch.uint8)
        myelin_seg.data = myelin_seg.data.to(torch.uint8)
        fg.data = fg.data.to(torch.uint8)

        return (img, seg, segx, myelin_seg, fg)

    def save_seg(self):

        '''
        Store locally the processed original image (normalized & intensity rescaled), the merged segmentations and foreground mask
        '''

        img = self.img.data.numpy().squeeze()  # the img must be float to be able to pass it to the conv.
        img = nib.Nifti1Image(img, self.seg.affine)
        seg_all = self.seg_all.data.numpy().astype(np.uint8).squeeze()
        seg_all = nib.Nifti1Image(seg_all, self.seg.affine)
        fg = self.fg.data.numpy().astype(np.uint8).squeeze()
        fg = nib.Nifti1Image(fg, self.seg.affine)
        nib.save(img, os.path.join("/work3/s210289/msc_thesis/data/processed", 'img_proc.nii'))
        nib.save(seg_all, os.path.join("/work3/s210289/msc_thesis/data/processed", 'seg_all.nii'))
        nib.save(fg, os.path.join("/work3/s210289/msc_thesis/data/processed", 'foreground.nii'))

    @staticmethod
    def merge_segmentations(seg, segx, myelin_seg):

        '''
        Create an only mask for axonal and extra-axonal structures:
        [0: unlabeled, 1: blood vessel, 2: vacuole, 3: cell cluster, 4: axon, 5: myelin]
        '''
        seg_all = tio.LabelMap(tensor=segx.data, affine=segx.affine)

        seg_all.data[seg.data != 0] = 4
        seg_all.data[myelin_seg.data != 0] = 5

        return seg_all

    @staticmethod
    def foreground_mask(img):

        '''
        Seperate foreground and background
        '''

        # Apply a threshold in an image and return the resulting image:
        #   By checking the histrogram of voxel intensities, having values avobe and under 0, we decided that we will mask leaving out zero:
        mask = (img.data != 0).squeeze()

        # Closing = dilation followed by erosion. Closes holes and connects objects.
        #   (binary_closing performs faster for binary images than grayscale closing)
        mask_2 = binary_closing(mask, footprint=ball(2))
        return torch.from_numpy(mask_2).long().unsqueeze(0)

    @staticmethod
    def get_myelin(seg):

        '''
        Segment the contour of the axons, which we will consider myelin.
        '''

        seg_dilated = tio.LabelMap(tensor=torch.from_numpy(dilation(seg.data.squeeze(), ball(3))).long().unsqueeze(0))
        contour = torch.from_numpy(seg_dilated.data.numpy() - seg.data.numpy())
        contour_seg = tio.LabelMap(tensor=contour)

        # If we would like to refine the mask:
        # if ref:
        #    subject_contour = tio.Subject(image=img, contour =contour_seg)
        #    mask_contour = tio.transforms.Mask(masking_method='contour')
        #    img_contour = mask_contour(subject_contour).image
        #    seg_contour = mask_contour(subject_contour).contour
        #    seg_contour.data[img_contour.data>=90]=0
        #    return seg_contour
        # else:
        return contour_seg
