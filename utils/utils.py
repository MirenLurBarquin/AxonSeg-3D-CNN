import io
import json

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torchio as tio
from matplotlib.colors import ListedColormap
from sklearn.metrics import f1_score
import torch


def show_slices(slices, cmap='gray'):

    """ Function to display a row with the image slices """

    fig, axes = plt.subplots(1, len(slices), figsize=(10, 20))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap=cmap, origin="lower")


def print_used_memory():
    gib = psutil.virtual_memory().used / 2**30
    print(f'RAM used: {gib:.1f} GiB')


def prepare_plot(origImage, origMask, predMask):
    # initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(origImage)
	ax[1].imshow(origMask)
	ax[2].imshow(predMask)
	# set the titles of the subplots
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")
	# set the layout of the figure and display it
	figure.tight_layout()
	figure.show()


def load_subject(img_path: str, seg_path: str, fg_path: str):
    img = tio.ScalarImage(img_path)
    seg_all = tio.LabelMap(seg_path)
    foreground = tio.LabelMap(fg_path)
    img.data = img.data.to(torch.float32)

    return tio.Subject(img=img, seg=seg_all, fg=foreground)


def load_subject_comb(img_path: str):
    img = tio.ScalarImage(img_path)
    img.data = img.data.to(torch.float32)
    
    return tio.Subject(img=img)


def show_out(inputs, targets, pred, img_idx=0, slice_idx=30, cmap=None):

    if cmap is None:
        # initialize our figure
        figure, ax = plt.subplots(nrows=3, ncols=3, figsize=(13, 13))
        label_colors_rgb = [
            (0, 0, 0),    # Label 0: Unlabeled
            (255, 0, 0),    # Label 1: Blood vessel (red)
            (0, 255, 0),    # Label 2: vacuole (green)
            (0, 0, 255),    # Label 3: Cell cluster (blue)
            (255, 255, 0),  # Label 4: Axon (yellow)
            (186, 85, 211)  # Label 5: myelin (medium orchid)
        ]
        # Normalize RGB values to the range [0, 1]
        label_colors_normalized = np.array(label_colors_rgb) / 255.0
        # Create a custom color map
        custom_cmap = ListedColormap(label_colors_normalized)

        # If any label is not present (pred[img_idx][0][slice_idx, :, :].unique()!=6 or target[img_idx][0][slice_idx, :, :].unique()!=6 or), the colors wont be properly represented:
        #   Represent all the labels, this is just for visualization and it is not affecting the training.
        if len(np.unique(pred[img_idx][0][slice_idx, :, :].cpu().detach().numpy())) != 6 or len(np.unique(targets[img_idx][0][slice_idx, :, :].cpu().detach().numpy())) != 6:
            for i in range(6):
                pred[img_idx][0][slice_idx, 0, i] = i

        if len(np.unique(pred[img_idx][0][:, slice_idx, :].cpu().detach().numpy())) != 6 or len(np.unique(targets[img_idx][0][:, slice_idx, :].cpu().detach().numpy())) != 6:
            for i in range(6):
                pred[img_idx][0][0, slice_idx, i] = i

        if len(np.unique(pred[img_idx][0][:, :, slice_idx].cpu().detach().numpy())) != 6 or len(np.unique(targets[img_idx][0][:, :, slice_idx].cpu().detach().numpy())) != 6:
            for i in range(6):
                pred[img_idx][0][0, i, slice_idx] = i

    elif cmap == 'dif':
        # initialize our figure
        figure, ax = plt.subplots(nrows=3, ncols=5, figsize=(21, 13))
        missing_mask = np.where(((pred == 0) & (targets != 0)), 1, 0)
        newly_found_mask = np.where(((targets == 0) & (pred != 0)), 1, 0)
        misclassified_mask = np.where(((pred != targets) & (pred != 0) & (targets != 0)), 1, 0)
        well_mask = np.where(((pred != 0) & (targets != 0) & (pred == targets)), 1, 0)
        mask = missing_mask * 1 + newly_found_mask * 2 + misclassified_mask * 3 + well_mask * 4
        label_colors_rgb = [
            (0, 0, 0),    # Label 0: Unlabeled
            (0, 0, 255),    # Label 1: missing (blue)
            (255, 255, 0),    # Label 2: new (yellow)
            (255, 0, 0),    # Label 3: missclasified (red)
            (0, 255, 0),    # Label 4: well classified (green)

        ]

        # If any label is not present (mask[img_idx,0, slice_idx, :, :].unique()!=5), the colors wont be properly represented:
        #   Represent all the labels, this is just for visualization and it is not affecting the training.
        if len(np.unique(mask[img_idx, 0, slice_idx, :, :])) != 5:
            mask[img_idx, 0, slice_idx, 0, 0] = 0
            mask[img_idx, 0, slice_idx, 0, 1] = 1
            mask[img_idx, 0, slice_idx, 0, 2] = 2
            mask[img_idx, 0, slice_idx, 0, 3] = 3
            mask[img_idx, 0, slice_idx, 0, 4] = 4
        if len(np.unique(mask[img_idx, 0, :, slice_idx, :])) != 5:
            mask[img_idx, 0, 0, slice_idx, 0] = 0
            mask[img_idx, 0, 0, slice_idx, 1] = 1
            mask[img_idx, 0, 0, slice_idx, 2] = 2
            mask[img_idx, 0, 0, slice_idx, 3] = 3
            mask[img_idx, 0, 0, slice_idx, 4] = 4
        if len(np.unique(mask[img_idx, 0, :, :, slice_idx])) != 5:
            mask[img_idx, 0, 0, 0, slice_idx] = 0
            mask[img_idx, 0, 0, 1, slice_idx] = 1
            mask[img_idx, 0, 0, 2, slice_idx] = 2
            mask[img_idx, 0, 0, 3, slice_idx] = 3
            mask[img_idx, 0, 0, 4, slice_idx] = 4

        # Normalize RGB values to the range [0, 1]
        label_colors_normalized = np.array(label_colors_rgb) / 255.0
        # Create a custom color map
        custom_cmap_mask = ListedColormap(label_colors_normalized)
        custom_cmap = 'gray'
        ax[0, 3].imshow(pred[img_idx][0][slice_idx, :, :].cpu().detach().numpy(), cmap=custom_cmap)
        ax[1, 3].imshow(pred[img_idx][0][:, slice_idx, :].cpu().detach().numpy(), cmap=custom_cmap)
        ax[2, 3].imshow(pred[img_idx][0][:, :, slice_idx].cpu().detach().numpy(), cmap=custom_cmap)

        ax[0, 3].imshow(mask[img_idx, 0, slice_idx, :, :], cmap=custom_cmap_mask, alpha=0.3)
        ax[1, 3].imshow(mask[img_idx, 0, :, slice_idx, :], cmap=custom_cmap_mask, alpha=0.3)
        ax[2, 3].imshow(mask[img_idx, 0, :, :, slice_idx], cmap=custom_cmap_mask, alpha=0.3)

        ax[0, 4].imshow(inputs[img_idx][0][slice_idx, :, :].cpu().detach().numpy(), cmap=custom_cmap)
        ax[1, 4].imshow(inputs[img_idx][0][:, slice_idx, :].cpu().detach().numpy(), cmap=custom_cmap)
        ax[2, 4].imshow(inputs[img_idx][0][:, :, slice_idx].cpu().detach().numpy(), cmap=custom_cmap)

        ax[0, 4].imshow(mask[img_idx, 0, slice_idx, :, :], cmap=custom_cmap_mask, alpha=0.3)
        ax[1, 4].imshow(mask[img_idx, 0, :, slice_idx, :], cmap=custom_cmap_mask, alpha=0.3)
        ax[2, 4].imshow(mask[img_idx, 0, :, :, slice_idx], cmap=custom_cmap_mask, alpha=0.3)

        ax[0, 3].set_title("Pred. performance")
        ax[0, 4].set_title("Pred. performance")

    # plot the original image, its mask, and the predicted mask

    ax[0, 2].imshow(pred[img_idx][0][slice_idx, :, :].cpu().detach().numpy(), cmap=custom_cmap)
    ax[1, 2].imshow(pred[img_idx][0][:, slice_idx, :].cpu().detach().numpy(), cmap=custom_cmap)
    ax[2, 2].imshow(pred[img_idx][0][:, :, slice_idx].cpu().detach().numpy(), cmap=custom_cmap)
    ax[0, 0].imshow(inputs[img_idx][0][slice_idx, :, :].cpu().detach().numpy(), cmap='gray')
    ax[0, 1].imshow(targets[img_idx][0][slice_idx, :, :].cpu().detach().numpy(), cmap=custom_cmap)
    ax[1, 0].imshow(inputs[img_idx][0][:, slice_idx, :].cpu().detach().numpy(), cmap='gray')
    ax[1, 1].imshow(targets[img_idx][0][:, slice_idx, :].cpu().detach().numpy(), cmap=custom_cmap)
    ax[2, 0].imshow(inputs[img_idx][0][:, :, slice_idx].cpu().detach().numpy(), cmap='gray')
    ax[2, 1].imshow(targets[img_idx][0][:, :, slice_idx].cpu().detach().numpy(), cmap=custom_cmap)

    # set the titles of the subplots
    ax[0, 0].set_title("Image")
    ax[0, 1].set_title("Original Mask")
    ax[0, 2].set_title("Predicted Mask")
    plt.tight_layout()

    # set the layout of the figure and display it
    buf = io.BytesIO()
    fig = plt.gcf()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return fig


def is_not_json_compliant(variable):
    try:
        json.dumps(variable)
        return False
    except (TypeError, OverflowError):
        return True


def f1_score_batch_avg(batch_predictions, batch_targets):
    f1_scores = []
    for predictions, targets in zip(batch_predictions, batch_targets):
        # Flatten the predictions and targets
        predictions_flat = predictions.detach().cpu().numpy().flatten()
        targets_flat = targets.detach().cpu().numpy().flatten()

        # Calculate the F1 score for the current image
        f1 = f1_score(targets_flat, predictions_flat, average='weighted')
        f1_scores.append(f1)

    return np.average(f1_scores)
