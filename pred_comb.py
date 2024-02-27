# Import the necessary packages
import os
import pickle
import time
import random

import matplotlib.pyplot as plt
import torch
import torchio as tio
from monai.losses import MaskedDiceLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
import nibabel as nib
import numpy as np
from pytorchtools import EarlyStopping
import config as config
from utils.CustomGridSamplerNoAug import CustomGridSampler
from utils.utils import print_used_memory, load_subject_comb, show_out, f1_score_batch_avg
import neptune
from utils.MaskedGeneralizedDiceLoss import MaskedGeneralizedDiceLoss
from utils.MaskedDiceCELoss import MaskedDiceCELoss
from utils.MaskedGeneralizedDiceCELoss import MaskedGeneralizedDiceCELoss
from torch.nn import CrossEntropyLoss

if config.CH:
    if config.K == 3:
        from models.unet3d_0_CH import UNet
    elif config.K == 5:
        from models.unet3d_0_CH_k5 import UNet
elif config.K == 3:
    from models.unet3d_0 import UNet
elif config.K == 5:
    from models.unet3d_0_k5 import UNet


def make_prediction(model, subject, loss_fn, aggr: bool=False):
    patch_size = config.PATCH_SIZE
    #patch_overlap = config.PATCH_OVERLAP
    patch_overlap = 10
    test_subject = subject  # without any transforms
    grid_sampler_test = tio.inference.GridSampler(test_subject, patch_size, patch_overlap)
    test_patches_loader = DataLoader(grid_sampler_test, batch_size=config.BATCH_SIZE)
    if aggr:
        aggregator = tio.inference.GridAggregator(grid_sampler_test)

    test_loss = []
    test_f1 = []
    print("[INFO] Predicting...")
    # load our model from disk and flash it to the current device
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for test_patches_batch in tqdm(test_patches_loader):
            # send the input to the device
            inputs = test_patches_batch['img'][tio.DATA].to(config.DEVICE)  # key 'img' is in subject
            locations = test_patches_batch[tio.LOCATION]
            # make the predictions and calculate the validation loss
            logits = model(inputs)
            labels = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)
            outputs = labels

            if aggr:
                aggregator.add_batch(outputs, locations)

            del inputs, logits, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if aggr:             
        output_tensor = aggregator.get_output_tensor()
        return output_tensor.numpy().astype(np.uint32).squeeze()

if __name__ == '__main__':
    rnd_id = 3760 #random.randint(1000, 1999)
    run = neptune.init_run(
        project="dtu-msc-thesis/msc-project",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4YzU1ODNmOC0wNDI1LTQwOGEtYTA5YS03M2I2MDBmZGVjOGYifQ==",
        source_files=["config.py", "jobfile_gpu.sh", "pred_noaug_comb.py"])

    params = {"file": "pred_full_combvol", "patch_size": config.PATCH_SIZE, "rnd_id": rnd_id, "num_epoch": config.NUM_EPOCHS, "batch_size": config.BATCH_SIZE, "num_workers": config.NUM_WORKERS, "learning_rate": config.INIT_LR, "optimizer": "Adam", "device": config.DEVICE, "samples_per_volume": config.SAMPLES_PER_VOLUME, "max_queue_length": config.MAX_QUEUE_LENGTH, "loss": config.NAME_LOSS, "CH": config.CH, "kernel_size": config.K}
    run["parameters"] = params

    # Fix random seed: always use a fixed random seed to guarantee that when you run the code twice you will get the same outcome.
    torch.manual_seed(0)

    # Check if CUDA is available:
    print('CUDA available: ', torch.cuda.is_available())
    print('Device: ', config.DEVICE)
    print('_____________________________________________________________')

    # Run the bsub command to get the job ID and save the output to a file
    os.system(f'bsub -oo jobid_{rnd_id}.txt echo $LSB_JOBID')

    # Get the data subject and the create dataset
    n_classes = config.NUM_CLASSES
    img_path ='/work3/s210289/msc_thesis/data/processed/img_proc.nii'
    subject = load_subject_comb(img_path)


    test_loss_fn = config.TEST_LOSS

    # Read the job ID from the file
    time.sleep(5)
    #with open(f'/work3/s210289/msc_thesis/jobid_{rnd_id}.txt', 'r') as f:
    #   job_id = f.readline().strip()

    #run["jobID"] = job_id

    aggr = config.AGGREGATOR
    print("[INFO] load up model...")
    model = torch.load(os.path.join(config.BASE_OUTPUT, f"main_no_aug_{rnd_id}.pth")).to(config.DEVICE)

    pred = make_prediction(model, subject, test_loss_fn, aggr)
    output = nib.Nifti1Image(pred, subject.img.affine)
    nib.save(output,f'output/output_pred_{rnd_id}_cross_red.nii')
