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
import numpy as np
from pytorchtools import EarlyStopping
import config as config
from utils.CustomGridSamplerNoAug import CustomGridSampler
from utils.utils import print_used_memory, load_subject, show_out, f1_score_batch_avg
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


def train_model(model, subjects, loss_fn, opt, rnd_id):

    # Create the training and test data loaders
    patch_size = config.PATCH_SIZE
    samples_per_volume = config.SAMPLES_PER_VOLUME
    max_queue_length = config.MAX_QUEUE_LENGTH
    patch_overlap = config.PATCH_OVERLAP

    sampler_train = CustomGridSampler(patch_size, mask='fg', patch_overlap=patch_overlap, train='train')
    sampler_val = CustomGridSampler(patch_size, mask='fg', patch_overlap=patch_overlap, train='val')
    train_patches_queue = tio.Queue(subjects, max_queue_length, samples_per_volume * 7, sampler_train, num_workers=config.NUM_WORKERS, shuffle_patches=True, shuffle_subjects=True)
    val_patches_queue = tio.Queue(subjects, max_queue_length, samples_per_volume * 2, sampler_val, num_workers=config.NUM_WORKERS, shuffle_patches=False, shuffle_subjects=False)
    train_patches_loader = DataLoader(train_patches_queue, batch_size=config.BATCH_SIZE, num_workers=0)  # num_workers must be 0
    val_patches_loader = DataLoader(val_patches_queue, batch_size=config.BATCH_SIZE, num_workers=0)  # num_workers must be 0

    avg_train_losses = []
    avg_valid_losses = []
    avg_train_f1 = []
    avg_valid_f1 = []

    # initialize the total training and validation loss
    train_losses = []
    valid_losses = []
    train_f1 = []
    valid_f1 = []

    early_stopping = EarlyStopping(patience=7, verbose=True, path=f'checkpoint_{rnd_id}.pt')  # Early stops the training if validation loss doesn't improve after a given patience.

    # Loop over epochs
    print("[INFO] training the network...")
    startTime = time.time()
    for e in tqdm(range(config.NUM_EPOCHS)):

        # set the model in training mode
        model.train()  # This directs the PyTorch engine to track our computations and gradients and build a computational graph to backpropagate later.

        print("[INFO] Epoch ", e + 1, ":")
        print("[INFO] Train:")
        for train_patches_batch in tqdm(train_patches_loader):
            inputs = train_patches_batch['img'][tio.DATA].to(config.DEVICE)  # key 'img' is in subject
            targets = train_patches_batch['seg'][tio.DATA]  # key 'seg' is in subject

            # perform a forward pass and calculate the training loss:
            logits = model(inputs).cpu()
            logits = logits.requires_grad_()
            targets = targets.float().requires_grad_()
            labels = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)
            if isinstance(loss_fn, MaskedGeneralizedDiceLoss) or isinstance(loss_fn, MaskedDiceLoss) or isinstance(loss_fn, MaskedGeneralizedDiceCELoss) or isinstance(loss_fn, MaskedDiceCELoss):
                mask = targets.clone()
                mask[mask > 0] = 1
                loss = loss_fn(logits, targets, mask)
            elif isinstance(loss_fn, CrossEntropyLoss):
                loss = loss_fn(logits, targets.squeeze())
            else:
                loss = loss_fn(logits, targets)

            f1_score = f1_score_batch_avg(batch_predictions=labels, batch_targets=targets)

            train_losses.append(loss.item())
            train_f1.append(f1_score.item())

            run[f"training/epoch_{str(+1)}/batch/trainloss"].append(loss)
            run[f"training/epoch_{str(+1)}/batch/trainf1"].append(f1_score)

            run[f"training/epoch_{str(e+1)}/batch/train_prediction_example_colored"].append(show_out(inputs, targets, labels))
            run[f"training/epoch_{str(e+1)}/batch/train_prediction_example"].append(show_out(inputs, targets, labels, cmap='dif'))
            # first, zero out any previously accumulated gradients, then perform backpropagation, and then update model parameters
            opt.zero_grad()  # switch off the gradient computation and freeze the model weights
            loss.backward()  # directs PyTorch to compute gradients of our loss w.r.t. all variables involved in the computation graph
            opt.step()

            # Unload the current batch from (GPU) memory to free up memory
            del inputs, targets, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # switch off autograd: we do not track gradients since we will not be learning or backpropagating
        print("[INFO] Validating:")
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            for val_patches_batch in tqdm(val_patches_loader):
                # send the input to the device
                inputs = val_patches_batch['img'][tio.DATA].to(config.DEVICE)   # key 'img' is in subject
                targets = val_patches_batch['seg'][tio.DATA]   # key 'seg' is in subject

                # make the predictions and calculate the validation loss
                logits = model(inputs).cpu()
                labels = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)

                if isinstance(loss_fn, MaskedGeneralizedDiceLoss) or isinstance(loss_fn, MaskedDiceLoss) or isinstance(loss_fn, MaskedGeneralizedDiceCELoss) or isinstance(loss_fn, MaskedDiceCELoss):
                    mask = targets.clone()
                    mask[mask > 0] = 1
                    loss = loss_fn(logits, targets, mask)
                elif isinstance(loss_fn, CrossEntropyLoss):
                    loss = loss_fn(logits, targets.squeeze(), mask)
                else:
                    loss = loss_fn(logits, targets)

                f1_score = f1_score_batch_avg(batch_predictions=labels, batch_targets=targets)

                valid_losses.append(loss.item())    # losses in the given epoch
                valid_f1.append(f1_score.item())
                run[f"training/epoch_{str(+1)}/batch/valloss"].append(loss)
                run[f"training/epoch_{str(+1)}/batch/valf1"].append(f1_score)
                run[f"training/epoch_{str(e+1)}/batch/val_prediction_example"].append(show_out(inputs, targets, labels, cmap='dif'))
                run[f"training/epoch_{str(e+1)}/batch/val_prediction_example_colored"].append(show_out(inputs, targets, labels))

                del inputs, targets, logits, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # calculate the average training and validation loss
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        train_f1_score = np.average(train_f1)
        valid_f1_score = np.average(valid_f1)
        avg_train_f1.append(train_f1_score)
        avg_valid_f1.append(valid_f1_score)

        epoch_len = len(str(config.NUM_EPOCHS))

        print_msg = (f'[{e+1:>{epoch_len}}/{config.NUM_EPOCHS:>{epoch_len}}] ' + f'train_loss: {train_loss:.5f} ' + f'valid_loss: {valid_loss:.5f}')

        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        train_f1 = []
        valid_f1 = []

        # update our training history
        run["AvgTrain_loss_per_epoch"].append(train_loss)
        run["AvgVal_loss_per_epoch"].append(valid_loss)
        run["AvgTrain_f1_per_epoch"].append(train_f1_score)
        run["AvgVal_f1_per_epoch"].append(valid_f1_score)
        # print the model training and validation information
        print(f"[INFO] EPOCH: {e + 1}/{config.NUM_EPOCHS}")
        print("Train loss: {:.6f}, Val loss: {:.4f}".format(train_loss, valid_loss))

        if e > 19:  # the first 20 epoch are quite volatile
            # early_stopping needs the validation loss to check if it has decresed, and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)
            run["checkpoint"].upload(f"checkpoint_{rnd_id}.pt")

            if early_stopping.early_stop:
                print("Early stopping")
                break

    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(f'checkpoint_{rnd_id}.pt'))

    # serialize the model to disk
    torch.save(model, os.path.join(config.BASE_OUTPUT, f"main_no_aug_{rnd_id}.pth"))  # save the weights of our trained U-Net model
    print_used_memory()

    filename = f'model_{rnd_id}.pkl'
    pickle.dump(model, open(filename, 'wb'))
    run["model"].upload(f"model_{rnd_id}.pkl")

    return model, avg_train_losses, avg_valid_losses


def make_prediction(model, test_subjects_dataset, loss_fn, rnd_id):
    # Create the training and test data loaders
    patch_size = config.PATCH_SIZE
    samples_per_volume = config.SAMPLES_PER_VOLUME
    max_queue_length = config.MAX_QUEUE_LENGTH
    patch_overlap = config.PATCH_OVERLAP

    sampler_test = CustomGridSampler(patch_size, mask='fg', patch_overlap=patch_overlap, train='test')
    test_patches_queue = tio.Queue(test_subjects_dataset, max_queue_length, samples_per_volume, sampler_test, num_workers=config.NUM_WORKERS)
    test_patches_loader = DataLoader(test_patches_queue, batch_size=config.BATCH_SIZE, num_workers=0,)  # num_workers must be 0

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
            targets = test_patches_batch['seg'][tio.DATA].to(config.DEVICE)  # key 'img' is in subject

            # make the predictions and calculate the validation loss
            logits = model(inputs)
            labels = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)
            outputs = labels

            if isinstance(loss_fn, MaskedGeneralizedDiceLoss):
                mask = targets.clone()
                mask[mask > 0] = 1
                loss = loss_fn(logits, targets, mask)
                run["test/mask"].append(show_out(mask, targets, labels))
            else:
                loss = loss_fn(logits, targets)

            f1_score = f1_score_batch_avg(batch_predictions=labels, batch_targets=targets)

            run["test/testloss"].append(loss)
            run["test/testf1"].append(f1_score)

            run["test/batch/test_prediction_example_colored"].append(show_out(inputs, targets, labels))
            run["test/batch/test_prediction_example"].append(show_out(inputs.cpu(), targets.cpu(), labels.cpu(), cmap='dif'))

            test_loss.append(loss.item())
            test_f1.append(f1_score.item())

            del inputs, targets, logits, outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print('TEST_STEPS=', len(test_loss))
    print('TEST_LOSS=', test_loss)
    avgTestLoss = np.average(test_loss)
    run['AvgTestLoss'] = avgTestLoss
    avgTestF1 = np.average(test_f1)
    run['AvgTestF1'] = avgTestF1
    return avgTestLoss


if __name__ == '__main__':
    rnd_id = random.randint(1000, 1999)
    run = neptune.init_run(
        project="dtu-msc-thesis/msc-project",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4YzU1ODNmOC0wNDI1LTQwOGEtYTA5YS03M2I2MDBmZGVjOGYifQ==",
        source_files=["config.py", "jobfile_gpu.sh", "main_noaug.py"])

    params = {"file": "main_noaug", "patch_size": config.PATCH_SIZE, "rnd_id": rnd_id, "num_epoch": config.NUM_EPOCHS, "batch_size": config.BATCH_SIZE, "num_workers": config.NUM_WORKERS, "learning_rate": config.INIT_LR, "optimizer": "Adam", "device": config.DEVICE, "samples_per_volume": config.SAMPLES_PER_VOLUME, "max_queue_length": config.MAX_QUEUE_LENGTH, "loss": config.NAME_LOSS, "CH": config.CH, "kernel_size": config.K}
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
    img_path = "/work3/s210289/msc_thesis/data/processed/img_proc.nii"
    fg_path = "/work3/s210289/msc_thesis/data/processed/foreground.nii"
    seg_path = "/work3/s210289/msc_thesis/data/processed/seg_all.nii"
    subject = load_subject(img_path, seg_path, fg_path)

    half1 = tio.Crop((0, 0, 0, 0, 205, 0))
    subject_half1 = half1(subject)
    half2 = tio.Crop((0, 0, 0, 0, 0, 205))
    subject_half2 = half2(subject)

    subjects = [subject_half1, subject_half2]  # without any transforms

    # Create the train and validation datasets
    subjects_dataset = tio.SubjectsDataset(subjects)

    # Initialize our UNet model
    model = UNet(n_classes=n_classes).to(config.DEVICE)
    summary(model, (1, config.PATCH_SIZE, config.PATCH_SIZE, config.PATCH_SIZE), batch_size=config.BATCH_SIZE)

    # Initialize loss function and optimizer
    loss_fn = config.LOSS
    test_loss_fn = config.TEST_LOSS
    opt = Adam(model.parameters(), lr=config.INIT_LR)

    # Read the job ID from the file
    time.sleep(5)
    with open(f'/work3/s210289/msc_thesis/jobid_{rnd_id}.txt', 'r') as f:
        job_id = f.readline().strip()

    run["jobID"] = job_id

    model, train_loss, valid_loss = train_model(model, subjects_dataset, loss_fn, opt, rnd_id)

    print("FINAL_TRAIN_LOSSES=", train_loss)
    print("FINAL_VALID_LOSSES=", valid_loss)

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.5)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    run['EarlyStopping'].upload(f"loss_plot_{rnd_id}.png")

    print("[INFO] load up model...")
    model = torch.load(os.path.join(config.BASE_OUTPUT, f"main_no_aug_{rnd_id}.pth")).to(config.DEVICE)

    pred = make_prediction(model, subjects_dataset, test_loss_fn, rnd_id)
