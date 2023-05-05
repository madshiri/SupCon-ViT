from __future__ import print_function, division
import torch
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, transforms
from vit_model import *
import logging
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import time
import os
import copy
import random


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def train_model(model, optimizer, scheduler, logger, dataloaders, device, pos_weight, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            i = 0
            y_true = []
            y_pred = []
            for inputs, labels in dataloaders[phase]:
                labels = labels.to(device)
                inputs = inputs.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, mode='class')
                    loss = F.binary_cross_entropy_with_logits(
                            outputs.squeeze(1), labels.float(), pos_weight=pos_weight)
                    #corrects = torch.sum(outputs.logits.argmax(dim=1) == labels.data)
                    i += 1
                    print(loss.item())
                    #print(corrects)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                y_true.append(labels.cpu())
                y_pred.append(outputs.squeeze(1).cpu())
                running_loss += loss.item()
                #running_corrects += corrects

            y_true = torch.cat(y_true)
            y_pred = torch.cat(y_pred) > 0.5
            logger.debug(f'{phase}/metrics {classification_report(y_true, y_pred)}')
            print(f'{phase}/metrics {classification_report(y_true, y_pred)}')
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase])
            #epoch_acc = running_corrects.double() / dataset_sizes[phase]

            #print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            #logger.debug(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print(f'{phase}/loss: {epoch_loss:.4f}')
            logger.debug(f'{phase}/loss: {epoch_loss:.4f}')

            # deep copy the model
            #if phase == 'val' and epoch_acc > best_acc:
            if phase == 'val' and epoch_loss > best_loss:
                # best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    cudnn.benchmark = True
    plt.ion()   # interactive mode

    # Data augmentation and normalization for training
    # Just normalization for validation
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])

    seed = 28
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    data_dir = 'histo'
    all_patient_ids = sorted(os.listdir(data_dir))
    split = int(0.95 * len(all_patient_ids))
    # train validation split
    image_datasets = dict()
    samplers = dict()
    for mode in ['train', 'valid']:
        patient_ids = all_patient_ids[:split] if mode == "train" else all_patient_ids[split:]
        l = []
        imgs = []
        for patient_id in patient_ids:
            if mode == 'train':
                dataset = datasets.ImageFolder(os.path.join(data_dir, patient_id), valid_transform)
            else:
                dataset = datasets.ImageFolder(os.path.join(data_dir, patient_id), valid_transform)
            l.append(dataset)
            imgs.extend(dataset.imgs)
        weights = make_weights_for_balanced_classes(imgs, len(dataset.classes))
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        samplers[mode] = sampler
        image_datasets[mode] = torch.utils.data.ConcatDataset(l)

    dataloaders = dict()
    dataloaders['train'] = torch.utils.data.DataLoader(
        image_datasets['train'], batch_size=64, shuffle=True)
    dataloaders['valid'] = torch.utils.data.DataLoader(
        image_datasets['valid'], batch_size=64, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    pos_size = sum([sum(i.targets) for i in image_datasets['train'].datasets])
    pos_weight = torch.tensor([(dataset_sizes['train']-pos_size) / pos_size]).to(device)

    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=1)
    model.load_state_dict(torch.load('model-con-1epoch-32-imba.pth'))

    for par in model.parameters():
        par.requires_grad = False

    in_features = model.classifier.in_features
    # replace classifier with linear
    model.classifier = nn.Linear(in_features, 1)

    logger = logging.getLogger('ViT')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join('logs/con-vit-classifier-1epoch-imba.txt'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    model = model.to(device)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.AdamW(model.classifier.parameters(), lr=0.01)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model = train_model(model, optimizer_ft, exp_lr_scheduler, logger,
                        dataloaders, device, pos_weight, num_epochs=20)

    torch.save(model.state_dict(), 'model-con-classifier-1epoch-new.pth')


if __name__ == '__main__':
    main()
