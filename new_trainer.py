from __future__ import print_function, division
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, transforms
from vit_model import *
import logging
import matplotlib.pyplot as plt
import time
import os
import copy
import random
import losses


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


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


def train_model(model, criterion, optimizer, scheduler, logger, dataloaders, device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    #best_acc = 0.0
    best_loss = 0.0

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
            for inputs, labels in dataloaders[phase]:
                labels = labels.to(device)
                if phase == 'train':
                    inputs1, inputs2 = inputs[0], inputs[1]
                    inputs1 = inputs1.to(device)
                    inputs2 = inputs2.to(device)
                    inputs = torch.cat([inputs1, inputs2], dim=0)
                else:
                    inputs = inputs.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #outputs = model(inputs).logits
                    #_, preds = torch.max(outputs, 1)
                    #loss = criterion(outputs, labels)
                    features = model(inputs, mode='repr')
                    #features = model(inputs).squeeze(2).squeeze(2)
                    #f_lin = lin(features)
                    #norm_feat = F.normalize(f_lin, dim=1)
                    if phase == 'train':
                        try:
                            f1, f2 = torch.split(features, [32, 32], dim=0)
                        except:
                            continue
                        f = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    else:
                        f = features.unsqueeze(1)
                    #f = features.squeeze(2)
                    #f = f.squeeze(2)
                    #f = features.last_hidden_state[:, 0, :]
                    #f = torch.mean(features.last_hidden_state, 1)
                    #f_con = conv(f)
                    #f_norm = F.normalize(f, dim=1)
                    loss = criterion(f, labels)
                    #preds = model(features, mode='classification')
                    #corrects = torch.sum(preds.argmax(dim=1)[:16] == labels.data)
                    i += 1
                    print(loss)
                    #print(corrects)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                #running_corrects += corrects

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
                #best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    #print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    cudnn.benchmark = True
    plt.ion()   # interactive mode

    # Data augmentation and normalization for training
    # Just normalization for validation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])
    traine_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomVerticalFlip(), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
    ])
    traine_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.AutoAugment(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

    ])

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
                dataset = datasets.ImageFolder(os.path.join(data_dir, patient_id), TwoCropTransform(train_transform))
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
        image_datasets['train'], batch_size=32, shuffle=True)
    dataloaders['valid'] = torch.utils.data.DataLoader(
        image_datasets['valid'], batch_size=32, shuffle=False)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #model = models.resnet50(pretrained=True)
    #model = torch.nn.Sequential(*(list(model.children())[:-1]))
    #num_ftrs = model.fc.in_features
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=1)
    #classifier = nn.Linear(768, 2).to(device)
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    #model.fc = nn.Linear(num_ftrs, 2)

    logger = logging.getLogger('ViT')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join('logs/con-vit-67.txt'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    model = model.to(device)

    criterion = losses.SupConLoss(temperature=0.006)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.AdamW(model.parameters(), lr=0.00007)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(inplace=True), nn.Linear(768, 128))

    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, logger,
                        dataloaders, device, num_epochs=1)

    torch.save(model.state_dict(), 'model-con-1epoch-67.pth')


if __name__ == '__main__':
    main()
