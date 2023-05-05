from __future__ import print_function, division
import torch
import torch.backends.cudnn as cudnn
from sklearn.manifold import TSNE
import numpy as np
from torchvision import datasets, models, transforms
from transformers import ResNetForImageClassification
from vit_model import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import random
import matplotlib
matplotlib.use('TkAgg')


class new_model(nn.Module):
    def __init__(self,output_layer = None):
        super().__init__()
        self.pretrained = models.resnet18(pretrained=True)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1,len(self.layers)-self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])

        self.net = nn.Sequential(self.pretrained._modules)
        self.pretrained = None

    def forward(self,x):
        x = self.net(x)
        return x


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


def visualize_model(model, dataloaders, name, device):
    model.eval()
    label_list = []
    feature_list = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['valid']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            #features = model.resnet(inputs)
            features = model(inputs, mode='repr')
            #feature_list.extend(features.pooler_output.squeeze(2).squeeze(2).cpu().detach().numpy())
            feature_list.extend(features.cpu().detach().numpy())

            label_list.extend(labels.cpu().detach().numpy())

    labels = np.array(label_list)
    indices_0 = np.where(labels == 0)[0]
    indices_1 = np.where(labels == 1)[0]
    tsne = TSNE(3, verbose=1)
    tsne_proj = tsne.fit_transform(feature_list)
    tsne_0 = np.take(tsne_proj, indices_0, axis=0)
    tsne_1 = np.take(tsne_proj, indices_1, axis=0)
    #pca = PCA(n_components=2)
    #pca_proj = pca.fit_transform(feature_list)
    plot(tsne_proj[:, 0], tsne_proj[:, 1], tsne_proj[:, 2], labels, name)
    plot(tsne_0[:, 0], tsne_0[:, 1], tsne_0[:, 2], 'purple', name+'_0')
    plot(tsne_1[:, 0], tsne_1[:, 1], tsne_1[:, 2], 'yellow', name+'_1')


def plot(x, y, z, labels, name):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    scatter = ax.scatter(x, y, z, c=labels, alpha=0.4)
    fig.legend(handles=scatter.legend_elements()[0], labels=['Normal', 'Cancerous'])
    fig.savefig(f'{name}.png')


def main():
    cudnn.benchmark = True
    plt.ion()   # interactive mode

    seed = 28
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])
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

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                  sampler=samplers[x], num_workers=0)
                   for x in ['train', 'valid']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #model = ResNetForImageClassification.from_pretrained('microsoft/resnet-50')
    #model.classifier[1] = nn.Linear(2048, 1)
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=1)
    model.load_state_dict(torch.load('model-con-1epoch-32-imba.pth'))
    model = model.to(device)

    visualize_model(model, dataloaders, 'vis/valid-con-1epoch-imba-3d', device)


if __name__ == '__main__':
    main()
