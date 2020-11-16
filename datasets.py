import os
import os.path as osp
import pickle
import csv
import collections

import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from models.transforms import TransformFix

class NonTransDataset(Dataset):
    def __init__(self, data_root, setname, img_size):
        self.img_size = img_size

        self.data = np.load(osp.join(data_root, setname + "_images.npz"))["images"]
        self.label = np.load(osp.join(data_root, setname + "_labels.pkl"), allow_pickle=True)["labels"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if i == -1:
            return torch.zeros([3, self.img_size, self.img_size]), 0
        image, label = self.data[i], self.label[i]
        # image = self.transform(Image.open(path).convert('RGB'))
        image = Image.fromarray(image)
        return image, 1

class FixMatchDataSet(Dataset):

    def __init__(self, data_root, setname, img_size):
        self.img_size = img_size

        self.data = np.load(osp.join(data_root, setname + "_images.npz"))["images"]
        self.label = np.load(osp.join(data_root, setname + "_labels.pkl"), allow_pickle=True)["labels"]
        # print(self.data["images"].shape)

        if setname=='test' or setname=='val':
            self.transform_fix = TransformFix(img_size, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            self.transform = transforms.Compose([
                                               transforms.Resize((img_size, img_size)),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                       ])
        else:
            self.transform = transforms.Compose([
                                            transforms.RandomResizedCrop((img_size, img_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if i == -1:
            return torch.zeros([3, self.img_size, self.img_size]), 0
        image, label = self.data[i], self.label[i]
        # image = self.transform(Image.open(path).convert('RGB'))
        (strong, weak) = self.transform_fix(Image.fromarray(image))
        image = self.transform(Image.fromarray(image))
        return (image, strong, weak), 1
 
class DataSet(Dataset):

    def __init__(self, data_root, setname, img_size):
        self.img_size = img_size

        self.data = np.load(osp.join(data_root, setname + "_images.npz"))["images"]
        self.label = np.load(osp.join(data_root, setname + "_labels.pkl"), allow_pickle=True)["labels"]
        # print(self.data["images"].shape)

        if setname=='test' or setname=='val':
            self.transform = transforms.Compose([
                                               transforms.Resize((img_size, img_size)),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                       ])
        else:
            self.transform = transforms.Compose([
                                            transforms.RandomResizedCrop((img_size, img_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if i == -1:
            return torch.zeros([3, self.img_size, self.img_size]), 0
        image, label = self.data[i], self.label[i]
        # image = self.transform(Image.open(path).convert('RGB'))
        image = self.transform(Image.fromarray(image))
        return image, 1


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch #num_batches
        self.n_cls = n_cls # test_ways
        self.n_per = np.sum(n_per) # num_per_class
        self.number_distract = n_per[-1]

        label = np.array(label)
        self.m_ind = []
        for i in range(min(label), max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            indicator_batch = []
            classes = torch.randperm(len(self.m_ind))
            trad_classes = classes[:self.n_cls]
            for c in trad_classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                cls_batch = l[pos]
                cls_indicator = np.zeros(self.n_per)
                cls_indicator[:cls_batch.shape[0]] = 1
                if cls_batch.shape[0] != self.n_per:
                    cls_batch = torch.cat([cls_batch, -1*torch.ones([self.n_per-cls_batch.shape[0]]).long()], 0)
                batch.append(cls_batch)
                indicator_batch.append(cls_indicator)
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


# filenameToPILImage = lambda x: Image.open(x).convert('RGB')
numpyToPILImage = lambda x: Image.fromarray(x).convert('RGB')
def loadSplit(splitFile):
            dictLabels = {}
            with open(splitFile) as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvreader, None)
                for i,row in enumerate(csvreader):
                    filename = row[0]
                    label = row[1]
                    if label in dictLabels.keys():
                        dictLabels[label].append(filename)
                    else:
                        dictLabels[label] = [filename]
            return dictLabels


class EmbeddingDataset(Dataset):

    def __init__(self, data_root, img_size, setname = 'train'):
        self.img_size = img_size
        # Transformations to the image
        if setname=='train':
            self.transform = transforms.Compose([numpyToPILImage,
                                                transforms.Resize((img_size, img_size)),
                                                transforms.RandomCrop(img_size, padding=8),
                                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                                ])
        else:
            self.transform = transforms.Compose([numpyToPILImage,
                                                transforms.Resize((img_size, img_size)),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ])

        
        # self.ImagesDir = os.path.join(dataroot,'images')
        # self.data = loadSplit(splitFile = os.path.join(dataroot,'train' + '.csv'))

        self.data = np.load(osp.join(data_root, setname + "_images.npz"))["images"]
        self.label = np.load(osp.join(data_root, setname + "_labels.pkl"), allow_pickle=True)["labels"]

        # self.data = collections.OrderedDict(sorted(self.data.items()))
        # keys = list(self.data.keys())
        # self.classes_dict = {keys[i]:i  for i in range(len(keys))} # map NLabel to id(0-99)

        # self.Files = []
        # self.belong = []

        # for c in range(len(keys)):
        #     num = 0
        #     num_train = int(len(self.data[keys[c]]) * 9 / 10)
        #     for file in self.data[keys[c]]:
        #         if setname == 'train' and num <= num_train:
        #             self.Files.append(file)
        #             self.belong.append(c)
        #         elif setname =='val' and num>num_train:
        #             self.Files.append(file)
        #             self.belong.append(c)
        #         num = num+1


        self.__size = len(self.data)

    def __getitem__(self, index):

        # c = self.label[index]
        # File = self.Files[index]

        # path = os.path.join(self.ImagesDir,str(File))
        # try:
        #     images = self.transform(path)
        # except RuntimeError:
        #     import pdb;pdb.set_trace()
        # return images,c
        image, label = self.data[index], self.label[index]
        # image = self.transform(Image.open(path).convert('RGB'))
        image = self.transform(image)
        return image, label

    def __len__(self):
        return self.__size

