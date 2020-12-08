import math
import random

import numpy as np
import sklearn
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import normalize

from .transforms import TransformFix

import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        # self.linear = nn.Linear(input_dim, output_dim)
        self.linear1 = nn.Linear(input_dim, input_dim//2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(input_dim//2, output_dim)

    def forward(self, x):
        # out = self.linear(x)
        out = self.linear2(self.relu(self.linear1(x)))
        return out

class FixmatchPick_Torch(object):

    def __init__(self, img_size, input_dim, num_class, device, step=5, max_iter='auto',
                 reduce='pca', d = 5, norm='l2', mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], fixmatch_threshold = 0.95, lambda_u = 1, lr = 1e-3):
        self.step = step
        self.max_iter = max_iter
        self.num_class = num_class
        self.input_dim = input_dim
        self.initial_norm(norm)
        self.initial_embed(reduce, d)
        self.lr = lr

        self.device = device
        self.lambda_u = lambda_u

        self.transform_fix = TransformFix(img_size, mean, std)
        self.elasticnet = ElasticNet(alpha=1.0, l1_ratio=1.0, fit_intercept=True,
                                         normalize=True, warm_start=True, selection='cyclic')
        self.threshold = fixmatch_threshold
        self.support_X, self.support_y = None, None
    
    def fit(self, X, y):
        self.support_X = self.norm(X)
        self.support_y = y
    
    def train(self, X, y, weak_x, strong_x, epoch = 50):

        self.model.train()
        total_loss = 0.
        for i in range(epoch):
            pred = self.model(X)
            Lx = F.cross_entropy(pred, y, reduction='mean')

            # Pseudo label temperature removed
            weak_prob = self.model(weak_x)
            strong_prob = self.model(strong_x)
            pseudo_label = F.softmax(weak_prob.detach_(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(self.threshold).float()

            Lu = (F.cross_entropy(strong_prob, targets_u,
                                  reduction='none') * mask).mean()

            loss = Lx + self.lambda_u * Lu
            loss = Lx
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()
            total_loss += loss.detach().item()
            # print(loss.item())
        return total_loss

    def predict(self, X, unlabel_X, strong_X, weak_X, show_detail=False, query_y=None):
        
        # self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, 0, 2**20)
        # Assert call fit before predict
        assert self.support_X is not None and self.support_y is not None, "Need to call function 'fit' before 'predict'"
        
        # Get support data and numbers
        support_X, support_y = torch.from_numpy(self.support_X).to(self.device), torch.from_numpy(self.support_y).to(self.device)
        way, num_support = self.num_class, len(support_X)
        
        # Normalize data
        query_X = torch.from_numpy(self.norm(X)).to(self.device)

        unlabel_X = torch.from_numpy(self.norm(unlabel_X)).to(self.device)
        weak_X = torch.from_numpy(self.norm(weak_X)).to(self.device)
        strong_X = torch.from_numpy(self.norm(strong_X)).to(self.device)

        
        # TODO: PROBLEM!!! If there's no unlabeled data, 
        # its number will be replaced by the number of query data.
        # Is this correct?
        num_unlabel = unlabel_X.shape[0]
        embeddings = torch.cat([support_X, unlabel_X])

        # PROBLEM!!! 
        # Not quite sure what this step mean?
        if self.max_iter == 'auto':
            # set a big number
            self.max_iter = num_support + num_unlabel
        elif self.max_iter == 'fix':
            # self.max_iter = math.ceil(num_unlabel/self.step)
            self.max_iter = 2
        else:
            assert float(self.max_iter).is_integer()
        
        support_set = np.arange(num_support).tolist()
        
        self.model = LogisticRegressionModel(self.input_dim, self.num_class).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        total_loss = self.train(embeddings[support_set], y[support_set], weak_X, strong_X)
        if show_detail:
                predicts = self.model(query_X)
                score = F.softmax(predicts.detach_(), dim=-1)
                max_probs, predicts = torch.max(score, dim=-1)
                predicts = predicts.cpu().detach().numpy()
                start_acc = np.mean(predicts == query_y)
       
        # Define the accuracy list
        if show_detail:
            acc_list = []

        # print(self.max_iter)
        for idx in range(self.max_iter):
            self.model.eval()

            prob = self.model(unlabel_X)
            score = F.softmax(prob.detach_(), dim=-1)
            max_probs, pseudo_y = torch.max(score, dim=-1)
            weak_prob = self.model(weak_X)
            strong_prob = self.model(strong_X)

            # Transform the labels into one-hot encoding
            y = torch.cat([support_y, pseudo_y])
            
            # Add new data to support set
            old_len = len(support_set)
            support_set = self.expand(support_set, way, num_support, pseudo_y, weak_prob, strong_prob)
            if old_len!=len(support_set):
                self.model = LogisticRegressionModel(self.input_dim, self.num_class).to(self.device)
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
                # Do we want to remove weak X from weak aug after expansion?
                total_loss = self.train(embeddings[support_set], y[support_set], weak_X, strong_X)
            
            if show_detail:
                predicts = self.model(query_X)
                score = F.softmax(predicts.detach_(), dim=-1)
                max_probs, predicts = torch.max(score, dim=-1)
                predicts = predicts.cpu().detach().numpy()
                acc_list.append(np.mean(predicts == query_y))

            if len(support_set) == len(embeddings):
                break

        self.model.eval()
        # Use the final classifier to do the prediction
        predicts = self.model(query_X)
        score = F.softmax(predicts.detach_(), dim=-1)
        max_probs, predicts = torch.max(score, dim=-1)
        predicts = predicts.cpu().detach().numpy()
        # print(predicts)
        # print(query_y)
        if show_detail:
            end_acc = np.mean(predicts == query_y)
            print("Final Improvements: %.2f" % (end_acc-start_acc))
            acc_list.append(np.mean(predicts == query_y))
            return acc_list
        return predicts


    def expand(self, support_set, way, num_support, pseudo_y, weak_prob, strong_prob):
        weak_prob = weak_prob.detach()
        strong_prob = strong_prob.detach()
        pseudo_y = pseudo_y.detach()
        weak_argmax = torch.argmax(weak_prob, dim = 1)
        strong_argmax = torch.argmax(strong_prob, dim = 1)
        

        # Determin for each row, whether the prediciton is above the threshold or not
        # passed = weak_prob[np.arange(len(weak_prob)), weak_argmax[weak_argmax == strong_argmax]] > self.threshold

        # Init array to store how many data are selected for each class
        # selected_per_class = np.zeros(way)

        # Randomly choose 
        # num2select = min((way*self.step), len(weak_argmax))


        i = 0

        for i in range(len(weak_argmax)):
            # print(weak_prob[i, weak_argmax[i]])
            if (i+num_support not in support_set) \
                    and (weak_argmax[i] == strong_argmax[i]) and (weak_prob[i, weak_argmax[i]] > self.threshold):
                
                support_set.append(i+num_support)
                # selected_per_class[weak_argmax[i]] += 1

            # # If already each class has more than num_step data selected  
            # if np.sum(selected_per_class >= self.step) == way:
            #     break

        """ 
        Implementation Note:
            Each time ICI select step*ways. It is not always optimal. 
            First try randomly select step*ways every time. 
            Then try not to always select step*ways. Can be smaller
        """
        return support_set

    def initial_embed(self, reduce, d):
        reduce = reduce.lower()
        assert reduce in ['isomap', 'itsa', 'mds', 'lle', 'se', 'pca', 'none']
        if reduce == 'isomap':
            from sklearn.manifold import Isomap
            embed = Isomap(n_components=d)
        elif reduce == 'itsa':
            from sklearn.manifold import LocallyLinearEmbedding
            embed = LocallyLinearEmbedding(n_components=d,
                                           n_neighbors=5, method='ltsa')
        elif reduce == 'mds':
            from sklearn.manifold import MDS
            embed = MDS(n_components=d, metric=False)
        elif reduce == 'lle':
            from sklearn.manifold import LocallyLinearEmbedding
            embed = LocallyLinearEmbedding(n_components=d, n_neighbors=5,eigen_solver='dense')
        elif reduce == 'se':
            from sklearn.manifold import SpectralEmbedding
            embed = SpectralEmbedding(n_components=d)
        elif reduce == 'pca':
            from sklearn.decomposition import PCA
            embed = PCA(n_components=d)
        if reduce == 'none':
            self.embed = lambda x: x
        else:
            self.embed = lambda x: embed.fit_transform(x)

    def initial_norm(self, norm):
        norm = norm.lower()
        assert norm in ['l2', 'none']
        if norm == 'l2':
            self.norm = lambda x: normalize(x)
        else:
            self.norm = lambda x: x

    def initial_classifier(self, classifier):
        assert classifier in ['lr', 'svm']
        if classifier == 'svm':
            from sklearn.svm import SVC
            self.classifier = SVC(C=10, gamma='auto', kernel='linear',probability=True)
        elif classifier == 'lr':
            from sklearn.linear_model import LogisticRegression
            self.classifier = LogisticRegressionModel(self.input_dim, self.num_class)

    def label2onehot(self, label, num_class):
        result = np.zeros((label.shape[0], num_class))
        for ind, num in enumerate(label):
            result[ind, num] = 1.0
        return result