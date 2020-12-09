import math
import random

import numpy as np
import sklearn
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from tqdm import tqdm

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

class ICITorch(object):

    def __init__(self, input_dim, num_class, device, step=5, max_iter='auto',
                 reduce='pca', d=5, norm='l2', lr = 1e-3):
        self.step = step
        self.max_iter = max_iter
        self.num_class = num_class
        self.input_dim = input_dim
        self.initial_embed(reduce, d)
        self.initial_norm(norm)
        self.elasticnet = ElasticNet(alpha=1.0, l1_ratio=1.0, fit_intercept=True,
                                         normalize=True, warm_start=True, selection='cyclic')

        self.lr = lr
        self.device = device
    def fit(self, X, y):
        self.support_X = self.norm(X)
        self.support_y = y

    def train(self, X, y, epoch = 50):
        """ Training loops for PyTorch Logistic Regression model """
        # TODO: Enable batch loading
        self.model.train()
        total_losses = []
        # pbar = tqdm(range(epoch))
        # for i in pbar:
        for i in range(epoch):
            pred = self.model(X)
            loss = F.cross_entropy(pred, y, reduction='mean')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss = loss.detach().item()
            total_losses.append(loss)
            
            # pbar.set_postfix({"Loss": loss})
        # print(np.sum(total_losses))
        return total_losses 

    def predict(self, X, unlabel_X=None, show_detail=False, query_y=None):
        support_X, support_y = self.support_X, self.support_y
        way, num_support = self.num_class, len(support_X)
        query_X = self.norm(X)
        if unlabel_X is None:
            unlabel_X = query_X
        else:
            unlabel_X = self.norm(unlabel_X)
        num_unlabel = unlabel_X.shape[0]
        assert self.support_X is not None

        embeddings = np.concatenate([support_X, unlabel_X])
        X = self.embed(embeddings)
        H = np.dot(np.dot(X, np.linalg.inv(np.dot(X.T, X))), X.T)
        X_hat = np.eye(H.shape[0]) - H
        if self.max_iter == 'auto':
            # set a big number
            self.max_iter = num_support + num_unlabel
        elif self.max_iter == 'fix':
            # self.max_iter = math.ceil(num_unlabel/self.step)
            self.max_iter = 15
        else:
            assert float(self.max_iter).is_integer()
        support_set = np.arange(num_support).tolist()
        

        support_X, support_y = torch.from_numpy(self.support_X).to(self.device), torch.from_numpy(self.support_y).to(self.device)
        way, num_support = self.num_class, len(support_X)
        # Normalize data
        query_X = torch.from_numpy(query_X).to(self.device)
        unlabel_X = torch.from_numpy(unlabel_X).to(self.device)
        embeddings = torch.from_numpy(embeddings).to(self.device)

        # Train the simple classifier
        self.model = LogisticRegressionModel(self.input_dim, self.num_class).to(self.device)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr,
        #                   momentum=0.9, nesterov=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.train(support_X, support_y)
        
        if show_detail:
            predicts = self.model(query_X)
            score = F.softmax(predicts.detach_(), dim=-1)
            max_probs, predicts = torch.max(score, dim=-1)
            predicts = predicts.cpu().detach().numpy()
            
            # # print(predicts)
            start_acc = np.mean(predicts == query_y)

        if show_detail:
            acc_list = []
        for idx in range(self.max_iter):
            self.model.eval()
            pseudo_preds = self.model(unlabel_X)
            score = F.softmax(pseudo_preds.detach_(), dim=-1)
            max_probs, pseudo_y = torch.max(score, dim=-1)

            y = torch.cat([support_y, pseudo_y])
            Y = self.label2onehot(y, way)
            y_hat = np.dot(X_hat, Y)
            support_set = self.expand(support_set, X_hat, y_hat, way, num_support, pseudo_y,
                                      embeddings, y)
            # y = np.argmax(Y, axis=1)
            # y = support_y
            self.model = LogisticRegressionModel(self.input_dim, self.num_class).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            # self.train(embeddings[support_set], y[support_set], epoch = 50 + idx*10)
            self.train(embeddings[support_set], y[support_set])

            if show_detail:
                predicts = self.model(query_X)
                score = F.softmax(predicts.detach_(), dim=-1)
                max_probs, predicts = torch.max(score, dim=-1)
                predicts = predicts.cpu().detach().numpy()
                
                acc_list.append(np.mean(predicts == query_y))
            if len(support_set) == len(embeddings):
                break
        
        self.model.eval()
        predicts = self.model(query_X)
        score = F.softmax(predicts.detach_(), dim=-1)
        max_probs, predicts = torch.max(score, dim=-1)
        predicts = predicts.cpu().detach().numpy()
        if show_detail:
            # end_acc = np.mean(predicts == query_y)
            # print("Final Improvements: %.2f" % (end_acc-start_acc))
            acc_list.append(np.mean(predicts == query_y))
            return acc_list
        return predicts


    def expand(self, support_set, X_hat, y_hat, way, num_support, pseudo_y, embeddings, targets):
        _, coefs, _ = self.elasticnet.path(X_hat, y_hat, l1_ratio=1.0)
        coefs = np.sum(np.abs(coefs.transpose(2, 1, 0)[
                       ::-1, num_support:, :]), axis=2)
        selected = np.zeros(way)
        for gamma in coefs:
            for i, g in enumerate(gamma):
                if g == 0.0 and \
                    (i+num_support not in support_set) and \
                        (selected[pseudo_y[i]] < self.step):
                    support_set.append(i+num_support)
                    selected[pseudo_y[i]] += 1
            if np.sum(selected >= self.step) == way:
                break
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
            self.classifier = LogisticRegression(
                C=10, multi_class='auto', solver='lbfgs', max_iter=1000)

    def label2onehot(self, label, num_class):
        result = np.zeros((label.shape[0], num_class))
        for ind, num in enumerate(label):
            result[ind, num] = 1.0
        return result
