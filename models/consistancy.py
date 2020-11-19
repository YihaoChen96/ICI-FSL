import math
import random

import numpy as np
import sklearn
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import normalize

from .transforms import TransformFix

class FixmatchPick(object):

    def __init__(self, img_size, classifier='lr', num_class=None, step=5, max_iter='auto',
                 reduce='pca', d=5, norm='l2', mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], fixmatch_threshold = 0.95):
        self.step = step
        self.max_iter = max_iter
        self.num_class = num_class
        self.initial_embed(reduce, d)
        self.initial_norm(norm)
        self.initial_classifier(classifier)
        self.transform_fix = TransformFix(img_size, mean, std)
        self.elasticnet = ElasticNet(alpha=1.0, l1_ratio=1.0, fit_intercept=True,
                                         normalize=True, warm_start=True, selection='cyclic')
        self.threshold = fixmatch_threshold
        self.support_X, self.support_y = None, None
    def fit(self, X, y):
        self.support_X = self.norm(X)
        self.support_y = y

    def predict(self, X, unlabel_X, strong_X, weak_X, show_detail=False, query_y=None):
        
        # Assert call fit before predict
        assert self.support_X is not None and self.support_y is not None, "Need to call function 'fit' before 'predict'"
        
        # Get support data and numbers
        support_X, support_y = self.support_X, self.support_y
        way, num_support = self.num_class, len(support_X)
        
        # Normalize data
        query_X = self.norm(X)

        unlabel_X = self.norm(unlabel_X)
        weak_X = self.norm(weak_X)
        strong_X = self.norm(strong_X)
        
        # TODO: PROBLEM!!! If there's no unlabeled data, 
        # its number will be replaced by the number of query data.
        # Is this correct?
        num_unlabel = unlabel_X.shape[0]
        embeddings = np.concatenate([support_X, unlabel_X])

        # PROBLEM!!! 
        # Not quite sure what this step mean?
        if self.max_iter == 'auto':
            # set a big number
            self.max_iter = num_support + num_unlabel
        elif self.max_iter == 'fix':
            self.max_iter = math.ceil(num_unlabel/self.step)
        else:
            assert float(self.max_iter).is_integer()
        
        support_set = np.arange(num_support).tolist()

        # Train the classifier on the support set
        self.classifier.fit(self.support_X, self.support_y)
        
        # Define the accuracy list
        if show_detail:
            acc_list = []
        

        for _ in range(self.max_iter):
            if show_detail:
                predicts = self.classifier.predict(query_X)
                acc_list.append(np.mean(predicts == query_y))

            # Get pseudo labels for the unlabeled set
            pseudo_y = self.classifier.predict(unlabel_X)
            # weak_y = self.classifier.predict(weak_X)
            weak_prob = self.classifier.predict_proba(weak_X)
            # strong_y = self.classifier.predict(strong_X)
            strong_prob = self.classifier.predict_proba(strong_X)

            # Transform the labels into one-hot encoding
            y = np.concatenate([support_y, pseudo_y])
            
            # Add new data to support set
            support_set = self.expand(support_set, way, num_support, pseudo_y, weak_prob, strong_prob)

            # Fit the classifier on expanded embeddings and labels
            self.classifier.fit(embeddings[support_set], y[support_set])

            # If all data were expanded
            if len(support_set) == len(embeddings):
                break
        
        # Use the final classifier to do the prediction
        predicts = self.classifier.predict(query_X)
        if show_detail:
            acc_list.append(np.mean(predicts == query_y))
            return acc_list
        return predicts


    def expand(self, support_set, way, num_support, pseudo_y, weak_prob, strong_prob):
        weak_argmax = np.argmax(weak_prob, axis = 1)
        strong_argmax = np.argmax(strong_prob, axis = 1)
        

        # Determin for each row, whether the prediciton is above the threshold or not
        # passed = weak_prob[np.arange(len(weak_prob)), weak_argmax[weak_argmax == strong_argmax]] > self.threshold

        # Init array to store how many data are selected for each class
        selected_per_class = np.zeros(way)

        # Randomly choose 
        # num2select = min((way*self.step), len(weak_argmax))


        i = 0

        for i in range(len(weak_argmax)):
            if (i+num_support not in support_set) \
                    and (weak_argmax[i] == strong_argmax[i]) and (weak_prob[i, weak_argmax[i]] > self.threshold):
                
                support_set.append(i+num_support)
                selected_per_class[weak_argmax[i]] += 1

            # If already each class has more than num_step data selected  
            if np.sum(selected_per_class >= self.step) == way:
                break

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
            self.classifier = LogisticRegression(
                C=10, multi_class='auto', solver='lbfgs', max_iter=1000)

    def label2onehot(self, label, num_class):
        result = np.zeros((label.shape[0], num_class))
        for ind, num in enumerate(label):
            result[ind, num] = 1.0
        return result