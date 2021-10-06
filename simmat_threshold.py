import numpy as np 
import pandas as pd
import scipy 
import pickle
from scipy.spatial import distance as scidist
import sys
sys.path.insert(1, "D:\\Work\\Research\\")
from src.evaluate import distance
from sklearn.cluster import KMeans
import time
import os, shutil

class SMG:
    def __init__(self, dataset):
        file_reader = open(dataset, 'rb')
        self.dataset = pickle.load(file_reader)
        if(type(self.dataset) != pd.core.frame.DataFrame):
            self.dataset = pd.DataFrame(self.dataset)
        self.feature_centroids = {}
        self.simmat = {}
        self.class_index = {}
        self.class_list = []
        self.gen_index()
    
    def gen_feature_centroids(self):
        model = KMeans(n_clusters=1)
        self.class_list = self.dataset['cls'].unique()
        for name in self.class_list:
            if(name not in self.feature_centroids.keys()):
                X = np.array(self.dataset.loc[self.dataset['cls'] == name]['hist'])
                X = np.vstack(X)
                model.fit(X)
                self.feature_centroids[name] = model.cluster_centers_
    
    def gen_similarity_matrix(self):
        model = KMeans(n_clusters=1)
        for index, name in enumerate(self.class_list):
            class_similarity = np.empty((len(self.class_list)))
            for second_index, second_name in enumerate(self.class_list):
                class_similarity[second_index] = distance(self.feature_centroids[name], self.feature_centroids[second_name], d_type="cosine")
            self.simmat[name] = class_similarity
            
    def gen_index(self):
        self.gen_feature_centroids()
        self.gen_similarity_matrix()

start_time = time.time()
test = SMG("D:\\Work\\Research\\Features - Base\\resnet-80")
print(round(time.time() - start_time, 1))
