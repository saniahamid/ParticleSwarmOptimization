# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 01:52:26 2018

@author: Sania Hamid
"""
from time import time
import gzip, pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
from scipy.spatial import distance

import operator
import random
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

# Load the dataset
with gzip.open('mnist.pkl.gz','rb') as ff :
    u = pickle._Unpickler( ff )
    u.encoding = 'latin1'
    train_set, valid_set, test_set = u.load()
    print( train_set[0].shape, train_set[1].shape )
    print( valid_set[0].shape, valid_set[1].shape )
    print( test_set[0].shape, test_set[1].shape )
    
X_train = train_set[0][0:100]
Y_train = train_set[1]

w = 0.79
c1 = 1.49
c2 = 1.49

creator.create("FitnessMax",base.Fitness, weights = (1.0,))
creator.create("Particle",np.ndarray,fitness = creator.FitnessMax, speed = list, smin = None, smax = None, best = None,
               n_clusters = 20, clusters = dict, centroids = list )

def generate(smin,smax):
    part = creator.Particle([0,1,2,3,4,5,6,7,8,9])
    part.speed = np.random.uniform(smin, smax, 784)
    part.smin = smin
    part.smax = smax
    
    for i in range(20):
        part.clusters[str(i)] = []
        part.centroids.append(np.random.rand(1,784))
    return part

def updateParticle(part, best, phi1, phi2):
    part.speed = w * part.speed + c1 * (part.best - part) + c2 * (best - part)
    part = part + part.speed
    for centroid_loc in part.centroids:
        centroid_loc += part.speed

def evaluate(ind):
    ind.clusters = init_clusters()
    for i in range(100):
        nearest_clust = get_nearestclust_cluster(ind, X_train[i])
        ind.clusters[nearest_clust].append(X_train[i])
      
    quant_error = 0
    for i in range(20):
        C = ind.clusters[str(i)]
        for j in range(len(C)):
            quant_error += (distance.euclidean(C[j], ind.centroids[i])/len(C))
    err_result = quant_error/20
    return (err_result, 1./20)

def init_clusters():
    clusters = {}
    for i in range(20):
        clusters[str(i)] = []
    return clusters

def get_nearestclust_cluster(ind, image_data):
    dist = []
    for clust in ind.centroids:
        dist.append(distance.euclidean(clust,image_data))
    return (str(dist.index(min(dist))))

def SSE(particle):
    clust_summation = 0
    particle.clusters = init_clusters()
    for i in range(100):
        nearest_clust = get_nearestclust_cluster(particle, X_train[i])
        particle.clusters[nearest_clust].append(X_train[i])
    #summation = 0
    
    for i in range(20):
        image_data = particle.clusters[str(i)]
        for cell in image_data:
            clust_summation += distance.euclidean(cell, particle.centroids[i])
    return clust_summation

toolbox = base.Toolbox()
toolbox.register("particle", generate,smin=-6, smax=10)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)
toolbox.register("evaluate", evaluate)


def main():
    pop = toolbox.population(n=20)
    stats = tools.Statistics(lambda indi: indi.fitness.values)
    stats.register("avg",np.mean)
    stats.register("std",np.std)
    stats.register("min",np.min)
    stats.register("max",np.max)
    

    
    GEN = 100
    best = None
    
    for g in range(GEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if part.best is None or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if best is None or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
                best.centroids = part.centroids
        #for part in pop:
            
        print("Some of Squared Error: " + str(SSE(best)))
        for i in range(20):
            print(str(i) + ": " + str(len(best.clusters[str(i)])))
        return pop, best

if __name__ == "__main__":
    main()   
            
    
    
    
        
    
    
    
