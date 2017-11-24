#!/usr/bin/python

import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

count = 0
sample=[]
X_pca=[]

with open(sys.argv[1], 'rb') as f:
    reader=csv.reader(f.read().splitlines())
    for row in reader:
        sample.append(row)

def dimension_reduce(X_tr, W_eig):
    X_pca=np.dot(X_tr, W_eig)
    return X_pca

#Data Normalization!
X=np.array(sample)
Y=X.astype(np.float)
mean=np.mean(Y,axis=0)
std=np.std(Y,axis=0)
mean_extd=np.tile(mean,(X.shape[0],1))
std_reciprocal=np.reciprocal(std)
std_extd=np.diag(std_reciprocal)
X_tmp=Y-mean_extd
X_tmp=np.matrix(X_tmp)
std_extd=np.matrix(std_extd)
X_normalized=np.dot(X_tmp,std_extd)


#Compute Eigenvalue and Eigenvectors!
P,D,Q=np.linalg.svd(X_normalized)
D=np.square(D)
D_sum=np.sum(D)

for i in xrange(D.shape[0]):
    print (D[i]/D_sum)*100

#Figure out total principal components we want to keep!
k=0
temp=0
for i in range(D.shape[0]):
    temp = temp + (D[i]/D_sum)*100
    k = k+1
    if (temp > 99.0):
        break;


#Plot first two principal components!
temp = Q.T
X_tmp = temp[:,0:k]
X_dim_reduced = dimension_reduce(X_normalized, X_tmp)

fig = plt.figure(figsize=(8,8))
plt.scatter([X_dim_reduced[:,0]],[X_dim_reduced[:,1]],c='r')
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
plt.show()





