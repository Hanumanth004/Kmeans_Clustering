import sys
import numpy as np
import matplotlib.pyplot as plt
import csv


count = 0
sample=[]
X_pca=[]

K=int(sys.argv[2])
max_iter=int(sys.argv[3])

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
X_tmp = Y - mean_extd
X_tmp = np.matrix(X_tmp)
std_extd = np.matrix(std_extd)
X_normalized = np.dot(X_tmp,std_extd)
X_normalized = np.array(X_normalized)

new_means=[]
means=np.random.random((K,X.shape[1]))
new_means=np.zeros_like(means)
num_assigned=np.zeros((K,1))



#Perform K-means!
for k in xrange(max_iter):
    for i in xrange(X.shape[0]):
        X_tmp = X_normalized[i]
        X_tile = np.tile(X_tmp, (K,1))
        y=X_tile-means
        y=np.square(y)
        y1=np.sum(y,axis=1)
        mn=np.argmin(y1)
        new_means[mn]+=X_tmp
        num_assigned[mn]+=1

    for i in range(K):
        if(num_assigned[i] > 0):
            new_means[i]=new_means[i]/num_assigned[i]

    mean_diff=new_means-means
    mean_diff_sqr=np.square(mean_diff)
    mean_diff_sum=np.sum(mean_diff_sqr)
    mean_diff_sqrt=np.sqrt(mean_diff_sum)
    print mean_diff_sqrt

    means = new_means
    num_assigned=np.zeros((K,1))
    new_means=np.zeros_like(means)


cluster_num=np.zeros((X.shape[0],1))
num_elements=np.zeros((K,1))

for i in xrange(X.shape[0]):
    X_tmp = X_normalized[i]
    X_tile = np.tile(X_tmp, (K,1))
    y=X_tile-means
    y=np.square(y)
    y1=np.sum(y,axis=1)
    mn=np.argmin(y1)
    cluster_num[i]=mn
    num_elements[mn]+=1


X_cluster=[]
temp=[]

for i in xrange(K):
    for k in range(X.shape[0]):
        if (cluster_num[k]==i):
            X_cluster.append(X[k])


intra_clust_dist=np.zeros((K,1))

max_dist=0.0

X_cluster=np.array(X_cluster)
X_cluster=X_cluster.astype(np.float)

for i in xrange(K):
    for j in xrange(num_elements[i]):
        for k in xrange(num_elements[i]):
            y=X_cluster[j] - X_cluster[k]
            y=np.square(y)
            y1=np.sum(y)
            y1=np.sqrt(y1)
            if max_dist < y1:
                max_dist = y1
    intra_clust_dist[i] = max_dist

maximum = np.amax(intra_clust_dist)
print "Intra cluster distance!"
print maximum


min_dist = float('inf') 
offset = 0
min_distance=np.zeros((K,K))
min_dist_temp=[]
temp = 0
for m in xrange(K-1):
    for i in xrange(K-m-1):
        offset += num_elements[i+m]
        offset = int(offset)
        for j in xrange(num_elements[m]):
            for k in xrange(num_elements[i+m+1]):
                y=X_cluster[j+temp] - X_cluster[k+offset]
                y=np.square(y)
                y1=np.sum(y)
                y1=np.sqrt(y1)
                if min_dist >= y1:
                    min_dist = y1
        min_dist_temp.append(min_dist)

    temp+=num_elements[m]
    temp=int(temp)
    offset=temp

minimum = np.amin(min_dist_temp)
print "Inter cluster distance!"
print minimum


P,D,Q=np.linalg.svd(X_normalized)
temp=Q.T
X_pca = np.dot(X_normalized,temp[:,0:2])

fig = plt.figure(figsize=(8,8))
plt.scatter(X_pca[:,0],X_pca[:,1],c=cluster_num)
plt.title("K-means without PCA K=20, max_iter=20")
plt.show()
