import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

x = np.array ([[1,2],
               [1.5 , 1.8],
               [4,8],[3,9]])
#plt.scatter(x[:,0],x[:,1], color = 'g')
#plt.show()

clf = KMeans(n_clusters=4)
clf.fit(x)
centroids = clf.cluster_centers_ #these are attributes of kmeans
labels = clf.labels_ #these are attributes of kmeans
colors = ["g."]

for i in range(len(x)):
    plt.plot(x[i][0], x[i][1],  markersize = 5)
plt.scatter(centroids[:,0],centroids[:,1],marker ='x',s = 150,linewidth = 5)
plt.show()
