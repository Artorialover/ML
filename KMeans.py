import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X=np.random.rand(100,2)

model=KMeans(n_clusters=3)
model.fit(X)


labels=model.labels_
cen=model.cluster_centers_
print(cen)

plt.scatter(X[:,0],X[:,1],c=labels,cmap='viridis')
plt.scatter(cen[:,0],cen[:,1],marker='*',color='red')
plt.show()


