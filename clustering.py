import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
df = pd.read_csv('/content/gender_classification_v7.csv') 
df.head()
df.info() 
df.dropna(inplace=True) 
g=[]
for i in df['gender']: 
    if i=='Male':
        g.append(1) 
    else:
        g.append(0)
df['gender'] = g 
df.head()
plt.scatter(df['long_hair'], df['gender']) 
plt.ylabel('gender') 
plt.xlabel('long_hair')
plt.show()
plt.scatter(df['forehead_width_cm'], df['gender']) 
plt.ylabel('gender') 
plt.xlabel('forehead_width_cm')
plt.show()
plt.scatter(df['forehead_height_cm'], df['gender']) 
plt.ylabel('gender') 
plt.xlabel('forehead_height_cm')
plt.show()
plt.scatter(df['nose_wide'], df['gender']) 
plt.ylabel('gender') 
plt.xlabel('nose_wide')
plt.show()
plt.scatter(df['nose_long'], df['gender']) 
plt.ylabel('gender') 
plt.xlabel('nose_long')
plt.show()
plt.scatter(df['lips_thin'], df['gender']) 
plt.ylabel('gender') 
plt.xlabel('lips_thin')
plt.show()
plt.scatter(df['distance_nose_to_lip_long'], df['gender']) 
plt.ylabel('gender') 
plt.xlabel('distance_nose_to_lip_long')
plt.show()
X = df.drop(['gender'], axis=1) 
y = df['gender']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
sse=[]
for i in range(1,10):
    model = KMeans(n_clusters=i) 
    model.fit(X_train, y_train) 
    sse.append(model.inertia_)
plt.plot(range(1,10),sse) 
plt.xlabel('K value') 
plt.ylabel('SSE') 
plt.show()
# Hence n_clusters=4
model = KMeans(n_clusters=4) 
model.fit(X_train, y_train) 
y_pred = model.predict(X_test)
y_pred
df2 = X_test 
df2['preds'] = y_pred 
df2.head()
d1 = df2[df2.preds==0] 
d2 = df2[df2.preds==1] 
d3 = df2[df2.preds==2] 
d4 = df2[df2.preds==3]
plt.scatter(d1.forehead_height_cm, d1.preds, color='green') 
plt.scatter(d2.forehead_height_cm, d2.preds, color='red') 
plt.scatter(d3.forehead_height_cm, d3.preds, color='yellow') 
plt.scatter(d4.forehead_height_cm, d4.preds, color='blue') 
plt.show()
plt.scatter(d1.forehead_width_cm, d1.preds, color='green') 
plt.scatter(d2.forehead_width_cm, d2.preds, color='red') 
plt.scatter(d3.forehead_width_cm, d3.preds, color='yellow') 
plt.scatter(d4.forehead_width_cm, d4.preds, color='blue') 
plt.show()

from sklearn.datasets import make_blobs
df = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.6, random_state=50) 
points=df[0]

import scipy.cluster.hierarchy as hr
from sklearn.cluster import AgglomerativeClustering 
dendrogram = hr.dendrogram(hr.linkage(points, method='ward')) 
plt.scatter(df[0][:,0], df[0][:,1])
plt.show()
hc = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(points)
plt.scatter(points[y_hc==0,0], points[y_hc==0,1], color='green') 
plt.scatter(points[y_hc==1,0], points[y_hc==1,1], color='red') 
plt.scatter(points[y_hc==2,0], points[y_hc==2,1], color='yellow') 
plt.scatter(points[y_hc==3,0], points[y_hc==3,1], color='blue')
plt.show()