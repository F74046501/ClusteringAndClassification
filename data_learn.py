
# coding: utf-8

# In[1]:


import requests
import json
import pandas as pd

#分群 k-means
from sklearn import cluster, datasets

#決策樹
from sklearn import tree
from sklearn.cross_validation import train_test_split

#績效
from sklearn import metrics

#k-Nearest Neighbors 
from sklearn import neighbors
from sklearn.cross_validation import train_test_split
#for drawing
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#各種data的建立 是從網頁page source拿過來的, 資料來源：https://worldpenis.tadaa-data.de/index.html
region = ["Central Asia","Europe","Africa","Africa","South America","Europe","Australia","Europe","Western Asia","Southeast Asia","Europe","Europe","Central America/Caribean","Asia","South America","Europe","South America","Europe","Africa","Southeast Asia","Africa","North America","Africa","Africa","Africa","South America","Asia","South America","Africa","Central America/Caribean","Africa","Europe","Central America/Caribean","Western Asia","Europe","Europe","Central America/Caribean","Africa","South America","Africa","Central America/Caribean","Europe","Africa","Europe","Europe","Africa","Europe","Europe","Africa","Europe","North America","Central America/Caribean","South America","Central America/Caribean","Pacific Islands","Central America/Caribean","Asia","Europe","Europe","South Asia","Southeast Asia","South Asia","South Asia","Europe","Western Asia","Europe","Central America/Caribean","Asia","Western Asia","Central Asia","Africa","Pacific Islands","Asia","Asia","Western Asia","Asia","Europe","Western Asia","Africa","Europe","Europe","Europe","North America","Europe","Asia","Europe","Africa","Asia","Asia","Europe","Pacific Islands","Australia","Europe","Western Asia","South Asia","Western Asia","Central America/Caribean","Asia","South America","South America","Southeast Asia","Europe","Europe","Western Asia","Europe","Asia","Pacific Islands","Western Asia","Europe","Africa","Europe","Southeast Asia","Western Asia","Europe","Europe","Pacific Islands","Africa","Europe","South Asia","Africa","South America","Europe","Europe","Asia","Africa","Southeast Asia","Pacific Islands","Africa","Western Asia","Western Asia","Western Asia","Europe","Europe","North America","South America","Western Asia","South America","Southeast Asia","Western Asia"]
country = ["Afghanistan","Albania","Algeria","Angola","Argentina","Armenia","Australia","Austria","Bahrein","Bangladesh","Belarus","Belgium","Belize","Bhutan","Bolivia","Bosnia and Herzegovina","Brazil","Bulgaria","Burkina Faso","Cambodia","Cameroon","Canada","Cape Verde","Central African Rep.","Chad","Chile","China","Colombia","Congo-Brazzaville","Costa Rica","Cote d'Ivoire","Croatia","Cuba","Cyprus","Czech Republic","Denmark","Dominican Republic","DR Congo","Ecuador","Egypt","El Salvador","Estonia","Ethiopia","Finland","France","Gambia","Georgia","Germany","Ghana","Greece","Greenland","Guatemala","Guyana","Haiti","Hawaii","Honduras","Hong Kong SAR, China","Hungary","Iceland","India","Indonesia","Iran","Iraq","Ireland","Israel","Italy","Jamaica","Japan","Jordan","Kazakhstan","Kenya","Kiribati","Korea, North","Korea, Rep.","Kuwait","Laos","Latvia","Lebanon","Libya","Lithuania","Luxembourg","Macedonia, FYR","Mexico","Moldova","Mongolia","Montenegro","Morocco","Myanmar","Nepal","Netherlands","New Caledonia","New Zealand","Norway","Oman","Pakistan","Palestine","Panama","Papua New Guinea","Paraguay","Peru","Philippines","Poland","Portugal","Qatar","Romania","Russian Federation","Samoa","Saudi Arabia","Scotland","Senegal","Serbia","Singapore","Syria","Slovak Republic","Slovenia","Solomon Islands","South Africa","Spain","Sri Lanka","Sudan","Suriname","Sweden","Switzerland","Taiwan","Tanzania","Thailand","Tonga","Tunisia","Turkey","Turkmenistan","United Arab Emirates","Ukraine","United Kingdom","United States","Uruguay","Uzbekistan","Venezuela, RB","Vietnam","Yemen, Rep."]
dick_length_soft = [9.5,9.8,9.9,10,9.4,10.5,10,11.7,8.8,8.2,11.2,12.1,10.9,8.1,9.2,12.2,9.9,11.7,10.3,7.2,10.6,9.5,9.6,9.8,9.8,9.2,8.8,10.5,11,9.6,10.4,11.8,11,8.3,11.3,12,11.1,11.4,11.1,9.4,8.8,11,9.2,9.2,9.9,10.3,9.4,9.9,11,8.9,9.4,9.2,9.3,10.8,9.1,8.6,8.2,10,9.9,8.8,8.5,8.5,8.3,8.7,9.3,8.5,11.3,8.6,9.2,9.3,10.4,9.8,7,7.9,8.3,7.4,10,10.7,9.3,9.9,9.4,9.5,9.6,9.4,8.7,9.2,9.4,7.4,7.3,9.9,9.2,10.1,9.8,8.7,8.3,10.3,10.7,9.9,9.8,9.5,7.9,9.8,10.3,8.4,8.7,9.6,8.9,8.4,9.7,10.1,10.1,8.1,8.9,9.7,9.5,9.1,9.7,9.2,7.4,11.4,8.4,9.8,9.7,8.5,8.4,6.9,9,9.9,9.3,9.2,8.5,9.7,9.7,9.6,9.2,9.1,10.6,7.4,8.6]
dick_length_hard = [13.69,14.19,14.49,15.73,14.88,13.12,15.7,14.69,12.93,11.2,13.98,15.14,15.75,11.12,14.7,15.27,15.7,14.66,15.19,9.84,16.65,13.92,14.05,15.33,15.39,14.59,12.9,16.75,17.33,13.81,15.22,14.77,15.87,12.17,14.17,15.07,15.99,17.93,17.59,13.85,12.71,13.78,13.53,13.59,14.5,15.07,13.87,14.48,17.31,12.18,13.87,13.24,14.75,15.67,13.21,12.4,11.19,14.68,14.56,12.93,11.67,11.58,11.35,12.78,13.6,12.5,16.3,12.6,13.5,13.46,16.28,14.19,9.6,10.8,11.38,10.14,14.69,16.82,13.74,14.55,13.82,13.98,14.09,13.76,12.77,13.56,13.86,10.1,9.98,15.6,13.24,15.79,14.34,12.85,11.4,15.08,15.49,13.62,15.53,15.07,10.85,15.41,15.14,12.41,12.73,14.16,12.87,12.4,14.3,15.89,14.78,11.16,13.1,14.19,14.01,13.14,15.29,13.58,10.18,17.95,13.34,15.36,14.25,12.5,11.5,9.43,12.94,14.61,13.7,13.48,12.54,14.2,14.3,14.15,14.67,13.43,16.93,10.15,12.7]
dick_v_soft = [62.6,73.38,62.4,73.34,59.25,61.8,87.73,87.6,51.79,42.81,73.81,108.19,88.48,44.4,56.69,89.47,81.96,80.53,78.72,42.38,82.67,72.6,76.39,74.9,74.9,67.47,55.47,85.24,92.87,60.51,81.11,101.56,80.67,52.32,77.77,107.3,81.41,101.93,97.38,71.84,50.6,87.54,70.31,57.99,93.6,72.42,63.31,77.21,92.87,63.92,71.84,55.41,56.02,77.56,59.97,51.8,48.26,70.31,72.61,59.27,56.01,50.03,54.7,70.62,59.95,63.64,81.16,57.92,61.97,61.29,82.76,61.77,50.27,52.06,51.15,44.57,71.82,78.47,65.39,63.81,61.94,69.67,60.51,64.7,50.02,66.07,74.8,42.55,52.43,91.89,63.32,85.27,85.98,52.4,58.36,59.22,72.07,65.24,57.68,59.88,45.42,67.45,66.39,54.14,46.55,67.5,63.92,54.14,66.76,78.77,86.93,48.79,52.38,65.33,66.8,61.29,62.52,68.88,45.6,101.93,66.85,81.14,80.31,69,57.81,38.74,56.73,78.78,54.74,49.23,52.38,74.13,74.13,71.88,70.31,61.29,87.76,46.64,48.29]
dick_v_hard = [142.08,166.97,138.76,174.89,155.24,121.33,218.35,171.15,128.84,97.89,144.07,211.84,207.6,94.06,149.63,177.32,210.82,159.42,179.33,96.83,200.13,168.13,171.36,177.72,179.9,175.08,127.39,224.57,223.83,139.09,182.35,198.33,185.82,130.09,153.04,210.86,188.46,242.99,252.85,161.63,116.88,172.44,159.73,135.66,214.36,163.6,144.96,177.17,225.33,137.72,166.98,127.49,144.1,182.57,139.02,120.49,101.95,162.94,167.12,135.84,129.09,126.58,115.94,163.77,148.15,147.81,187.1,132.6,151.88,137.74,196.96,144.95,107.46,109.74,119.14,95.17,166.38,205.47,149.16,147.06,142.92,159.4,140.15,148.36,114.4,154.35,169.86,90.99,111.9,227.93,146.95,212.35,197.33,131.5,125.03,145.73,168.45,139.87,150.9,156.13,105.04,165.58,152.75,134.72,106.43,155.04,149.95,133.01,154.45,188.82,198.16,113.2,130.53,148.3,153.14,143.38,150.73,160.58,97.84,244.38,175.56,199.64,186.66,158.17,120.82,88.02,131.72,177.61,136.51,121.67,130.6,170.4,169.93,168.42,184.46,151.85,228.74,107.75,119.85]
dick_circle_soft = [9.1,9.7,8.9,9.6,8.9,8.6,10.5,9.7,8.6,8.1,9.1,10.6,10.1,8.3,8.8,9.6,10.2,9.3,9.8,8.6,9.9,9.8,10,9.8,9.8,9.6,8.9,10.1,10.3,8.9,9.9,10.4,9.6,8.9,9.3,10.6,9.6,10.6,10.5,9.8,8.5,10,9.8,8.9,10.9,9.4,9.2,9.9,10.3,9.5,9.8,8.7,8.7,9.5,9.1,8.7,8.6,9.4,9.6,9.2,9.1,8.6,9.1,10.1,9,9.7,9.5,9.2,9.2,9.1,10,8.9,9.5,9.1,8.8,8.7,9.5,9.6,9.4,9,9.1,9.6,8.9,9.3,8.5,9.5,10,8.5,9.5,10.8,9.3,10.3,10.5,8.7,9.4,8.5,9.2,9.1,8.6,8.9,8.5,9.3,9,9,8.2,9.4,9.5,9,9.3,9.9,10.4,8.7,8.6,9.2,9.4,9.2,9,9.7,8.8,10.6,10,10.2,10.2,10.1,9.3,8.4,8.9,10,8.6,8.2,8.8,9.8,9.8,9.7,9.8,9.2,10.2,8.9,8.4]
dick_circle_hard = [11.42,12.16,10.97,11.82,11.45,10.78,13.22,12.1,11.19,10.48,11.38,13.26,12.87,10.31,11.31,12.08,12.99,11.69,12.18,11.12,12.29,12.32,12.38,12.07,12.12,12.28,11.14,12.98,12.74,11.25,12.27,12.99,12.13,11.59,11.65,13.26,12.17,13.05,13.44,12.11,10.75,12.54,12.18,11.2,13.63,11.68,11.46,12.4,12.79,11.92,12.3,11,11.08,12.1,11.5,11.05,10.7,11.81,12.01,11.49,11.79,11.72,11.33,12.69,11.7,12.19,12.01,11.5,11.89,11.34,12.33,11.33,11.86,11.3,11.47,10.86,11.93,12.39,11.68,11.27,11.4,11.97,11.18,11.64,10.61,11.96,12.41,10.64,11.87,13.55,11.81,13,13.15,11.34,11.74,11.02,11.69,11.36,11.05,11.41,11.03,11.62,11.26,11.68,10.25,11.73,12.1,11.61,11.65,12.22,12.98,11.29,11.19,11.46,11.72,11.71,11.13,12.19,10.99,13.08,12.86,12.78,12.83,12.61,11.49,10.83,11.31,12.36,11.19,10.65,11.44,12.28,12.22,12.23,12.57,11.92,13.03,11.55,10.89]


# In[3]:


#把data用.csv檔的格式存入，之後方便做出dataset
with open('data.csv', 'wt') as f:
    #這是屬性列
    print('dick_length_soft//dick_length_hard//dick_v_soft//dick_v_hard//dick_circle_soft//dick_circle_hard',file=f)
    index = 0
    for i in country:
        #這是data列
        print(dick_length_soft[index],end='//', file=f)#end='//'代表用//分隔資料不是用預設的，
        print(dick_length_hard[index],end='//', file=f)
        print(dick_v_soft[index],end='//', file=f)
        print(dick_v_hard[index],end='//', file=f)
        print(dick_circle_soft[index],end='//', file=f)
        print(dick_circle_hard[index],end='\n', file=f)
        index = index + 1


# In[4]:


#read in .csv file, and show it
dick = pd.read_csv('data.csv',sep="//")
print(dick)


# In[5]:


#count how many region are in the data
#統計出總共有哪些區域 方便等等去做每個區域的預測精準度分析
region_index = 0
all_region = []
for i in country:
    tmp = region[region_index]
    region_exist = 0
    for i in all_region:
        if(i == tmp):
            region_exist = 1
    if(region_exist==0):
        all_region.append(tmp)
    region_index = region_index + 1
    
print('資料裡面存在有',len(all_region),'個區域，分別是：')
print(all_region)


# In[6]:


#K-means
#把dataframe轉成numpy.ndarray
dick_array = dick.values
#call func,去做k means,想要做出5個分類來看看（因為世界有五大洲）
kmeans_fit = cluster.KMeans(n_clusters = 5).fit(dick_array.data)
#得到分類結果
cluster_labels = kmeans_fit.labels_

Central_Asia = ['Central Asia']
Europe = ['Europe']
Africa = ['Africa']
South_America = ['South America']
Australia = ['Australia']
Western_Asia = ['Western Asia']
Southeast_Asia = ['Southeast Asia']
Central_America_Caribean = ['Central America/Caribean']
Asia = ['Asia']
North_America = ['North America']
Pacific_Islands = ['Pacific Islands']
South_Asia = ['South Asia']

kmeans_index = 0
for i in country:
    if(region[kmeans_index] == 'Central Asia'):
        Central_Asia.append(cluster_labels[kmeans_index])
    elif(region[kmeans_index] == 'Europe'):
        Europe.append(cluster_labels[kmeans_index])
    elif(region[kmeans_index] == 'Africa'):
        Africa.append(cluster_labels[kmeans_index])
    elif(region[kmeans_index] == 'South America'):
        South_America.append(cluster_labels[kmeans_index])
    elif(region[kmeans_index] == 'Australia'):
        Australia.append(cluster_labels[kmeans_index])
    elif(region[kmeans_index] == 'Western Asia'):
        Western_Asia.append(cluster_labels[kmeans_index])
    elif(region[kmeans_index] == 'Southeast Asia'):
        Southeast_Asia.append(cluster_labels[kmeans_index])
    elif(region[kmeans_index] == 'Central America/Caribean'):
        Central_America_Caribean.append(cluster_labels[kmeans_index])
    elif(region[kmeans_index] == 'Asia'):
        Asia.append(cluster_labels[kmeans_index])
    elif(region[kmeans_index] == 'North America'):
        North_America.append(cluster_labels[kmeans_index])
    elif(region[kmeans_index] == 'Pacific Islands'):
        Pacific_Islands.append(cluster_labels[kmeans_index])
    elif(region[kmeans_index] == 'South Asia'):
        South_Asia.append(cluster_labels[kmeans_index])
    kmeans_index = kmeans_index + 1
print("分群結果：")
print(Central_Asia)
print(South_Asia)
print(Asia)
print(Western_Asia)
print(Southeast_Asia)
print(Australia)
print(Pacific_Islands)
print(Africa)
print(Europe)
print(South_America)
print(North_America)
print(Central_America_Caribean)


# In[7]:


#把要用的陣列恢復原狀
Central_Asia = ['Central Asia']
Europe = ['Europe']
Africa = ['Africa']
South_America = ['South America']
Australia = ['Australia']
Western_Asia = ['Western Asia']
Southeast_Asia = ['Southeast Asia']
Central_America_Caribean = ['Central America/Caribean']
Asia = ['Asia']
North_America = ['North America']
Pacific_Islands = ['Pacific Islands']
South_Asia = ['South Asia']


# In[8]:


# Hierarchical Clustering 演算法
hclust = cluster.AgglomerativeClustering(linkage = 'ward', affinity = 'euclidean', n_clusters = 5)

# 印出分群結果
hclust.fit(dick_array)
cluster_labels = hclust.labels_

hclust_index = 0
for i in country:
    if(region[hclust_index] == 'Central Asia'):
        Central_Asia.append(cluster_labels[hclust_index])
    elif(region[hclust_index] == 'Europe'):
        Europe.append(cluster_labels[hclust_index])
    elif(region[hclust_index] == 'Africa'):
        Africa.append(cluster_labels[hclust_index])
    elif(region[hclust_index] == 'South America'):
        South_America.append(cluster_labels[hclust_index])
    elif(region[hclust_index] == 'Australia'):
        Australia.append(cluster_labels[hclust_index])
    elif(region[hclust_index] == 'Western Asia'):
        Western_Asia.append(cluster_labels[hclust_index])
    elif(region[hclust_index] == 'Southeast Asia'):
        Southeast_Asia.append(cluster_labels[hclust_index])
    elif(region[hclust_index] == 'Central America/Caribean'):
        Central_America_Caribean.append(cluster_labels[hclust_index])
    elif(region[hclust_index] == 'Asia'):
        Asia.append(cluster_labels[hclust_index])
    elif(region[hclust_index] == 'North America'):
        North_America.append(cluster_labels[hclust_index])
    elif(region[hclust_index] == 'Pacific Islands'):
        Pacific_Islands.append(cluster_labels[hclust_index])
    elif(region[hclust_index] == 'South Asia'):
        South_Asia.append(cluster_labels[hclust_index])
    hclust_index = hclust_index + 1
print("分群結果：")
print(Central_Asia)
print(South_Asia)
print(Asia)
print(Western_Asia)
print(Southeast_Asia)
print(Australia)
print(Pacific_Islands)
print(Africa)
print(Europe)
print(South_America)
print(North_America)
print(Central_America_Caribean)


# In[9]:


#initialize th list
Central_Asia = ['Central Asia// ']
Europe = ['Europe// ']
Africa = ['Africa// ']
South_America = ['South America// ']
Australia = ['Australia// ']
Western_Asia = ['Western Asia// ']
Southeast_Asia = ['Southeast Asia// ']
Central_America_Caribean = ['Central America/Caribean// ']
Asia = ['Asia// ']
North_America = ['North America// ']
Pacific_Islands = ['Pacific Islands// ']
South_Asia = ['South Asia// ']


# In[10]:


#決策樹
# 切分訓練與測試資料
train_X, test_X, train_y, test_y = train_test_split(dick_array, region, test_size = 0.3)

# 建立分類器
clf = tree.DecisionTreeClassifier()
dick_clf = clf.fit(train_X, train_y)

# 預測
test_y_predicted = dick_clf.predict(test_X)
#print(test_y_predicted)

# 績效
accuracy = metrics.accuracy_score(test_y, test_y_predicted)
print('決策樹績效：',accuracy)

# 標準答案
#print(test_y)

index = 0
for i in test_y:
    if(test_y_predicted[index] == 'Central Asia'):
        Central_Asia.append(test_y[index])
    elif(test_y_predicted[index] == 'Europe'):
        Europe.append(test_y[index])
    elif(test_y_predicted[index] == 'Africa'):
        Africa.append(test_y[index])
    elif(test_y_predicted[index] == 'South America'):
        South_America.append(test_y[index])
    elif(test_y_predicted[index] == 'Australia'):
        Australia.append(test_y[index])
    elif(test_y_predicted[index] == 'Western Asia'):
        Western_Asia.append(test_y[index])
    elif(test_y_predicted[index] == 'Southeast Asia'):
        Southeast_Asia.append(test_y[index])
    elif(test_y_predicted[index] == 'Central America/Caribean'):
        Central_America_Caribean.append(test_y[index])
    elif(test_y_predicted[index] == 'Asia'):
        Asia.append(test_y[index])
    elif(test_y_predicted[index] == 'North America'):
        North_America.append(test_y[index])
    elif(test_y_predicted[index] == 'Pacific Islands'):
        Pacific_Islands.append(test_y[index])
    elif(test_y_predicted[index] == 'South Asia'):
        South_Asia.append(test_y[index])
    index = index + 1
print("決策樹 結果：")
print(Central_Asia)
print(South_Asia)
print(Asia)
print(Western_Asia)
print(Southeast_Asia)
print(Australia)
print(Pacific_Islands)
print(Africa)
print(Europe)
print(South_America)
print(North_America)
print(Central_America_Caribean)


# In[11]:


#initialize th list
Central_Asia = ['Central Asia// ']
Europe = ['Europe// ']
Africa = ['Africa// ']
South_America = ['South America// ']
Australia = ['Australia// ']
Western_Asia = ['Western Asia// ']
Southeast_Asia = ['Southeast Asia// ']
Central_America_Caribean = ['Central America/Caribean// ']
Asia = ['Asia// ']
North_America = ['North America// ']
Pacific_Islands = ['Pacific Islands// ']
South_Asia = ['South Asia// ']


# In[12]:


#k-Nearest Neighbors 分類器
# 切分訓練與測試資料
train2_X, test2_X, train2_y, test2_y = train_test_split(dick_array, region, test_size = 0.3)

#選擇 k
#讓程式幫我們怎麼選擇一個適合的 k，通常 k 的上限為訓練樣本數的 20%。
range = np.arange(1, round(0.2 * train2_X.shape[0]) + 1)
accuracies = []#儲存準確度數值的list

for i in range:
    # 建立分類器
    clf2 = neighbors.KNeighborsClassifier(n_neighbors = i)
    dick_clf2 = clf2.fit(train2_X, train2_y)
    test2_y_predicted = dick_clf2.predict(test2_X)
    # 績效
    accuracy = metrics.accuracy_score(test2_y, test2_y_predicted)
    accuracies.append(accuracy)

#各種k的結果去做視覺化
plt.scatter(range, accuracies)
plt.show()
appr_k = accuracies.index(max(accuracies)) + 1
print(appr_k)

#用預設的k=5下去跑
clf2 = neighbors.KNeighborsClassifier()
dick_clf2 = clf2.fit(train2_X, train2_y)
test2_y_predicted = dick_clf2.predict(test2_X)
# 績效
accuracy = metrics.accuracy_score(test2_y, test2_y_predicted)
print('\nk-Nearest Neighbors(預設 k = 5)績效：',accuracy)

index = 0
for i in test2_y:
    if(test2_y_predicted[index] == 'Central Asia'):
        Central_Asia.append(test2_y[index])
    elif(test2_y_predicted[index] == 'Europe'):
        Europe.append(test2_y[index])
    elif(test2_y_predicted[index] == 'Africa'):
        Africa.append(test2_y[index])
    elif(test2_y_predicted[index] == 'South America'):
        South_America.append(test2_y[index])
    elif(test2_y_predicted[index] == 'Australia'):
        Australia.append(test2_y[index])
    elif(test2_y_predicted[index] == 'Western Asia'):
        Western_Asia.append(test2_y[index])
    elif(test2_y_predicted[index] == 'Southeast Asia'):
        Southeast_Asia.append(test2_y[index])
    elif(test2_y_predicted[index] == 'Central America/Caribean'):
        Central_America_Caribean.append(test2_y[index])
    elif(test2_y_predicted[index] == 'Asia'):
        Asia.append(test2_y[index])
    elif(test2_y_predicted[index] == 'North America'):
        North_America.append(test2_y[index])
    elif(test2_y_predicted[index] == 'Pacific Islands'):
        Pacific_Islands.append(test2_y[index])
    elif(test2_y_predicted[index] == 'South Asia'):
        South_Asia.append(test2_y[index])
    index = index + 1
print("k-Nearest Neighbors 結果：")
print(Central_Asia)
print(South_Asia)
print(Asia)
print(Western_Asia)
print(Southeast_Asia)
print(Australia)
print(Pacific_Islands)
print(Africa)
print(Europe)
print(South_America)
print(North_America)
print(Central_America_Caribean)


# In[13]:


#用決策樹的模型去跑k-Nearest Neighbors的test data
test_weird_predicted = dick_clf.predict(test2_X)
# 績效
accuracy_weird = metrics.accuracy_score(test2_y, test_weird_predicted )
print('用決策樹的模型去跑k-Nearest Neighbors的test data的績效：',accuracy_weird)

#反過來用k-Nearest Neighbors的模型去跑決策樹的test data
test_weird2_predicted = dick_clf2.predict(test_X)
# 績效
accuracy2_weird = metrics.accuracy_score(test_y, test_weird2_predicted )
print('用k-Nearest Neighbors的模型去跑決策樹的test data的績效：',accuracy2_weird)


# In[14]:


#讀取height.csv
height = pd.read_csv('height.csv')
#把dataframe轉成numpy.ndarray
height_array = height.values
tmp_height = 100.0
small_height_name = []
small_height_cm = []
index = 0
#整理height.csv 因為裡面有重複性的資料 把重複的挑掉
for i in height_array:
    tmp = height_array[index,0]
    tmp_height = height_array[index,3]
    if(tmp == 'Zimbabwe'): 
        small_height_name.append(tmp)
        small_height_cm.append(tmp_height)
        break
    if(tmp != height_array[index+1,0]):
        small_height_name.append(tmp)
        small_height_cm.append(tmp_height)
    index = index + 1
print('整理後的height.csv資料個數：',len(small_height_name))
count = 0
index = 0
#尋找dick和height共同國家名稱的資料
for i in small_height_name:
    for j in country:
        if(small_height_name[index] == j):
            count = count + 1
    index = index + 1
print('dick的資料個數：',len(country))
print('共同的國家資料個數：',count)


# In[15]:


#創一個新的dataset 結合dick跟height資料的（註：有些資料有缺失，所以把原本的dick資料刪掉139-106 ＝33個）
with open('dick_with_height.csv', 'wt') as f:
    #這是屬性列
    print('height(cm)//dick_length_soft//dick_length_hard//dick_v_soft//dick_v_hard//dick_circle_soft//dick_circle_hard',file=f)
    index = 0
    for i in country:
        #把height.csv裡面有辦法跟原本的dick data相同的國家的height資料標上去
        find = 0
        index_d_with_h =0
        for j in small_height_name:
            if(j == i):
                print(small_height_cm[index_d_with_h],end='//', file=f)
                find = 1
            index_d_with_h = index_d_with_h + 1
        #有相對應的國家的話才把dick的資料也填上去
        if(find == 1):
            print(dick_length_soft[index],end='//', file=f)#end='//'代表用//分隔資料不是用預設的，
            print(dick_length_hard[index],end='//', file=f)
            print(dick_v_soft[index],end='//', file=f)
            print(dick_v_hard[index],end='//', file=f)
            print(dick_circle_soft[index],end='//', file=f)
            print(dick_circle_hard[index],end='\n', file=f)
        index = index + 1


# In[16]:


print('請接續執行令一個檔案： creative_data.ipynb')

