  
# I wrote this code in Jupyter Notebook but there's some issue with github showing .ipynb files(thread already opened)
# So, I am changing the extension from .ipynb to .py so that if anyone wants, he/she can see the code easily
# For each new line, that code is actually written in new cell of jupyter notebook

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data',names=['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat'])

print(df.head())
print(df.info())

print(df.isnull().any())

print(df['class'].value_counts())

sns.countplot(df["class"])
plt.title("Class")
plt.show()

print(df.describe())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df = df.apply(le.fit_transform)

sns.countplot(df['habitat'])

sns.countplot(df["cap-surface"])

sns.countplot('stalk-root',data = df)

sns.distplot(df['odor'])

sns.countplot(df['odor'])

sns.distplot(df['class'])

plt.figure(figsize=(9,5))
sns.countplot(x="cap-surface",data = df ,hue = "class")
plt.show()

x = df.drop(["class"],axis=1)
y = df["class"].values

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2,random_state=41)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(train_x,train_y)
pred_1 = lg.predict(test_x)
print(accuracy_score(test_y,pred_1)) # 0.9347692307692308

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(train_x,train_y)
pred_2 = dt.predict(test_x)
print(accuracy_score(test_y,pred_2))

print(confusion_matrix(test_y,pred_2))

print(classification_report(test_y,pred_2))

from sklearn.metrics import roc_auc_score,roc_curve
print(roc_auc_score(test_y,dt.predict(test_x))) #1.0

y_pred=  dt.predict_proba(test_x)[:,1]

fpr,tpr,thres = roc_curve(test_y,y_pred)
plt.plot([0,1],[0,1],'r--')
plt.plot(fpr,tpr,marker='.')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC-Curve')
plt.show()

import pickle as pc

fd = open('mush.obj','wb')
pc.dump(dt,fd)
fd.close()

fo = open('mush.obj','rb')
res = pc.load(fo)
fo.close()

print(res.score(train_x,train_y))

print(res.score(test_x,test_y))
