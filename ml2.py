#kütüphanelerin çağrılması
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#verinin yüklenmesi
df=pd.read_csv("ISIC_2019_Training_Metadata.csv")
df2= pd.read_csv("ISIC_2019_Training_GroundTruth.csv")

# verinin ilk 5 satirini gosterir
print (df.head())
print (df2.head())

# veri icindeki ozelliklerin veri tipleri
print (df.info())
print (df2.info())

# verinin boyu
print (df.shape)
print (df2.shape)

# eksik verilerin analizi
kolon_eksik_deger_toplami = df.isnull().sum()
print(kolon_eksik_deger_toplami)

#istenmeyen kolon lesion_id'nin silinmesi
df.drop(columns="lesion_id",axis=1,inplace=True)
print (df.head())

# boş değer sayısının bulunması
print (df.isnull().sum())

# boş değerlerin silinmesi
df = df.dropna(axis=0)
print (df.isnull().sum())
print (df.shape)

df.age_approx=df.age_approx.astype(int)
df.anatom_site_general=df.anatom_site_general.astype(str)
df.gender=df.gender.astype(str)

#csv dosyalarını birleştirme
s=pd.DataFrame([t for t in np.where(df2==1, df2.columns,"").flatten()
                if len (t)>0], columns=(["sinif"]))
ydf=pd.concat([df,s], axis=1)
ydf.head(5)

# birleştirilmiş veri setindeki boş değerlerin silinmesi
ydf = ydf.dropna(axis=0)
print (ydf.isnull().sum())
print (ydf.shape)

#gerekmeyen kolon image'nin silinmesi
ydf.drop(columns="image",axis=1,inplace=True)
print (ydf.head())

#verideki lezyon türlerinin dağılımını gösteren grafik
ydf.groupby("sinif").size().plot(kind='bar', color='#0097A7',edgecolor='black')
plt.xticks(rotation=0)
plt.xlabel("Türler")
plt.ylabel("Örnek Sayılar")
plt.title("Lezyon Türleri")
plt.show()

#lezyonun bulunduğu bölgeyi gösteren grafik
ydf.groupby("anatom_site_general").size().plot(kind='bar',color='#4DB6AC',edgecolor='black')
plt.xlabel("Bölgeler")
plt.ylabel("Örnek Sayılar")
plt.title("Lezyonun Bulunduğu Bölgeler")
plt.show()

#cinsiyet dağılımını gösteren grafik
ydf.groupby("gender").size().plot(kind='bar',color='#80DEEA',edgecolor='black')
plt.xticks(rotation=0)
plt.xlabel("Cinsiyet")
plt.ylabel("Örnek Sayılar")
plt.title("Cinsiyet Dağılımı")
plt.show()

#yaş dağılımını gösteren grafik
ydf.groupby("age_approx").size().plot(kind='bar',color='#9575CD',edgecolor='black')
plt.xlabel("Yaş")
plt.ylabel("Örnek Sayılar")
plt.title("Yaş Dağılımı")
plt.grid(True)
plt.show()

#sınıfı MEL olanların cinsiyet dağılımı
ydf[ydf["sinif"]=="MEL"].groupby("gender").size().plot(kind='bar',color='#E57373',edgecolor='black')
plt.xticks(rotation=0)
plt.xlabel("Cinsiyet")
plt.ylabel("Örnek Sayılar")
plt.title("Sınıfı MEL Olanların Cinsiyet Dağılımı")
plt.show()

#sınıfı MEL olanların lezyon bölgesinin dağılımını gösteren grafik
ydf[ydf["sinif"]=="MEL"].groupby("anatom_site_general").size().plot(kind='bar',color='#880E4F',edgecolor='black')
plt.xlabel("Bölgeler")
plt.ylabel("Örnek Sayılar")
plt.title("Sınıfı MEL Olanların Lezyonunun Bulunduğu Bölgeler")
plt.show()


#kategorik değerleri nümerik hale çevirme

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
dtype_object=ydf.select_dtypes(include=['object'])
print (dtype_object.head())
for x in dtype_object.columns:
    ydf[x]=le.fit_transform(ydf[x])

print (ydf.head())

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ydf["sinif"]=le.fit_transform(ydf["sinif"])


#veri setini bağımsız ve bağımlı değişkenlere ayırma
X = ydf.iloc[:,:4].values
y = ydf["sinif"].values

#veriyi %80 eğitim %20 test olarak bölme
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# özellik ölçeklendirme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


score=[]
algorithms=[]


#KNN
from sklearn.neighbors import KNeighborsClassifier

#model and accuracy
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
knn.predict(X_test)
score.append(knn.score(X_test,y_test)*100)
algorithms.append("KNN")
print("KNN accuracy =",knn.score(X_test,y_test)*100)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred=knn.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)

#Confusion Matrix on Heatmap
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title(" KNN Confusion Matrix")
plt.show()

from sklearn.metrics import classification_report

target_names=["AK","BCC","BKL","DF","MEL","NV","SCC","VASC"]
print(classification_report(y_true, y_pred, target_names=target_names))

#Navie-Bayes
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
#Training
nb.fit(X_train,y_train)
#Test
score.append(nb.score(X_test,y_test)*100)
algorithms.append("Navie-Bayes")
print("Navie Bayes accuracy =",nb.score(X_test,y_test)*100)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred=nb.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)

#Confusion Matrix on Heatmap
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Navie Bayes Confusion Matrix")
plt.show()
target_names=["AK","BCC","BKL","DF","MEL","NV","SCC","VASC"]
print(classification_report(y_true, y_pred, target_names=target_names))

#Support Vector Machine
from sklearn.svm import SVC
svm=SVC(random_state=1)
svm.fit(X_train,y_train)
score.append(svm.score(X_test,y_test)*100)
algorithms.append("Support Vector Machine")
print("svm test accuracy =",svm.score(X_test,y_test)*100)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred=svm.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)

#Confusion Matrix on Heatmap
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Support Vector Machine Confusion Matrix")
plt.show()
target_names=["AK","BCC","BKL","DF","MEL","NV","SCC","VASC"]
print(classification_report(y_true, y_pred, target_names=target_names))

# DecisionTree
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
print("Decision Tree accuracy:",dt.score(X_test,y_test)*100)
score.append(dt.score(X_test,y_test)*100)
algorithms.append("Decision Tree")

#Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred=dt.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)

#Confusion Matrix on Heatmap
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Decision Tree Confusion Matrix")
plt.show()
target_names=["AK","BCC","BKL","DF","MEL","NV","SCC","VASC"]
print(classification_report(y_true, y_pred, target_names=target_names))

# LogisticRegression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs')
lr.fit(X_train,y_train)
score.append(lr.score(X_test,y_test)*100)
algorithms.append("Logistic Regression")
print("Logistic Regression accuracy {}".format(lr.score(X_test,y_test)))
#Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred=lr.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)
#Confusion Matrix on Heatmap
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Logistic Regression Confusion Matrix")
plt.show()
target_names=["AK","BCC","BKL","DF","MEL","NV","SCC","VASC"]
print(classification_report(y_true, y_pred, target_names=target_names))

# Artificial Neural Networks

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#Spliting the dataset in independent and dependent variables
X = ydf.iloc[:,:24].values
print (X.shape[0])
y = ydf['sinif'].values.reshape(X.shape[0], 1)

#split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=82)

#standardize the dataset
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
sc.fit(X_test)
X_test = sc.transform(X_test)

sknet = MLPClassifier(hidden_layer_sizes=(8), learning_rate_init=0.01, max_iter=100)
sknet.fit(X_train, y_train)

score.append(sknet.score(X_test,y_test)*100)
algorithms.append("Artificial Neural Networks")

y_pred = sknet.predict(X_test)
y_true=y_pred

cm=confusion_matrix(y_true,y_pred)
#Confusion Matrix on Heatmap
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Artificial Neural Networks Confusion Matrix")
plt.show()
target_names=["AK","BCC","BKL","DF","MEL","NV","SCC","VASC"]
print(classification_report(y_true, y_pred, target_names=target_names))

print (algorithms)
print (score)

x_pos = [i for i, _ in enumerate(algorithms)]

plt.bar(x_pos, score, color='#26A69A',edgecolor='black')
plt.xlabel("Algoritmalar")
plt.ylabel("Basari Yuzdeleri")
plt.title("Basari Siralamalar")

plt.xticks(x_pos, algorithms,rotation=90)

plt.show()

