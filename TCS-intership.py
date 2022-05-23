#!/usr/bin/env python
# coding: utf-8

# # Build a model that classifies the side effects of a drug 

# **Dataset Description**

# * Name (categorical)       : Name of the patient
# * Age (numerical)          : Age group range of user
# * Race (categorical)       : Race of the patients 
# * Condition (categorical)  : Name of condition
# * Date (date)              : date of review entry
# * Drug (categorical)       : Name of drug
# * EaseOfUse (numerical)    : 5 star rating
# * Effectiveness (numerical): 5 star rating
# * Sex (categorical)        : gender of user
# * Side (text)              : side effects associated with drug (if any)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df1=pd.read_csv('Side_effect.csv')
df1.head()


# In[3]:


df1.columns


# In[4]:


df1.info()


# In[5]:


df1.describe()


# In[6]:


df1.describe(include='object')


# In[7]:


df1.shape


# # Pre-Processing

# # missing values

# In[8]:


df1.isnull().sum()


# In[9]:


df1['Race'] = df1['Race'].fillna(df1['Race'].mode()[0])
df1['Sex'] = df1['Sex'].fillna(df1['Sex'].mode()[0])
df1['Age'] = df1['Age'].fillna(df1['Age'].mode()[0])
df1['Condition'] = df1['Condition'].fillna(df1['Condition'].mode()[0])
df1['Drug'] = df1['Drug'].fillna(df1['Drug'].mode()[0])


# In[10]:


df1.isnull().sum()


# # Outliers

# In[11]:


# Plot boxplot to find outliers
#boxplot
df_numerical = df1.select_dtypes(exclude='object')
x=1
plt.figure(figsize = (20, 15))
for col in df_numerical.columns:
    plt.subplot(6,4,x)
    sns.boxplot(df1[col])
    x+=1
plt.tight_layout()


# **There are no outliers in the dataset**

# # Encoding

# In[12]:


df1.describe(include='object')


# **Label Encoding**

# In[13]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[14]:


df1.head()


# In[15]:


data =df1.copy()


# In[16]:


data['Condition']=le.fit_transform(data['Condition'])


# In[17]:


data['Drug']=le.fit_transform(data['Drug'])


# In[18]:


data['Race']=le.fit_transform(data['Race'])
data['Sex']=le.fit_transform(data['Sex'])


# In[19]:


data.head()


# In[20]:


data['Age']=data.Age.map({
    '0-10': 0,
    '11-20':1,
    '21-30':2,
    '31-40':3,
    '41-50':4,
    '51-60':5,
    '61-70':6,
    '71-80':7,
    '81-90':8,
    '91-100':9,
    })


# In[21]:


data.shape


# In[22]:


data.head()


# # Feature Reduction

# We can drop Name and Date columns as they dont influence the side effects

# In[23]:


data=data.drop(['Name','Date'],axis=1)


# In[24]:


#Lets see the correlation
plt.figure(figsize=(10,7))
sns.heatmap(data.corr(),annot=True,cmap='RdYlGn')


# Since  Effectivenes and EaseofUse are highly correlated to each other we only need to take one of them for modelling.

# In[25]:


data=data.drop(['EaseofUse'],axis=1)


# In[26]:


data.shape


# In[ ]:





# In[27]:


y=data['Sides']
x=data.drop(['Sides'],axis=1)


# # **Scaling**

# In[28]:


cols= x.columns


# In[29]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
scale=['Effectiveness']
x[scale] = ss.fit_transform(x[scale])


# In[30]:


x.head()


# In[31]:


x.describe()


# In[ ]:





# # Exploratory Data Analysis (EDA)

# **Gender and Gender vs Side Effects**

# In[32]:


plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
df1['Sex'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True)
plt.subplot(1, 2, 2)
sns.countplot(data = df1, x = 'Sex',hue='Sides' ,palette='coolwarm')
plt.title('Gender vs Side Effects', size=20)
plt.show()
df1.Sex.value_counts(normalize=True)


# In[ ]:





# **Race and Race vs Side Effects**

# In[33]:


plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
sns.countplot(data=df1,x='Race',order=df1['Race'].value_counts().index)
plt.xlabel('Race', fontsize=15)
plt.subplot(1, 2, 2)
sns.countplot(data = df1, x = 'Race',hue='Sides' ,palette='coolwarm')
plt.title('Race vs Side Effects', size=20)
plt.show()
df1.Race.value_counts(normalize=True)


# **Side Effect**

# In[34]:


plt.figure(figsize=(10,8))
plt.title('% of Side effect')
tr = pd.DataFrame(df1['Sides'].value_counts())
tr_names = tr.index
count = tr['Sides']
plt.style.use('ggplot')
plt.rc('font', size=12)
plt.pie(count, autopct='%1.1f%%', labels = tr_names, pctdistance=0.9, labeldistance=1.1,shadow=True, startangle=90)
plt.show()
df1.Sides.value_counts(normalize=True)


# **Conditions**

# In[35]:


plt.figure(figsize=(15,9))
plt.title('% of different Conditions of People')
tr = pd.DataFrame(df1['Condition'].value_counts())
tr_names = tr.index
count = tr['Condition']
plt.style.use('ggplot')
plt.rc('font', size=12)
plt.pie(count, autopct='%1.1f%%', labels = tr_names, pctdistance=0.9, labeldistance=1.1)
plt.show()


# In[36]:


df1.Condition.value_counts()


# In[37]:


df1[df1.Drug=='celexa']


# **Drug**

# In[38]:


plt.figure(figsize=(15,9))
plt.title('% of different Drugs')
tr = pd.DataFrame(df1['Drug'].value_counts())
tr_names = tr.index
count = tr['Drug']
plt.style.use('ggplot')
plt.rc('font', size=12)
plt.pie(count, autopct='%1.1f%%', labels = tr_names, pctdistance=0.9, labeldistance=1.1)
plt.show()


# **Age and Age vs Side Effects**

# In[39]:


plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
sns.countplot(data=df1,x='Age',order=df1['Age'].value_counts().index)
plt.xlabel('Race', fontsize=15)
plt.subplot(1, 2, 2)
sns.countplot(data = df1, x = 'Age',hue='Sides' ,palette='coolwarm')
plt.title('Age vs Side Effects', size=20)
plt.show()
df1.Sides.value_counts(normalize=True)


# **Effectiveness and Effectiveness vs Side Effects**

# In[40]:


plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
sns.countplot(data=df1,x='Effectiveness',order=df1['Effectiveness'].value_counts().index)
plt.xlabel('Effectiveness', fontsize=15)
plt.subplot(1, 2, 2)
sns.countplot(data = df1, x = 'Effectiveness',hue='Sides' )
plt.title('Effectiveness vs Side Effects', size=20)
plt.show()
df1.Effectiveness.value_counts(normalize=True)


# **Effectiveness vs Sex** 

# In[41]:


plt.figure(figsize=(8,5))
sns.countplot(data = df1, x = 'Effectiveness',hue='Sex' )
plt.title('Effectiveness vs Sex', size=20)
plt.show()
df1.Effectiveness.value_counts(normalize=True)


# **Ease of Use and Ease of Use vs Side Effects**

# In[42]:


plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
sns.countplot(data=df1,x='EaseofUse',order=df1['EaseofUse'].value_counts().index)
plt.xlabel('Ease of Use', fontsize=15)
plt.subplot(1, 2, 2)
sns.countplot(data = df1, x = 'EaseofUse',hue='Sides' )
plt.title('EaseofUse vs Side Effects', size=20)
plt.show()
df1.EaseofUse.value_counts(normalize=True)


# **Effectiveness vs Age**

# In[43]:


plt.figure(figsize=(12, 7.5))
sns.countplot(x='Effectiveness',hue='Age', data=df1,)
plt.title("Effectiveness vs Age",fontweight="bold", size=20)


# **Ease of Use vs Age**

# In[44]:


plt.figure(figsize=(9, 6))
sns.countplot(x='EaseofUse',hue='Age', data=df1)
plt.title("Ease of Use vs Age",fontweight="bold", size=20)
plt.show()


# **Ease of Use vs Sex**

# In[45]:


plt.figure(figsize=(9,5))
sns.countplot(x='EaseofUse',hue='Sex', data=df1)
plt.title("Ease of Use vs Sex",fontweight="bold", size=20)
plt.show()


# **Correlation Heatmap**

# In[46]:


#Lets see the correlation
plt.figure(figsize=(10,7))
sns.heatmap(data.corr(),annot=True,cmap='RdYlGn')


# In[ ]:





# # Splitting Dataset into train and test datasets

# In[47]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,classification_report


# In[ ]:





# # Modelling

# **a. LogisticRegression**

# In[48]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,classification_report
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)


# In[49]:


print('Accuracy on Logistic Regression is : ',accuracy_score(y_test,y_pred))
print('precision is : ',precision_score(y_test,y_pred,average='macro'))
print('recall is : ',recall_score(y_test,y_pred,average='macro'))
print('f1 score is : ',f1_score(y_test,y_pred,average='macro'))


# In[50]:


confusion_matrix(y_test,y_pred)


# In[ ]:





# **b. KNN**

# In[51]:


from sklearn.neighbors import KNeighborsClassifier
#find optimum k- value.We have to create model with varied k values
acc_values=[]
neighbors=np.arange(3,18) #taking values 3 to 15 into a variable

#loop to ceate KNN model for each k-value
for k in neighbors:
    knn=KNeighborsClassifier(n_neighbors=k,metric='minkowski')#instance of KNN to variable
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
    #append accuracy values to acc_values to find out the maximum accuracy
    acc=accuracy_score(y_test,y_pred)
    acc_values.append(acc)


# In[52]:


acc_values


# In[53]:


#find correspomd k value corresponding to highest accuracy
plt.plot(neighbors,acc_values,'o-')
plt.xlabel('k-value')
plt.ylabel('Accuracy')


# In[54]:


#Make a model with k as 16
knn=KNeighborsClassifier(n_neighbors=16,metric='minkowski')#instance of KNN to variable
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)


# In[55]:


print('Accuracy on KNN is : ',accuracy_score(y_test,y_pred))
print('precision is : ',precision_score(y_test,y_pred,average='macro'))
print('recall is : ',recall_score(y_test,y_pred,average='macro'))
print('f1 score is : ',f1_score(y_test,y_pred,average='macro'))


# In[56]:


confusion_matrix(y_test,y_pred)


# In[ ]:





# **c. Random Forest Classifier**

# In[57]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)


# In[58]:


print('Accuracy on Random Forest is : ',accuracy_score(y_test,y_pred))
print('f1 score is : ',f1_score(y_test,y_pred,average='macro'))
print('precision is : ',precision_score(y_test,y_pred,average='macro'))
print('recall is : ',recall_score(y_test,y_pred,average='macro'))


# In[59]:


confusion_matrix(y_test,y_pred)


# In[ ]:





# **d. Fine tuning random forest classifier**

# In[60]:


rft=RandomForestClassifier(bootstrap=True,n_estimators=500,oob_score=True,max_depth=20,criterion='entropy',random_state=92)
rft.fit(x_train,y_train)
y_pred=rft.predict(x_test)


# In[61]:


print('Accuracy on Fine tuned Random Forest is : ',accuracy_score(y_test,y_pred))
print('f1 score is : ',f1_score(y_test,y_pred,average='macro'))
print('precision is : ',precision_score(y_test,y_pred,average='macro'))
print('recall is : ',recall_score(y_test,y_pred,average='macro'))


# In[62]:


confusion_matrix(y_test,y_pred)


# In[ ]:





# **e. Gradient Boosting**

# In[66]:


from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier()
gb.fit(x_train,y_train)
y_pred=gb.predict(x_test)


# In[67]:


print('Accuracy on Gradient Boosting is : ',accuracy_score(y_test,y_pred))
print('precision is : ',precision_score(y_test,y_pred,average='macro'))
print('recall is : ',recall_score(y_test,y_pred,average='macro'))
print('f1 score is : ',f1_score(y_test,y_pred,average='macro'))


# In[68]:


confusion_matrix(y_test,y_pred)


# In[ ]:





# **f. Gaussian Naive Bayes**

# In[63]:


from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)


# In[64]:


print('Accuracy on naive_bayes is : ',accuracy_score(y_test,y_pred))
print('precision is : ',precision_score(y_test,y_pred,average='macro'))
print('recall is : ',recall_score(y_test,y_pred,average='macro'))
print('f1 score is : ',f1_score(y_test,y_pred,average='macro'))


# In[65]:


confusion_matrix(y_test,y_pred)

