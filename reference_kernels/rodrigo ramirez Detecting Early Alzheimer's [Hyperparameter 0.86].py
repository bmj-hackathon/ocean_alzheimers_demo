# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] {"_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5", "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19"}
# Alzheimer's is a type of dementia that affects a person's Memory, Thought and Behavior. It is a disease that begins mildly and affects parts of the brain, which makes the person have difficulty, to remember newly learned information, constant changes in mood, and confusion with events, times and places.
#  
# Alzheimer's usually starts after age 60. The risk increases as the person ages. The risk of having this disease is greater if there are people in the family who have had this disease.
#  
# As for the treatments that have been done for this disease, there is none that can stop the progress of this. So far what these treatments can achieve is to help alleviate some symptoms, reducing their intensity and contributing to a higher quality of life for patients and their families.
#
# <img src="https://gx0ri2vwi9eyht1e3iyzyc17-wpengine.netdna-ssl.com/wp-content/uploads/2017/01/dementia2-804x369.jpg" alt="AzBoruta" border="0">
#
# ## objective
#
# Implement classification algorithms for the analysis of the medical dataset, in order to provide a prediction tool for the early diagnosis of the disease.
#

# %% [markdown] {"_cell_guid": "79c7e3d0-c299-4dcb-8224-4455121ee9b0", "_uuid": "d629ff2d2480ee46fbb7e2d37f6b5fab8052498a"}
# # Table of Contents
#
# * **1. [ Declaration of functions](#ch1)**
# * ** 2 [ Analysis of data](#ch2)**
#      * 2.1 [Read dataset](#ch3) 
#      * 2.2 [Correlation Analysis](#ch4) 
#      * 2.3 [Correlation matrix](#ch5) 
#      * 2.4 [Dispersion matrix](#ch6) 
#      * 2.5 [Graphs of all these correlations](#ch7) 
#      * 2.6 [Miscellaneous Graphics](#ch8) 
# * ** 3 [Preprocessing](#ch9)**
#      * 3.1 [Remove Useless Columns](#ch10)
#      * 3.2 [LabelEncoder](#ch11)
#      * 3.3 [Imputation of lost values](#ch12)
#      * 3.4 [Standardization](#ch13)
#      * 3.5 [Export them to then select the features](#ch14)
# * **  4 [Modeling](#ch15)** 
#      * 4.1 [Tuning Hyperparameters for better models](#ch15)
#      * 4.2 [Generating our models](#ch16)
#      * 4.3 [Cross Validation](#ch17)
# * **  5. [Importance of characteristics](#ch18)**
# * **  6. [Predictions](#ch19)**
# * ** 7. [Performance Metric for each model](#ch21)**
#     * 7.1 [Report ](#ch22)
#     * 7.2 [Results ](#ch23)

# %% [markdown] {"_uuid": "7f52c8211de403a552ab8430e456d2756f85e18d"}
# ## Required libraries

# %% {"_uuid": "2ebc01d30748a7f8b4c0a8168ca3a735759c364b"}
import pandas as pd
from scipy.io import arff
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import mean_squared_error

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


FOLDS =10
# %matplotlib inline

# %% [markdown] {"_uuid": "5659b18a7a0ad9e80c347f9de923dff1ade4c679"}
# <a id="ch1"></a>
# # 1. Declaration of functions
#
# ## Graphing functions 

# %% {"_uuid": "20de97f163e3f954fb3bc087cc786d2837a8cb38"}
# Function to graph number of people by age
def cont_age(field):
    plt.figure()
    g = None
    if field == "Age":
        df_query_mri = df[df["Age"] > 0]
        g = sns.countplot(df_query_mri["Age"])
        g.figure.set_size_inches(18.5, 10.5)
    else:
        g = sns.countplot(df[field])
        g.figure.set_size_inches(18.5, 10.5)
    
sns.despine()


# %% {"_uuid": "5100e7ace02856617767413ac8d8095553e513ae"}
# Function to graph number of people per state [Demented, Nondemented]
def cont_Dementes(field):
    plt.figure()
    g = None
    if field == "Group":
        df_query_mri = df[df["Group"] >= 0]
        g = sns.countplot(df_query_mri["Group"])
        g.figure.set_size_inches(18.5, 10.5)
    else:
        g = sns.countplot(df[field])
        g.figure.set_size_inches(18.5, 10.5)
    
sns.despine()


# %% {"_uuid": "49442c36a93eb7ae4309ea60e2b1fb9e0ffb70ba"}
# 0 = F y 1= M
def bar_chart(feature):
    Demented = df[df['Group']==1][feature].value_counts()
    Nondemented = df[df['Group']==0][feature].value_counts()
    df_bar = pd.DataFrame([Demented,Nondemented])
    df_bar.index = ['Demented','Nondemented']
    df_bar.plot(kind='bar',stacked=True, figsize=(8,5))


# %% {"_uuid": "39f02e184f4d42a388c2b795dab056f287c3fba8"}
def report_performance(model):

    model_test = model.predict(X_test)

    print("Confusion Matrix")
    print("{0}".format(metrics.confusion_matrix(y_test, model_test)))
    print("")
    print("Classification Report")
    print(metrics.classification_report(y_test, model_test))


# %% [markdown] {"_uuid": "501dd87529e4b189d939a2825e3f5ad1b7070d8e"}
# <a id="ch2"></a>
#  # 2. Analysis of data

# %% [markdown] {"_uuid": "06248da1c4b7609d1f85f2426011cfcfb4d72875"}
# <a id="ch3"></a>
# ## 2.1 read dataset

# %% {"_uuid": "b63a8e1f9288872b53e6ad6f990ac7c0fc97123b"}
data = '../input/oasis_longitudinal.csv'
df = pd.read_csv (data)
df.head()

# %% {"_uuid": "713b18fd969edf0e9aa0a6e68c61e531bb7355e3"}
df.describe()

# %% {"_uuid": "eebaebea551cf2c7a91b3e2b9d80456808941946"}
nu = pd.DataFrame(df['Group']=='Nondemented')
nu["Group"].value_counts() 

# %% [markdown] {"_uuid": "51d8c6ffa4e41d8256564b4a9d604923a3464131"}
# <a id="ch4"></a>
# ## 2.2 Correlation Analysis

# %% {"_uuid": "8bd677e96bfeeccabfd70c77616b89d2a8c347ce"}
f, ax = plt.subplots(figsize=(10, 8)) 
corr = df.corr(method = 'pearson') 
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), 
            square=True, ax=ax) 

# %% [markdown] {"_uuid": "36202cfe36bfbcd75eb4b43ec54de4c4158aa919"}
# <a id="ch5"></a>
# ## 2.3 Correlation matrix

# %% {"_uuid": "e6f313373dc63730c9de5bfab4255d1d7346c408"}
df.corr(method = 'pearson') 

# %% [markdown] {"_uuid": "a1df4cd154639e883683e0daaf2c0ea4544f999d"}
# <a id="ch6"></a>
# ## 2.4 Dispersion matrix

# %% {"_uuid": "9c50f5cacd13e99384235067cd5c5bd798c6e350"}
pd.scatter_matrix(df, alpha = 0.3, figsize = (14,8), diagonal = 'kde'); 

# %% [markdown] {"_uuid": "2e2c7926dcf0dc3e964740169cb8e0012ad1fb53"}
# <a id="ch7"></a>
# ## 2.5 Graphs of all these correlations

# %% {"_uuid": "e4ab9603689cbb6242b4060d3822d07700ad08ab"}
g = sns.PairGrid(df, vars=['Visit','MR Delay','M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF'],
                 hue='Group', palette='RdBu_r')
g.map(plt.scatter, alpha=0.8)
g.add_legend();

# %% [markdown] {"_uuid": "99e538c90d79479353e84552ed0446d16da67c44"}
# <a id="ch8"></a>
# ## 2.6 Miscellaneous Graphics

# %% [markdown] {"_uuid": "bc6b4c6c0def7c297f9ab698433cbb748e701b02"}
# **Number of Demented, Nondemented and Converted depending on the sex of the patient**

# %% {"_uuid": "e6da7bb59c2064af09690ea63fa723afc8dff94b"}
import seaborn as sb
sb.factorplot('M/F',data=df,hue='Group',kind="count")

# %% [markdown] {"_uuid": "d839d5ebaba69986bded9a37121c9718e157b2ca"}
# **Variation of the dementia according to the MMSE depending on the scores of each patient**

# %% {"_uuid": "086762422bd8252c1d97e176f8c7341559ad5334"}
facet= sns.FacetGrid(df,hue="Group", aspect=3)
facet.map(sns.kdeplot,'MMSE',shade= True)
facet.set(xlim=(0, df['MMSE'].max()))
facet.add_legend()
plt.xlim(12.5)

# %% [markdown] {"_uuid": "42c2bebc9c02bb0c6437aab22832e9332a345d74"}
# **Number of patients of each age**

# %% {"_uuid": "acdba17ed5d87a4f4fbdf7751a8840ce66ca4dcd"}
cont_age("Age")

# %% [markdown] {"_uuid": "20d1085113ae9c4ef81b347c846ae8c90ff00f2c"}
#  <a id="ch9"></a>
# # 3. Preprocessing

# %% [markdown] {"_uuid": "b4b194d8089d696818d79a9303131302515252ce"}
# **Replace data Convert a Dement**

# %% {"_uuid": "d096dae5de2f7a4544abfa6c496811d7fdc37002"}
df['Group'] = df['Group'].replace(['Converted'], ['Demented'])
df.head(3)

# %% [markdown] {"_uuid": "bf53cc8ccd91ffd707d82afc4d227fdca08eaa6f"}
#  <a id="ch10"></a>
# ## 3.1 Remove Useless Columns

# %% {"_uuid": "7ca9466c9db0f65a4e1ec46cadad7080bf607657"}
df.drop(['Subject ID'], axis = 1, inplace = True, errors = 'ignore')
df.drop(['MRI ID'], axis = 1, inplace = True, errors = 'ignore')
df.drop(['Visit'], axis = 1, inplace = True, errors = 'ignore')
#for this study the CDR we eliminated it
df.drop(['CDR'], axis = 1, inplace = True, errors = 'ignore')
df.head(3)

# %% [markdown] {"_uuid": "7b3abad7c93bc1bd31d4a44675cd6f8517deaa02"}
#  <a id="ch11"></a>
# ## 3.2 LabelEncoder

# %% [markdown] {"_uuid": "06c9a31cdc0a23a83246d05586c0ff4a71e11690"}
# ****We are going to use Binarized LabelEncoder for our Binary attributes********

# %% [markdown] {"_uuid": "0d4205aed5eee697f17488cb368fc547a5dd221d"}
# **Which are sex and our class**

# %% {"_uuid": "d2ae2426ce922e18f52278d7b917eaad2b651a79"}
# 1 = Demented, 0 = Nondemented
df['Group'] = df['Group'].replace(['Demented', 'Nondemented'], [1,0])    
df.head(3)

# %% {"_uuid": "a663afa7389ce1014c44319d6c350d17b8c332de"}
# 1= M, 0 = F
df['M/F'] = df['M/F'].replace(['M', 'F'], [1,0])  
df.head(3)

# %% {"_uuid": "dda23ef8586251af5d65ebc18e9d32a1d8f5118e"}
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
encoder.fit(df.Hand.values)
list(encoder.classes_)
#Transoformamos
encoder.transform(df.Hand.values)
df[['Hand']]=encoder.transform(df.Hand.values)
encoder2=LabelEncoder()
encoder2.fit(df.Hand.values)
list(encoder2.classes_)

# %% [markdown] {"_uuid": "e0f9572d888b1e2e9795f6969639d6a1dbb5a5b1"}
#  <a id="ch12"></a>
# ## 3.3 Imputation of lost values
#
# For various reasons, many real-world data sets contain missing values, often encoded as blanks, NaNs, or other placeholders. However, these data sets are incompatible with scikit-learn estimators that assume that all values ​​in a matrix are numeric, and that they all have and have meaning. A basic strategy for using incomplete datasets is to discard rows and / or complete columns that contain missing values. However, this has the price of losing data that can be valuable (though incomplete). A better strategy is to impute the lost values, that is, to deduce them from the known part of the data.
#
# The Imputer class provides basic strategies for imputation of missing values, using either the mean, the median or the most frequent value of the row or column in which the missing values ​​are found. This class also allows different encodings of missing values.

# %% [markdown] {"_uuid": "f76aa69afbd190adbe83cf82c8343e9426e853bb"}
# **Lost data**

# %% {"_uuid": "7851fe23a569497ba9d5eaf435ef06d2b1825087"}
data_na = (df.isnull().sum() / len(df)) * 100
data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Lost proportion (%)' :round(data_na,2)})
missing_data.head(20)

# %% {"_uuid": "cc37e9a033f4171ebb9161b15f53d646b436503f"}
from sklearn.impute  import SimpleImputer
# We perform it with the most frequent value 
imputer = SimpleImputer ( missing_values = np.nan,strategy='most_frequent')

imputer.fit(df[['SES']])
df[['SES']] = imputer.fit_transform(df[['SES']])

# %% {"_uuid": "70eac6dd65e1781d1ec200f6cfea90b5d0b26c42"}
from sklearn.impute  import SimpleImputer
# We perform it with the median
imputer = SimpleImputer ( missing_values = np.nan,strategy='median')

imputer.fit(df[['MMSE']])
df[['MMSE']] = imputer.fit_transform(df[['MMSE']])

# %% [markdown] {"_uuid": "fefa5f5ca1e8ce1a3e638dc6c4e9af54ba57180d"}
#  <a id="ch13"></a>
# # 3.4 Standardization

# %% {"_uuid": "fd4d3030728fb0bef7240b4457f334bcd1c2efec"}
from sklearn.preprocessing import StandardScaler
df_norm = df
scaler = StandardScaler()
df_norm[['Age','MR Delay','M/F','Hand','EDUC','SES','MMSE','eTIV','nWBV','ASF']]=scaler.fit_transform(df[['Age','MR Delay','M/F','Hand','EDUC','SES','MMSE','eTIV','nWBV','ASF']])

# %% {"_uuid": "e8706cfae6c8bd38e3caf866b35050bf73ac9872"}
df_norm.head(3)

# %% [markdown] {"_uuid": "b6d4b9d7cce4bc9e72a947de55ebfab8f4e7c22c"}
#  <a id="ch14"></a>
# ## 3.5 Export them to then select the features
#
# df_norm.to_csv('DatasetSelectionAttributes.csv', sep=',',index=False)
#
# For the selection of attributes we use the R Boruta framework.
#
# **Commands (R) :**
#
# library(readr)
#
# library(Boruta)
#
# covertype <- read_csv('DatasetSelectionAttributes.csv')
#
# set.seed(111)
#
# boruta.trainer <- Boruta(Group~., data = covertype , doTrace = 2, maxRuns=500)
#
# print(boruta.trainer)
#
# plot(boruta.trainer, las = 2)
#

# %% [markdown] {"_uuid": "e9a493b343eac4005ddcf09272fa6125d7f84685"}
# ## Result:

# %% [markdown] {"_uuid": "4917561e7cf759873ae1a0f7edb760d28f93a598"}
# <a href="https://ibb.co/QMGP76c"><img src="https://i.ibb.co/cQd6KNv/AzBoruta.png" alt="AzBoruta" border="0"></a>

# %% [markdown] {"_uuid": "9c3849bfe3c7403a0e8574972a1dff68285ae730"}
# ## Remove Columns selected by boruta

# %% {"_uuid": "e2a89a37bb93f41aa22a7f7cebb24957d6034d6c"}
df_norm.drop(['Hand'], axis = 1, inplace = True, errors = 'ignore')
df_norm.drop(['MR Delay'], axis = 1, inplace = True, errors = 'ignore')

# %% {"_uuid": "81599beb793f5942aa9ba7b027ab8acf31841447"}
df_norm.head()

# %% [markdown] {"_uuid": "53897f67b6c2b870ce9f6d5f3cbccbbad8628225"}
#  <a id="ch15"></a>
# # 4 Modeling

# %% {"_uuid": "ea78c8be0913ddf6a938d15daf60ce722f147df6"}
X = df_norm.drop(["Group"],axis=1)
y = df_norm["Group"].values
X.head(3)

# %% {"_uuid": "faa2207b3aa24496c9397bbc987fd396fed2da2c"}
# We divide our data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0)

# %% {"_uuid": "c77781d27079c1793e1a5c97884927b1a611ba2f"}
print("{0:0.2f}% Train".format((len(X_train)/len(df_norm.index)) * 100))
print("{0:0.2f}% Test".format((len(X_test)/len(df_norm.index)) * 100))

# %% {"_uuid": "abac655a240c72975a42103f9f984c72034c0ee4"}
print("Original Demented : {0} ({1:0.2f}%)".format(len(df_norm.loc[df_norm['Group'] == 1]), 100 * (len(df_norm.loc[df_norm['Group'] == 1]) / len(df_norm))))
print("Original Nondemented : {0} ({1:0.2f}%)".format(len(df_norm.loc[df_norm['Group'] == 0]), 100 * (len(df_norm.loc[df_norm['Group'] == 0]) / len(df_norm))))
print("")
print("Training Demented : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), 100 * (len(y_train[y_train[:] == 1]) / len(y_train))))
print("Training Nondemented : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), 100 * (len(y_train[y_train[:] == 0]) / len(y_train))))
print("")
print("Test Demented : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), 100 * (len(y_test[y_test[:] == 1]) / len(y_test))))
print("Test Nondemented : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), 100 * (len(y_test[y_test[:] == 0]) / len(y_test))))

# %% [markdown] {"_uuid": "100e2f83cb807edc91a716dab7540556cffc98bf"}
#  <a id="ch16"></a>
# ## 4.1 Tuning Hyperparameters for better models
#
# Before adjusting our models, we will look for the parameters that give us a high AUC

# %% [markdown] {"_uuid": "d17625b4a685fb05082670552fe7c4ab24332e3b"}
# **1°  Random Forest**

# %% {"_uuid": "7237693f8eee1dbd9e45cbe8b0b8ff24de4c6c2a"}
# Number of trees in random forest
n_estimators = range(10,250)
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = range(1,40)
# Minimum number of samples required to split a node
min_samples_split = range(3,60)

# %% {"_uuid": "a0d09d0871af097a460fa80236324ac3bfb3d3f0"}
# Create the random grid
parametro_rf = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split}

# %% {"_uuid": "2cf1f076d7a6f0cee325d08c83b727bf5135a9e9"}
model_forest = RandomForestClassifier(n_jobs=-1)
forest_random = RandomizedSearchCV(estimator = model_forest, param_distributions = parametro_rf, n_iter = 100, cv = FOLDS, 
                               verbose=2, random_state=42, n_jobs = -1, scoring='roc_auc')
forest_random.fit(X_train, y_train)

# %% {"_uuid": "0224f75d1835ac7c4d5271cf986b4c559775ed81"}
forest_random.best_params_

# %% [markdown] {"_uuid": "5f57fe144afedb5877fd18f4777065061aa8c1d0"}
# **** 2° Extra Tree****

# %% {"_uuid": "b5e5a62ae2beed519eb20e113ff1682df4c241e9"}
# Number of trees in random forest
n_estimators = range(50,280)
# Maximum number of levels in tree
max_depth =  range(1,40)
# Minimum number of samples required to split a node
min_samples_leaf = [3,4,5,6,7,8,9,10,15,20,30,40,50,60]

# %% {"_uuid": "c0b29eac750cd2b8a8e53731198e8f89f4638385"}
# Create the random grid
parametro_Et = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_leaf': min_samples_leaf}

# %% {"_uuid": "5f47ecc598e4ee66684b4eaf1a9513f1419b6b63"}
model_et = ExtraTreesClassifier(n_jobs=-1)
et_random = RandomizedSearchCV(estimator = model_et, param_distributions = parametro_rf, n_iter = 100, cv = FOLDS, 
                               verbose=2, random_state=42, n_jobs = -1, scoring='roc_auc')
et_random.fit(X_train, y_train)

# %% {"_uuid": "6c2a5d8a82768aec4bb41ab7ffbbda134b29eef4"}
et_random.best_params_

# %% [markdown] {"_uuid": "7de49315288468d1e1928ca03bcb3a7bf44414aa"}
# **3° AdaBoos**

# %% {"_uuid": "55d11a966ee8ca563a01ab92ede9fe82272af278"}
n_estimators = range(10,200)

learning_rate = [0.0001, 0.001, 0.01, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1]

# %% {"_uuid": "4eda16148741ee383b7d504f7e46fd74a28d74ad"}
# Create the random grid
parametros_ada = {'n_estimators': n_estimators,
               'learning_rate': learning_rate}

# %% {"_uuid": "751326f24bd7b21171dc0292fe3dba671b1c15d7"}
model_ada = AdaBoostClassifier()

ada_random = RandomizedSearchCV(estimator = model_ada, param_distributions = parametros_ada, n_iter = 100, cv = FOLDS, 
                               verbose=2, random_state=42, n_jobs = -1, scoring='roc_auc')
ada_random.fit(X_train, y_train)

# %% {"_uuid": "f66f2e15910c4434f809a7a0b798271b6635eb64"}
ada_random.best_params_

# %% [markdown] {"_uuid": "21ae79ff86446e20248ca511c6d608d138c77589"}
# ** 4° Gradient Boosting**

# %% {"_uuid": "24540edd4693f730198248207547bcd67b9aa7a1"}
parametros_gb = {
    "loss":["deviance"],
    "learning_rate": [0.01, 0.025, 0.005,0.5, 0.075, 0.1, 0.15, 0.2,0.3,0.8,0.9],
    "min_samples_split": [0.01, 0.025, 0.005,0.4,0.5, 0.075, 0.1, 0.15, 0.2,0.3,0.8,0.9],
    "min_samples_leaf": [1,2,3,5,8,10,15,20,40,50,55,60,65,70,80,85,90,100],
    "max_depth":[3,5,8,10,15,20,25,30,40,50],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":range(1,100)
    }

# %% {"_uuid": "27659f261dd981e9cdeba0309a7c836214365cf1"}
model_gb= GradientBoostingClassifier()


gb_random = RandomizedSearchCV(estimator = model_gb, param_distributions = parametros_gb, n_iter = 100, cv = FOLDS, 
                               verbose=2, random_state=42, n_jobs = -1, scoring='roc_auc')
gb_random.fit(X_train, y_train)

# %% {"_uuid": "3d7ec918db008c026b884f52f19c92bcf798ad23"}
gb_random.best_params_

# %% [markdown] {"_uuid": "4271614cb9bd03bae8f8fe4f9071981350932f7f"}
# **5° Support Vector**

# %% {"_uuid": "5acdc09d776e72cf3230a4541693de4d1f6cbc4e"}
C = [0.001, 0.10, 0.1, 10, 25, 50,65,70,80,90, 100, 1000]

kernel =  ['linear', 'poly', 'rbf', 'sigmoid']
    
gamma =[1e-2, 1e-3, 1e-4, 1e-5,1e-6,1]

# %% {"_uuid": "e62f389fe79116fe57af2076ae066bd4e2e5e87e"}
# Create the random grid
parametros_svm = {'C': C,
            'gamma': gamma,
             'kernel': kernel}

# %% {"_uuid": "6ac468e384edce4de7baa72de1eb7af1009c34dd"}
model_svm = SVC()
from sklearn.model_selection import GridSearchCV
svm_random = GridSearchCV(model_svm, parametros_svm,  cv = FOLDS, 
                               verbose=2, n_jobs = -1, scoring='roc_auc')
svm_random.fit(X_train.values, y_train)

# %% {"_uuid": "896e14225d62ddb6e158bf86d2f91d869e812694"}
svm_random.best_params_

# %% [markdown] {"_uuid": "16a65fbf408ea34ec5d36ab63b5e36830f31850f"}
# **6° xgboost **

# %% {"_uuid": "70432072e09127c5d32aeea928b9df4a8cd99b38"}
param_xgb = {
        'silent': [False],
        'max_depth': [6, 10, 15, 20],
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'gamma': [0, 0.25, 0.5, 1.0],
        'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        'n_estimators': [50,100,120]}

# %% {"_uuid": "9ef3cb9b38c0b09b34b93d6bbc9908a70f407716"}
from sklearn.model_selection import GridSearchCV

model_xgb = xgb.XGBClassifier()
xgb_random = RandomizedSearchCV(estimator = model_xgb, param_distributions = param_xgb, n_iter = 100, cv = FOLDS, 
                               verbose=2, random_state=42, n_jobs = -1, scoring='roc_auc')
xgb_random.fit(X_train.values, y_train)

# %% {"_uuid": "4055baa61e206adebb6b77159b66c4d66e3b770e"}
xgb_random.best_params_

# %% [markdown] {"_uuid": "a7fe396d5d2edef803ae724290460ff155e3c28c"}
# # Selected Parameters
#
# After running RandomizedSearchCV several times, we found the most acceptable parameters for each of our models.
# We will save these parameters to then make the adjustment of our models.

# %% {"_uuid": "bbb88dbac57aa61404c291e3828305d468e43fdf"}
parametro_rf = {'n_estimators': 133,
 'min_samples_split': 3,
 'max_features': 'auto',
 'max_depth': 39}

parametro_et = {'n_estimators': 46,
 'min_samples_split': 3,
 'max_features': 'sqrt',
 'max_depth': 20}

parametro_ada = {'n_estimators': 40, 'learning_rate': 0.9}  

parametro_gb = {'subsample': 0.95,
 'n_estimators': 96,
 'min_samples_split': 0.15,
 'min_samples_leaf': 5,
 'max_features': 'log2',
 'max_depth': 50,
 'loss': 'deviance',
 'learning_rate': 0.15,
 'criterion': 'friedman_mse'}

parametro_svm = {'C': 25, 'gamma': 1, 'kernel': 'rbf'}

parametro_xgb= {'subsample': 0.6,
 'silent': False,
 'reg_lambda': 1.0,
 'n_estimators': 120,
 'min_child_weight': 0.5,
 'max_depth': 15,
 'learning_rate': 0.2,
 'gamma': 0.5,
 'colsample_bytree': 0.4,
 'colsample_bylevel': 1.0}


# %% [markdown] {"_uuid": "0abfb3c44838393ea9c97b5a491dc17392ed3956"}
# <a id="ch17"></a>
# ## 4. 2 Generating our models
#
# So now let's prepare five learning models as our classification. All these models can be invoked conveniently through the Sklearn library and are listed below:
#
# 1. random forest sorter
# 2. AdaBoost classifier.
# 3. Gradient Boosting classifer
# 4. Support vector machine
# 5. Extra Trees
#

# %% {"_uuid": "ff2764d379d9ccd90116093163fe4a6f326964d0"}
 #base models with hyper parameters already tuned
model_rf =  RandomForestClassifier(n_estimators=133,min_samples_split=3,max_features='auto',max_depth= 39)
model_et = ExtraTreesClassifier(n_estimators=133,min_samples_split=3,max_features='sqrt',max_depth= 20)
model_ada = AdaBoostClassifier(n_estimators=40,learning_rate=0.9)
model_gb = GradientBoostingClassifier(subsample = 0.95,n_estimators= 96,
                 min_samples_split = 0.15,
                 min_samples_leaf = 5,
                 max_features = 'log2',
                 max_depth = 50,
                 loss = 'deviance',
                 learning_rate = 0.15,
                 criterion= 'friedman_mse')
model_svc = SVC(C = 25, gamma= 1, kernel ='rbf')
model_xgb = xgb.XGBClassifier(psubsample= 0.6,
 silent= False,
 reg_lambda =1.0,
 n_estimators= 120,
 min_child_weight= 0.5,
 max_depth = 15,
 learning_rate= 0.2,
 gamma= 0.5,
 colsample_bytree=0.4,
 colsample_bylevel= 1.0)

# %% [markdown] {"_uuid": "ff80e40928f8e0b277c3d6ba3b18c497ec561910"}
# <a id="ch18"></a>
# ## 4.3 Cross Validation

# %% {"_uuid": "4aa9a339280fa5c49536d4ad2370ab64ab01da11"}
kf = KFold(n_splits=FOLDS, random_state = 0, shuffle = True)
for i, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
    Xtrain, Xval = X_train.values[train_index], X_train.values[val_index]
    ytrain, yval = y_train[train_index], y_train[val_index]
    
    model_rf.fit(Xtrain, ytrain)
    model_et.fit(Xtrain, ytrain)
    model_ada.fit(Xtrain, ytrain)
    model_gb.fit(Xtrain, ytrain)
    model_svc.fit(Xtrain, ytrain)
    model_xgb.fit(Xtrain, ytrain)
    

# %% [markdown] {"_uuid": "b805c2063d5d28565fc49f28d07526eb68f50057"}
# <a id="ch19"></a>
# # 5. Importance of characteristics 
#
# According to the Sklearn documentation, most classifiers are built with an attribute that returns important features by simply typing *. Feature_importances _ *. Therefore, we will invoke this very useful attribute through our graph of the function of the importance of the characteristic as such

# %% {"_uuid": "876489b6241cfdfb0af7ed89a2b49398a1778728"}
rf_feature = model_rf.feature_importances_
ada_feature = model_ada.feature_importances_
gb_feature = model_gb.feature_importances_
et_feature = model_et.feature_importances_
xbg_feature = model_xgb.feature_importances_

# %% {"_uuid": "de52135815cad54968212af34dd0afed526fea9b"}
cols = X.columns.tolist()
# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'Random Forest feature importances': rf_feature,
      'AdaBoost feature importances': ada_feature,
    'Gradient Boost feature importances': gb_feature,
    'Extra Trees  feature importances': et_feature,
    'Xgboost feature importances': xbg_feature,
    })

# %% {"_uuid": "7c53ce1cd3d0fb9033187647ec813648aecd9810"}
xbg_feature

# %% [markdown] {"_uuid": "d92c28b2dfbfb1c119b7bae4036edd15ca14c994"}
# ## Graphics:

# %% {"_uuid": "59592632ec80adffcec9396ea18deb32f0760088"}
# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['Random Forest feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Random Forest feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Random Forest Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['Extra Trees  feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Extra Trees  feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Extra Trees Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['AdaBoost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['AdaBoost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'AdaBoost Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['Gradient Boost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Gradient Boost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Gradient Boosting Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

trace = go.Scatter(
    y = feature_dataframe['Xgboost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Xgboost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'XgboostFeature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')


# %% {"_uuid": "4e9d06f9f3e3ff30a4f3805751abe98f09aee464"}
# Create the new column that contains the average of the values.
feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) # axis = 1 computes the mean row-wise
feature_dataframe.head(3)

# %% {"_uuid": "167109f7330a64aa19fc979c1fe9562457470b94"}
y = feature_dataframe['mean'].values
x = feature_dataframe['features'].values
data = [go.Bar(
            x= x,
             y= y,
            width = 0.5,
            marker=dict(
               color = feature_dataframe['mean'].values,
            colorscale='Portland',
            showscale=True,
            reversescale = False
            ),
            opacity=0.6
        )]

layout= go.Layout(
    autosize= True,
    title= 'Barplots of Mean Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='bar-direct-labels')

# %% [markdown] {"_uuid": "cb38ada668858a6dd3cd49110e6f095ad6991551"}
# <a id="ch20"></a>
# # 6. Predictions

# %% {"_uuid": "d8120099038b8d81ccc82a4df185ee98a2e5e127"}
Predicted_rf= model_rf.predict(X_test)
Predicted_ada = model_ada.predict(X_test)
Predicted_gb = model_gb.predict(X_test)
Predicted_et = model_et.predict(X_test)
Predicted_svm= model_svc.predict(X_test)
Predicted_xgb= model_xgb.predict(X_test.values)

# %% {"_uuid": "402892351d0958e2762bf85eabc8468642f7ea00"}
base_predictions_train = pd.DataFrame( {'RandomForest': Predicted_rf.ravel(),
      'AdaBoost': Predicted_ada.ravel(),
      'GradientBoost': Predicted_gb.ravel(),
      'ExtraTrees': Predicted_et.ravel(),
      'SVM': Predicted_svm.ravel(),
      'XGB': Predicted_xgb.ravel(),
     'Real value': y_test                                
                                        
    })
base_predictions_train.head(10)

# %% [markdown] {"_uuid": "3bef90fe8837e9a36e9a5a3c15d56b4f60ed526b"}
# <a id="ch21"></a>
# # 7. Performance Metric for each model

# %% {"_uuid": "1c2dc866fd6e058796f9da54fbad61ecf7f5c9f9"}
acc = [] # list to store all performance metric

# %% {"_uuid": "c3728209f44942309e3a72f03c9479d62d4e9b66"}
model='Random Forest'
test_score = cross_val_score(model_rf, X_train, y_train, cv=FOLDS, scoring='accuracy').mean() # Get recall for each parameter setting
test_recall = recall_score(y_test, Predicted_rf, pos_label=1)
fpr, tpr, thresholds = roc_curve(y_test, Predicted_rf, pos_label=1)
test_auc = auc(fpr, tpr)
acc.append([model,test_score, test_recall, test_auc, fpr, tpr, thresholds])

model='AdaBoost'
test_score = cross_val_score(model_ada, X_train, y_train, cv=FOLDS, scoring='accuracy').mean() # Get recall for each parameter setting
test_recall = recall_score(y_test, Predicted_ada, pos_label=1)
fpr, tpr, thresholds = roc_curve(y_test, Predicted_ada, pos_label=1)
test_auc = auc(fpr, tpr)
acc.append([model, test_score,test_recall, test_auc, fpr, tpr, thresholds])

model='Gradient Boosting'
test_score = cross_val_score(model_gb, X_train, y_train, cv=FOLDS, scoring='accuracy').mean() # Get recall for each parameter setting
test_recall = recall_score(y_test, Predicted_gb, pos_label=1)
fpr, tpr, thresholds = roc_curve(y_test, Predicted_gb, pos_label=1)
test_auc = auc(fpr, tpr)
acc.append([model, test_score,test_recall, test_auc, fpr, tpr, thresholds])

model='ExtraTrees'
test_score = cross_val_score(model_et, X_train, y_train, cv=FOLDS, scoring='accuracy').mean() # Get recall for each parameter setting
test_recall = recall_score(y_test, Predicted_et, pos_label=1)
fpr, tpr, thresholds = roc_curve(y_test, Predicted_et, pos_label=1)
test_auc = auc(fpr, tpr)
acc.append([model, test_score, test_recall, test_auc, fpr, tpr, thresholds])

model='SVM'
test_score = cross_val_score(model_svc, X_train, y_train, cv=FOLDS, scoring='accuracy').mean() # Get recall for each parameter setting
test_recall = recall_score(y_test, Predicted_svm, pos_label=1)
fpr, tpr, thresholds = roc_curve(y_test, Predicted_svm, pos_label=1)
test_auc = auc(fpr, tpr)
acc.append([model, test_score, test_recall, test_auc, fpr, tpr, thresholds])

model='Xgboost'
test_score = cross_val_score(model_xgb, X_train, y_train, cv=FOLDS, scoring='accuracy').mean() # Get recall for each parameter setting
test_recall = recall_score(y_test, Predicted_xgb, pos_label=1)
fpr, tpr, thresholds = roc_curve(y_test, Predicted_xgb, pos_label=1)
test_auc = auc(fpr, tpr)
acc.append([model,test_score, test_recall, test_auc, fpr, tpr, thresholds])


# %% [markdown] {"_uuid": "c66c6a8caeb84400028677930f5bce36e0a98284"}
# <a id="ch22"></a>
# ## 7.1 Report 
#
# for the Extra Trees model
#

# %% {"_uuid": "d229572c1852b254423d5edc17537c1737a1df59"}
report_performance(model_et)

# %% [markdown] {"_uuid": "e616329ca5ac4481d9b7e8642cf2cac860b10b04"}
# <a id="ch23"></a>
# ## 7.1 Results

# %% {"_uuid": "ed8a304646889765850fd27ad65aa6fa43caf6ff"}
result = pd.DataFrame(acc, columns=['Model', 'Accuracy', 'Recall', 'AUC', 'FPR', 'TPR', 'TH'])
result[['Model', 'Accuracy', 'Recall', 'AUC']]

# %% [markdown] {"_uuid": "134d3a98a6dfeac6ed5b28ffdf627b6082b83e64"}
#
