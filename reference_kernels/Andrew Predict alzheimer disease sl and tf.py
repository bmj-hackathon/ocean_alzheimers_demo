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

# %% [markdown] {"_uuid": "4d5bd2294394d8606842bf7123af9e2262b2a19e", "_cell_guid": "c6776709-b038-440f-a1dc-2674478d6541"}
# # Lets try to predict CRA of alzheimer disease

# %% [markdown] {"_uuid": "9c4f96ffede32196d46ba732a45fd3228410a35a", "_cell_guid": "0bd64fc6-b528-4ef4-b871-9a4a4860f91c"}
# ![](http://yourcooladviser.in/wp-content/uploads/2017/06/stages-of-alzheimers-disease-21.jpg)

# %% {"_uuid": "2681bb0e4e2880b71de69cc9ea91407f6513f09e", "_cell_guid": "267a1107-38c1-43a2-b5df-68a6a7474f30"}
# %matplotlib inline
import keras
import glob
import seaborn as sns
import pandas as pd
import numpy as np
import timeit
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

# %% {"_uuid": "29791fd41e02d63174720e7c6ffb7a1a887d381d", "_cell_guid": "b98b10f5-2c29-4ce5-b39a-5a001b6c723a"}
cross1=pd.read_csv('../input/oasis_longitudinal.csv') 
cross1 = cross1.fillna(method='ffill')
cross2=pd.read_csv('../input/oasis_cross-sectional.csv')
cross2 = cross2.fillna(method='ffill')

# %% {"_uuid": "4a95a9e2df4e2ef94ff900caf75fc175565a4418", "_cell_guid": "74444795-315e-459c-9180-1d5b663fe85c"}
cross1.head()

# %% {"_uuid": "b02147bd6c37d33dd0c25f04c4b34c3ff675abad", "_cell_guid": "440bab30-3ca1-4a28-a9bd-53019637c977"}
cross2.head()

# %% {"_uuid": "4e97279d694f11eb588a00b3d40f1259d9dcd3bf", "_cell_guid": "3f4f9570-ca41-4502-bfd3-c7d2b37fe63a"}
cross1.info()

# %% {"_uuid": "3636f3e2cb5d6a9df18c02c9a263c9c2dadf8d39", "_cell_guid": "8e13b910-78b3-4c24-88ef-7a255c6f59c3"}
cross2.head()

# %% {"_uuid": "88203fa8294f98c032e13c387f001cc2a1351210", "_cell_guid": "67eea83a-f77c-4896-8dee-e6a80e06fb68"}
cross2.info()

# %% {"_uuid": "423514b5e21677136624193fbb0e048738b7b3b6", "_cell_guid": "90955b40-0707-4b81-8284-a7bbf743a4bf"}
# %pylab inline
#lets plot some graphics from the first dataset

from pylab import rcParams
rcParams['figure.figsize'] = 8, 5
cols = ['Age','MR Delay', 'EDUC', 'SES', 'MMSE', 'CDR','eTIV','nWBV','ASF']
x=cross1.fillna('')
sns_plot = sns.pairplot(x[cols])

# %% {"_uuid": "4f83bdd0aa74bb1c3a79b69512b442444f008eae", "_cell_guid": "8191f2b4-09f3-4229-8546-ec245401318b"}
#lets plot correleation matrix
corr_matrix =cross1.corr()
rcParams['figure.figsize'] = 15, 10
sns.heatmap(corr_matrix)

# %% {"_uuid": "1e6baa557b66aaa38c045d01077e03b5ccafb00f", "_cell_guid": "429e3d6a-cffd-43d8-bd84-8b5edab83186"}
cross1.drop(['MRI ID'], axis=1, inplace=True)
cross1.drop(['Visit'], axis=1, inplace=True)

# %% {"_uuid": "573f598c75047ecd308be2ab4f0b44a565209b53", "_cell_guid": "473fb792-26e2-467d-9557-c180f9e04cc9"}
#cdr=cross1["CDR"]
cross1['CDR'].replace(to_replace=0.0, value='A', inplace=True)
cross1['CDR'].replace(to_replace=0.5, value='B', inplace=True)
cross1['CDR'].replace(to_replace=1.0, value='C', inplace=True)
cross1['CDR'].replace(to_replace=2.0, value='D', inplace=True)

# %% {"_uuid": "6a73c282acd404016abfe3d2e4164d3ff5e0f371", "_cell_guid": "586e6b82-a3c4-4490-9426-1e70191c4491"}
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
for x in cross1.columns:
    f = LabelEncoder()
    cross1[x] = f.fit_transform(cross1[x])

# %% {"_uuid": "85cfb4715d89b5c3c3d6cb95030b851ef92835f6", "_cell_guid": "b695d43e-09d6-424b-811d-246b5d6225b7"}
cross1.head()

# %% {"_uuid": "38fe0bc44b7fa1b15c1679d67e54b8233cadc6a1", "_cell_guid": "36d12732-bdf5-4d1a-be05-dd62fee5030c"}
#cdr.replace(to_replace=0.0, value='A', inplace=True)
#cdr.replace(to_replace=0.5, value='B', inplace=True)
#cdr.replace(to_replace=1.0, value='C', inplace=True)
#cdr.replace(to_replace=2.0, value='D', inplace=True)

# %% {"_uuid": "021db6993974b792c3305034e0b0dd6db74ae566", "_cell_guid": "36f4856f-d1d7-4fb0-8124-587b9555967e"}
#from sklearn.preprocessing import LabelBinarizer
#encoder=LabelBinarizer()
#z1=encoder.fit_transform(cdr)

# %% {"_uuid": "d667a9c0c1c156caea9da6fb3f484267fdb7b063", "_cell_guid": "1150f5b5-0a18-4657-a79a-0e76517a674d"}
#print(z1)

# %% [markdown] {"_uuid": "6b4522a821ede5b0cbaab4e10ada2335bdafd853", "_cell_guid": "896c75fa-69b1-45a5-8e95-ac9f696f08aa"}
# # Lets begin some machine learning

# %% {"_uuid": "70703d344c8ea04544772841acd9e1c0a2e449a5", "_cell_guid": "333e53bb-a8a3-4856-a5be-0111c218b9d1"}
train, test = train_test_split(cross1, test_size=0.3)

# %% {"_uuid": "fa98dfba5c02038c52333fd8c640e0676d47b3c4", "_cell_guid": "650c0c98-6e9d-45e7-9158-2f905af1e62f"}
X_train = train[['M/F', 'Age', 'EDUC', 'SES',  'eTIV', 'ASF']]
y_train = train.CDR
X_test = test[['M/F', 'Age', 'EDUC', 'SES',  'eTIV',  'ASF']]
y_test = test.CDR

# %% {"_uuid": "2e668544d58ce4a2b203b2d115072d8ecf2a5cda", "_cell_guid": "ac0c330e-c547-4c2c-a6e0-05ce3dbeff1a"}
# Import `StandardScaler` from `sklearn.preprocessing`
from sklearn.preprocessing import StandardScaler

# Define the scaler 
scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)

# %% {"_uuid": "acafe6708ab686ca9f11d57f3f154ef0937c8779", "_cell_guid": "f1a2371b-e373-4402-88d3-5c7ecf2d25bd"}
y_train=np.ravel(y_train)
X_train=np.asarray(X_train)

y_test=np.ravel(y_test)
X_test=np.asarray(X_test)


# %% {"_uuid": "09563cacdf821f7995d33e0e1e257cbd36b02350", "_cell_guid": "cc229161-b9aa-4368-ba60-cecdd0b202f0"}
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)

# %% {"_uuid": "a84ac22428eed53b0e19c8c858ee34409f260a8e", "_cell_guid": "b8189d8a-b147-4b63-9f1e-d81f9d174e6f"}
classifier.score(X_test, y_test)

# %% {"_uuid": "d5511f70baa6560c349abd137457c726bcc8155a", "_cell_guid": "300584fa-7232-486d-ba2a-c7c91081540b"}
classifier.score(X_train, y_train)

# %% {"_uuid": "63daf9da6272e0a4cfd96332c4c27d966824e6be", "_cell_guid": "00b25078-aa29-46bb-8200-680f3910b19f"}
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=12)
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)
print (classifier.score(X_train, y_train))
print (classifier.score(X_test, y_test))

# %% {"_uuid": "cacd681540ce96f081fcb4d605cfe78e9ab08204", "_cell_guid": "f45aab4a-84a2-41d5-bff9-426a10c23933"}
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
print(knn.score(X_train, y_train))
prediction = knn.predict(X_test)
print(knn.score(X_test, y_test))


# %% {"_uuid": "f8869f3b6ce6cac49fc29217046b98ed90dbefb7", "_cell_guid": "f7dc411b-3ec2-48df-beb3-c94a1a126295"}
from sklearn.svm import SVC
svc=SVC(kernel="linear", C=0.01)
svc.fit(X_train, y_train)
prediction = svc.predict(X_test)

# %% {"_uuid": "deb2b98e3667ccc3dbe1c54aa4f6e75beb273203", "_cell_guid": "b44b0b47-894b-4932-bfde-93672eaf00b6"}
svc.score(X_test, y_test)

# %% {"_uuid": "a25ece93bd467ba9dd7c51a4d884b2c25742161a", "_cell_guid": "5df7095e-712a-4772-b43d-5f8a1d49bd94"}
svc.score(X_train, y_train)

# %% {"_uuid": "b5223cf204152c80a9c0b85e1ef14c7b48cfea58", "_cell_guid": "7fef7cb8-3069-4ce4-bb3c-dd31c7b4e1dc"}
X_train.shape

# %% [markdown] {"_uuid": "b4d043d081ae1dff6d75da666876fc29a32c8b3f", "_cell_guid": "26a64846-5507-4361-b940-eda4ac63f616"}
# ## Neural net tensorflow

# %% [markdown] {"_uuid": "cddac10f4be1642197a4337d9f823c372ef45631", "_cell_guid": "eb4aacbe-0c2c-4d12-bf08-80082bade719"}
# ![](http://it-nowosti.ru/wp-content/uploads/2015/11/google-otkryvaet-isxodnyj-kod-si.jpg)

# %% {"_uuid": "33d0bcbd314c3ce1d5b9303dd2c0ce8b8269e99e", "_cell_guid": "92d8bed9-98b1-4235-a243-d088e0a17006"}
import tensorflow as tf
from sklearn import metrics
X_FEATURE = 'x'  # Name of the input feature.
feature_columns = [
      tf.feature_column.numeric_column(
          X_FEATURE, shape=np.array(X_train).shape[1:])]

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[35,70, 35], n_classes=4)

  # Train.
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={X_FEATURE: X_train}, y=y_train, num_epochs=100, shuffle=False)
classifier.train(input_fn=train_input_fn, steps=1000)

  # Predict.
test_input_fn = tf.estimator.inputs.numpy_input_fn(x={X_FEATURE: X_test}, y=y_test, num_epochs=1, shuffle=False)
predictions = classifier.predict(input_fn=test_input_fn)
y_predicted = np.array(list(p['class_ids'] for p in predictions))
y_predicted = y_predicted.reshape(np.array(y_test).shape)

  # Score with sklearn.
score = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy (sklearn): {0:f}'.format(score))

  # Score with tensorflow.
scores = classifier.evaluate(input_fn=test_input_fn)
print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))


#if __name__ == '__main__':
   # tf.app.run()

# %% {"_uuid": "18c9302408842c762182018d1dee2fd1593ddc19", "_cell_guid": "cb8ec758-9ab2-42b3-939c-151cdc04cf0a"}
y_train

# %% [markdown] {"_uuid": "fcbe4f4dd098a42fa42aaac08b0728e71c36efe0", "_cell_guid": "cfc6eeae-3fed-47b8-9204-5bae0233eef8"}
# # We need to concat both datasets because we have insufficient data

# %% {"_uuid": "6c13f975fa0d5fd031d55cd81e7c64ed3ea44cc2", "_cell_guid": "f23e1151-b8bb-47d6-87ca-0825897114b1"}
cross1.head()

# %% {"_uuid": "d81b8441f09064e5eaddc30276f1f600e0922157", "_cell_guid": "3a11a351-a85b-4208-804c-1dd9e9899245"}
cross2.head()

# %% {"_uuid": "35c5d211145fddfdef2f8bca48d1f26b2752597a", "_cell_guid": "2928668b-1a4b-4d97-97ce-025a7d72246b"}
#lets encode second dataset
for x in cross2.columns:
    f = LabelEncoder()
    cross2[x] = f.fit_transform(cross2[x])

# %% {"_uuid": "b1255b81b61abc58cd3df680c53c12d8aee08733", "_cell_guid": "602714b1-d05f-4976-836c-6f919ca03a04"}
#concanting both datasets
df = pd.concat([cross1,cross2])

# %% {"_uuid": "39332621b05deb24daf0217ab17cd384465d38fb", "_cell_guid": "9d884f13-a6a9-43bd-bc40-24a2f702c300"}
df = df.fillna(method='ffill')
df.head()


# %% {"_uuid": "f5d4f787751d4f103b699b79c161ef47e592e7f2", "_cell_guid": "c9d262f7-0ef9-4b2b-b987-b88472432e3a"}
train, test = train_test_split(cross1, test_size=0.3)
X_train1 = train[['ASF', 'Age', 'EDUC', 'Group',  'Hand', 'M/F','MMSE','MR Delay','SES','eTIV','nWBV']]
y_train1 = train.CDR
X_test1 = test[['ASF', 'Age', 'EDUC', 'Group',  'Hand', 'M/F','MMSE','MR Delay','SES','eTIV','nWBV']]
y_test1 = test.CDR

# %% {"_uuid": "f7c7e1c8bd8759b06b4be041c88f2a5f8df8273e", "_cell_guid": "4a4b1784-c040-4e4a-b82a-aeeed61ca61f"}
# Import `StandardScaler` from `sklearn.preprocessing`
from sklearn.preprocessing import StandardScaler

# Define the scaler 
scaler = StandardScaler().fit(X_train1)

# Scale the train set
X_train1 = scaler.transform(X_train1)

# Scale the test set
X_test1 = scaler.transform(X_test1)

# %% {"_uuid": "f2b10fc53fee8fd4b308f83895e9675e1b1be356", "_cell_guid": "a8d79a4a-222e-41a3-a4af-de31cb2e933f"}
y_train1=np.ravel(y_train1)
X_train1=np.asarray(X_train1)

y_test1=np.ravel(y_test1)
X_test1=np.asarray(X_test1)

# %% {"_uuid": "25d95be5baf08be7e66ceaff2b111460fc2941d6", "_cell_guid": "a935355c-8fde-44b7-a342-38ddad18fa35"}
X_train1

# %% {"_uuid": "973ce43bdabe13974bbfc931795cc0d24f42e60c", "_cell_guid": "8e2490a9-2c30-48a3-9237-61f962175d6a"}
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train1, y_train1)
prediction = classifier.predict(X_test1)
print(classifier.score(X_train1, y_train1))
print(classifier.score(X_test1, y_test1))

# %% {"_uuid": "e25dd2e400d70f2d30c214b8a686ac10fce00d59", "_cell_guid": "8527128a-13b5-40a5-90be-67fc6a3a9448"}
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=5)
classifier.fit(X_train1, y_train1)
prediction = classifier.predict(X_test1)
print (classifier.score(X_train1, y_train1))
print (classifier.score(X_test1, y_test1))

# %% {"_uuid": "153e62464c3e255af017add0007355d47f2a8cfb", "_cell_guid": "0a493fd2-c7a8-47f4-a9c8-42087e18fe8c"}
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train1, y_train1)
print(knn.score(X_train1, y_train1))
prediction = knn.predict(X_test1)
print(knn.score(X_test1, y_test1))

# %% {"_uuid": "1d915f8507acbfa831fee55b9a30fe5f1a174dab", "_cell_guid": "b7419fbd-8b22-4694-ab45-9f714e7f1954"}
import tensorflow as tf
from sklearn import metrics
X_FEATURE = 'x'  # Name of the input feature.
feature_columns = [
      tf.feature_column.numeric_column(
          X_FEATURE, shape=np.array(X_train1).shape[1:])]

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[35,70,35], n_classes=4)

  # Train.
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={X_FEATURE: X_train1}, y=y_train1, num_epochs=100, shuffle=False)
classifier.train(input_fn=train_input_fn, steps=1000)

  # Predict.
test_input_fn = tf.estimator.inputs.numpy_input_fn(x={X_FEATURE: X_test1}, y=y_test1, num_epochs=1, shuffle=False)
predictions = classifier.predict(input_fn=test_input_fn)
y_predicted = np.array(list(p['class_ids'] for p in predictions))
y_predicted = y_predicted.reshape(np.array(y_test1).shape)

  # Score with sklearn.
score = metrics.accuracy_score(y_test1, y_predicted)
print('Accuracy (sklearn): {0:f}'.format(score))

  # Score with tensorflow.
scores = classifier.evaluate(input_fn=test_input_fn)
print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))

# %% {"_uuid": "7bfb3154119d5c2ed34c2e4a4cce299c9c11bb5e", "_cell_guid": "f7746291-0a39-4a11-872f-3d4e15cdae15"}
import tensorflow as tf
from sklearn import metrics
X_FEATURE = 'x'  # Name of the input feature.
feature_columns = [
      tf.feature_column.numeric_column(
          X_FEATURE, shape=np.array(X_train1).shape[1:])]

classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=4)

  # Train.
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={X_FEATURE: X_train1}, y=y_train1, num_epochs=100, shuffle=False)
classifier.train(input_fn=train_input_fn, steps=1000)

  # Predict.
test_input_fn = tf.estimator.inputs.numpy_input_fn(x={X_FEATURE: X_test1}, y=y_test1, num_epochs=1, shuffle=False)
predictions = classifier.predict(input_fn=test_input_fn)
#y_predicted = np.array(list(p['class_ids'] for p in predictions))
y_predicted = y_predicted.reshape(np.array(y_test1).shape)

  # Score with sklearn.
score = metrics.accuracy_score(y_test1, y_predicted)
print('Accuracy (sklearn): {0:f}'.format(score))

  # Score with tensorflow.
scores = classifier.evaluate(input_fn=test_input_fn)
print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))

# %% [markdown] {"_uuid": "ce792e5a7c1966c322c4a7e4549b929ded904f5c", "_cell_guid": "2f4ffcf3-e985-4d08-816e-f0e5084a7874"}
# ### And the winner is  DecisionTreeClassifier! 

# %% [markdown] {"_uuid": "ff4d2ddf246e4f54b2ecfc33df429bab96452460", "_cell_guid": "293b6e4f-0ccf-4f27-be25-81ada798e455"}
# ### Conclusion: we need more data for more precise analysis
