import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

tumor = pd.read_csv('tumor_dataset.csv', sep=',')

tumor.info()
tumor.isnull().sum()

#---------(pd.cut 'bins') is required only for value thresholds --------------------
bins = [-0.5,0.5,1.5]
groupnames = ['No', 'Yes']

tumor['severity'] = pd.cut( tumor['severity'], bins = bins, labels = groupnames)
tumor['severity'].unique()
#-----------------------------------------------------------------------------------

#label_quality = LabelEncoder()
#tumor['severity'] = label_quality.fit_transform(tumor['severity'])

print( "Value Counts :\n", tumor['severity'].value_counts() )

X = tumor.drop(['severity'], axis=1)
y = tumor['severity']

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.30  )
pX_train = X_train # X_train preserving (before StandardScaler.fit_transform) for pickle file

## MUST Transform ##
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#---- Randon Forest Classifire (sklearn.ensemble) -----# 
rfc = RandomForestClassifier(n_estimators = 300)
rfc.fit(X_train,y_train)
rfc_output = rfc.predict(X_test)
#print(rfc_output[:30])
print('score', rfc.score(X_train,y_train) )



print(classification_report(y_test, rfc_output))
print(confusion_matrix(y_test, rfc_output))

#---- SVM Classifire (sklearn.svm) ----#
clf = SVC()
clf.fit(X_train,y_train)
clf_output = clf.predict(X_test)

print(classification_report(y_test, clf_output))
print(confusion_matrix(y_test, clf_output))

#---- Neural Networks (sklearn.neural_network) ----#
mlpc =  MLPClassifier(hidden_layer_sizes=(10,10,10,10,10), max_iter=1000)
mlpc.fit(X_train,y_train)
mlpc_output = mlpc.predict(X_test)

print(classification_report(y_test, mlpc_output))
print(confusion_matrix(y_test, mlpc_output))

#-- Accuracy --#
print("RFC :", accuracy_score(y_test, rfc_output))
print("SVC :", accuracy_score(y_test, clf_output))
print("Neural :", accuracy_score(y_test, mlpc_output))

#####################
#-- Teating new data --
#Xtest_data = [[5,60,-100000,5,1]] #Yes
Xtest_data = [[-100000,66,-100000,-100000,1]] #Yes
#Xtest_data = [[4,45,2,1,3]] #No
Xtest_data = [[4,53,-100000,5,3], [2,65,-100000,1,2], [5,75,4,5,3], [5,-100000,4,3,3], [4,40,1,-100000,-100000]]  # ['Y' 'N' 'Y' 'Y' 'N']

Xtest_data = sc.transform(Xtest_data)
Result = rfc.predict(Xtest_data)

print(Xtest_data, '\n Answer :' , Result)


##################################################
# Writting to pickle
import pickle

with open('tumor_Xy_train.pickle', 'wb') as f:
   pickle.dump([pX_train, y_train], f )

#reading from pickle
with open('tumor_Xy_train.pickle', 'rb') as f:
  pX_train, y_train = pickle.load(f)


#----------------------------------------------------------------------------


