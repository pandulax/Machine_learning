import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

wine = pd.read_csv('wine_data.csv', sep=',')

wine.info()
wine.isnull().sum()

#---------(pd.cut 'bins') is required only for value thresholds --------------------
bins = [5,15,25,35]
groupnames = ['One', 'Two', 'Three']

wine['Wine_Type'] = pd.cut( wine['Wine_Type'], bins = bins, labels = groupnames)
wine['Wine_Type'].unique()
#-----------------------------------------------------------------------------------

#--- Label to 'int' encoder ---#
#label_quality = LabelEncoder()
#wine['Wine_Type'] = label_quality.fit_transform(wine['Wine_Type'])

print( "Value Counts :\n", wine['Wine_Type'].value_counts() )

X = wine.drop(['Wine_Type'], axis=1)
y = wine['Wine_Type']

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.3  )
X_train_beforeScaling = X_train  #X_train preserving (before StandardScaler.fit_transform) for pickle file

##  Transform ##
sc = StandardScaler()
#sc = MinMaxScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


#---- Randon Forest Classifire (sklearn.ensemble) -----# 
rfc = RandomForestClassifier(n_estimators = 600)
rfc.fit(X_train,y_train)
rfc_output = rfc.predict(X_test)
#print(rfc_output[:30])

print(classification_report(y_test, rfc_output))
print(confusion_matrix(y_test, rfc_output))

#---- SVM Classifire (sklearn.svm) ----#
clf = SVC()
clf.fit(X_train,y_train)
clf_output = clf.predict(X_test)

print(classification_report(y_test, clf_output))
print(confusion_matrix(y_test, clf_output))

#---- Neural Networks (sklearn.neural_network) ----#
mlpc =  MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=500)
mlpc.fit(X_train,y_train)
mlpc_output = mlpc.predict(X_test)

print(classification_report(y_test, mlpc_output))
print(confusion_matrix(y_test, mlpc_output))

#-- Accuracy --#
print("\nRFC :", accuracy_score(y_test, rfc_output))
print("SVC :", accuracy_score(y_test, clf_output))
print("Neu :", accuracy_score(y_test, mlpc_output))


#############################################################
#------ Teating new data ------#
Xtest_data = [[12.72,1.81,2.2,18.8,86,2.2,2.53,0.26,1.77,3.9,1.16,3.14,714], 
              [14.1,2.16,2.3,18,105,2.95,3.32,0.22,2.38,5.75,1.25,3.17,1510], 
              [13.4,3.91,2.48,23,102,1.8,0.75,0.43,1.41,7.3,0.7,1.56,750], 
              [12.29,1.41,1.98,16,85,2.55,2.5,0.29,1.77,2.9,1.23,2.74,428]]            # ['Two' 'One' 'Three' 'Two']
#Xtest_data = [[13.73,1.5,2.7,22.5,101,3,3.25,0.29,2.38,5.7,1.19,2.71,1285], [13.58,2.58,2.69,24.5,105,1.55,0.84,0.39,1.54,8.66,0.74,1.8,750]] #one #three
#Xtest_data = [[14.37,1.95,2.5,16.8,113,3.85,3.49,0.24,2.18,7.8,0.86,3.45,1480]]       # one
#Xtest_data = [[11.66,1.88,1.92,16,97,1.61,1.57,0.34,1.15,3.8,1.23,2.14,428]]          # two
Xtest_data = [[12.77,2.39,2.28,19.5,86,1.39,0.51,0.48,0.64,9.899999,0.57,1.63,470]]   # three
Xtest_data = [[13.5,3.12,2.62,24,123,1.4,1.57,0.22,1.25,8.6,0.59,1.3,500]]  #three
#Xtest_data = [[13.27,4.28,2.26,20,120,1.59,0.69,0.43,1.35,10.2,0.59,1.56,835]] #three

Xtest_data1 = sc.transform(Xtest_data)
Result = rfc.predict(Xtest_data1)

print("\n", Xtest_data1, '\n\n Answer :' , Result, "\n")


##################################################
import joblib

#--- Writting to pickle ---#
joblib.dump( [X_train_beforeScaling, y_train], 'finalized.pkl' )

#--- Reading from pickle ---#
nX_train, ny_train = joblib.load('finalized.pkl')

new_sc1, new_sc2 = StandardScaler() , MinMaxScaler()
new_sc = new_sc1

new_sc.fit(nX_train)
nX_train = new_sc.transform(nX_train)

nXtest_data = new_sc.transform(Xtest_data)

n_rfc = RandomForestClassifier(n_estimators = 600)
n_rfc.fit(nX_train, ny_train)
result = n_rfc.predict(nXtest_data)

print("\n", result )

#----------------------------------------------------------------------------


