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

voice = pd.read_csv('voice_dataset.csv', sep=',')

voice.info()
voice.isnull().sum()

#---------(pd.cut 'bins') is required only for value thresholds --------------------
bins = [1,5,9]
groupnames = ['Male', 'Female']

voice['label'] = pd.cut( voice['label'], bins = bins, labels = groupnames)
voice['label'].unique()
#-----------------------------------------------------------------------------------

#--- Label to 'int' encoder ---#
#label_quality = LabelEncoder()
#voice['label'] = label_quality.fit_transform(voice['label'])

print( "Value Counts :\n", voice['label'].value_counts() )

X = voice.drop(['label'], axis=1)
y = voice['label']

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.30  )
pX_train = X_train # X_train preserving (before StandardScaler.fit_transform) for pickle file

## MUST Transform ##
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#---- Randon Forest Classifire (sklearn.ensemble) -----# 
rfc = RandomForestClassifier(n_estimators = 300, verbose=1)
rfc.fit(X_train,y_train)
plog = rfc.predict_proba(X_train)
score = rfc.score(X_train,y_train)
rfc_output = rfc.predict(X_test)
#print(rfc_output[:30])
print('Decision_path', plog)
print('score', score)

print(classification_report(y_test, rfc_output))
print(confusion_matrix(y_test, rfc_output))

#---- SVM Classifire (sklearn.svm) ----#
clf = SVC()
clf.fit(X_train,y_train)
clf_output = clf.predict(X_test)

print(classification_report(y_test, clf_output))
print(confusion_matrix(y_test, clf_output))

#---- Neural Networks (sklearn.neural_network) ----#
mlpc =  MLPClassifier(hidden_layer_sizes=(10,10,12,12), max_iter=1000)
mlpc.fit(X_train,y_train)
mlpc_output = mlpc.predict(X_test)

print(classification_report(y_test, mlpc_output))
print(confusion_matrix(y_test, mlpc_output))

#-- Accuracy --# 
print("\n")
print("RFC :", accuracy_score(y_test, rfc_output))
print("SVC :", accuracy_score(y_test, clf_output))
print("Neu :", accuracy_score(y_test, mlpc_output))
print("\n")
#####################
#-- Teating new data --
#Xtest_data = [[0.19859962,0.034093051,0.201526316,0.182,0.217,0.035,2.319008526,8.378493745,0.85623409,0.243939254,0.200789474,0.19859962,0.169922188,0.048241206,0.277456647,1.015024038,0.0234375,8.203125,8.1796875,0.099271729]] #Female
Xtest_data = [[0.138550521,0.077053993,0.127526596,0.08731383,0.202739362,0.115425532,1.626769867,6.291365079,0.966003775,0.752042013,0.012101064,0.138550521,0.104199319,0.019138756,0.262295082,0.24609375,0.0078125,2.71875,2.7109375,0.132351372]] #Male
#Xtest_data = [[0.197566161,0.033313703,0.201615599,0.177827298,0.219554318,0.041727019,1.775693998,5.301886432,0.850975747,0.215871405,0.218384401,0.197566161,0.154907432,0.048096192,0.272727273,1.64390625,0.0234375,9.6328125,9.609375,0.095018163]] #Female
#Xtest_data = [[0.1841419,0.053220381,0.203671233,0.129643836,0.222849315,0.093205479,2.066273446,7.260472007,0.892495457,0.339382204,0.214410959,0.1841419,0.125153413,0.047244094,0.277456647,0.795572917,0.0234375,6.046875,6.0234375,0.097355674], 
 #             [0.175614173,0.062979937,0.2,0.119661017,0.22,0.100338983,2.623436363,11.39969186,0.922510573,0.558088211,0.218305085,0.175614173,0.107823584,0.04,0.150943396,1.000679348,0.15625,5.1328125,4.9765625,0.156843157], 
 #             [0.112574609,0.080004969,0.131173021,0.030997067,0.172434018,0.14143695,2.471338483,10.23617621,0.942924916,0.657554266,0.008621701,0.112574609,0.150547581,0.016243655,0.275862069,0.3209375,0.0078125,0.7734375,0.765625,0.260425909], 
 #            [0.189613974,0.035932606,0.194115545,0.168433591,0.205288862,0.036855271,2.72441469,10.98686402,0.871215305,0.236683539,0.195616438,0.189613974,0.163058558,0.029684601,0.258064516,1.370192308,0.1640625,7.0,6.8359375,0.235948052],
 #            [0.151921813,0.058063399,0.156683938,0.098238342,0.198134715,0.099896373,2.890883873,12.17547315,0.89198311,0.403200384,0.098031088,0.151921813,0.109455888,0.073529412,0.277777778,0.400878906,0.09765625,0.825195313,0.727539063,0.565855705]] 
              # ['M' 'F' 'F' 'F' 'M]

Xtest_data = sc.transform(Xtest_data)
Result = rfc.predict(Xtest_data)

print(Xtest_data, '\n Answer :' , Result)


##################################################
# Writting to pickle
import pickle

with open('voice_Xy_train.pickle', 'wb') as f:
   pickle.dump([pX_train, y_train], f )

#reading from pickle
with open('voice_Xy_train.pickle', 'rb') as f:
  pX_train, y_train = pickle.load(f)


#----------------------------------------------------------------------------


