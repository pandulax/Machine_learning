import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


#Reading from pickle
with open('Xy_train.pickle', 'rb') as f:
    X_train, y_train = pickle.load(f)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)


Xtest_data = [[13.73,1.5,2.7,22.5,101,3,3.25,0.29,2.38,5.7,1.19,2.71,1285], [13.58,2.58,2.69,24.5,105,1.55,0.84,0.39,1.54,8.66,0.74,1.8,750]] #one #three
#Xtest_data = [[11.66,1.88,1.92,16,97,1.61,1.57,0.34,1.15,3.8,1.23,2.14,428]] #two
#Xtest_data = [[13.58,2.58,2.69,24.5,105,1.55,0.84,0.39,1.54,8.66,0.74,1.8,750]] #three

Xtest_data  = sc.transform(Xtest_data)

rfc = RandomForestClassifier(n_estimators = 300)
rfc.fit(X_train,y_train)
Result = rfc.predict(Xtest_data)

print('\n RFC Answer :' , Result)

