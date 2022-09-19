import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


#Reading from pickle
with open('tumor_Xy_train.pickle', 'rb') as f:
    X_train, y_train = pickle.load(f)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)


#-- Teating new data --
#Xtest_data = [[5,60,-100000,5,1]] #Yes
#Xtest_data = [[-100000,66,-100000,-100000,1]] #Yes
Xtest_data = [[4,45,2,1,3]] #No
#Xtest_data = [[4,53,-100000,5,3], [2,65,-100000,1,2], [5,75,4,5,3], [5,-100000,4,3,3], [4,40,1,-100000,-100000]]  # ['Y' 'N' 'Y' 'Y' 'N']


Xtest_data  = sc.transform(Xtest_data)

rfc = RandomForestClassifier(n_estimators = 300)
rfc.fit(X_train,y_train)
Result = rfc.predict(Xtest_data)

print('\n RFC Answer :' , Result)

