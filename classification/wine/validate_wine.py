import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
sc1 = StandardScaler()
sc2 = MinMaxScaler()

Xtest_data = [[12.72,1.81,2.2,18.8,86,2.2,2.53,0.26,1.77,3.9,1.16,3.14,714],
              [14.1,2.16,2.3,18,105,2.95,3.32,0.22,2.38,5.75,1.25,3.17,1510], 
              [13.4,3.91,2.48,23,102,1.8,0.75,0.43,1.41,7.3,0.7,1.56,750],
              [12.29,1.41,1.98,16,85,2.55,2.5,0.29,1.77,2.9,1.23,2.74,428],
              [12.77,2.39,2.28,19.5,86,1.39,0.51,0.48,0.64,9.899999,0.57,1.63,470],
              [14.06,2.15,2.61,17.6,121,2.6,2.51,0.31,1.25,5.05,1.06,3.58,1295]] # ['Two' 'One' 'Three' 'Two' 'Three' 'One']

#Xtest_data = [[12.93,3.8,2.65,18.6,102,2.41,2.41,0.25,1.98,4.5,1.03,3.52,770]]  #one
#Xtest_data = [[14.1,2.16,2.3,18,105,2.95,3.32,0.22,2.38,5.75,1.25,3.17,1510]]   #one
Xtest_data = [[12.72,1.81,2.2,18.8,86,2.2,2.53,0.26,1.77,3.9,1.16,3.14,714]]   #two
#Xtest_data = [[12.29,1.41,1.98,16,85,2.55,2.5,0.29,1.77,2.9,1.23,2.74,428]]    #two
#Xtest_data = [[13.5,3.12,2.62,24,123,1.4,1.57,0.22,1.25,8.6,0.59,1.3,500]]     #three
#Xtest_data = [[13.4,3.91,2.48,23,102,1.8,0.75,0.43,1.41,7.3,0.7,1.56,750]]     #three
Xtest_data = [[13.48,1.67,2.64,22.5,89,2.6,1.1,0.52,2.29,11.75,0.57,1.78,620]] #three
#Xtest_data = [[13.27,4.28,2.26,20,120,1.59,0.69,0.43,1.35,10.2,0.59,1.56,835]] #three

#Xtest_data = sc.fit_transform(Xtest_data)

#--- Reading from pickle ---#
sc = sc1

X_train, y_train = joblib.load('finalized.pkl')

sc.fit(X_train)
X_train = sc.transform(X_train)

Xtest_data = sc.transform(Xtest_data)

new_rfc = RandomForestClassifier(n_estimators = 600)
new_rfc.fit(X_train, y_train)

result = new_rfc.predict(Xtest_data)

print("\n", result )


