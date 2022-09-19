
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score


autompg = pd.read_csv('auto_mpg_data.csv', sep=',')

#print( "Value Counts :\n", autompg['mpg'].value_counts() )

X = autompg.drop(['mpg', 'car_name'], axis=1)
y = autompg['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 )

scaler = MinMaxScaler()
scaler.fit(X_train)
sc_X_train = scaler.transform(X_train)
sc_X_test = scaler.transform(X_test)

linereg = LinearRegression()
linereg.fit(X_train, y_train)

linereg_result = linereg.predict(X_test) 

print("\n LinearReg:")
print("Measn squared error:" , mean_squared_error(y_test, linereg_result))
print("Variance score:", r2_score(y_test, linereg_result))

#Xtest_data = [[8,390.0,190.0,3850.0,8.5,70,1], [4,79.0,67.0,1963.0,15.5,74,2]] #15, 26
#Xtest_data = [[8,350.0,160.0,4456.0,13.5,72,1]] #12
Xtest_data = [[8,390.0,190.0,3850.0,8.5,70,1], [4,79.0,67.0,1963.0,15.5,74,2], [4,97.0,67.0,2065.0,17.8,81,3]] #15, 26, 32.3
#Xtest_data = [[8,400.0,170.0,4746.0,12.0,71,1], [8,305.0,140.0,4215.0,13.0,76,1], [6,258.0,110.0,2962.0,13.5,71,1], [4,97.0,75.0,2265.0,18.2,77,3], [4,121.0,110.0,2600.0,12.8,77,2]] 
#              #13, 17.5, 18, 26, 21.5
#Xtest_data = [[4,120.0,74.0,2635.0,18.3,81,3 ], [4,141.0,80.0,3230.0,20.4,81,2], [6,145.0,76.0,3160.0,19.6,81,2]]  #31.6, 28.1, 30.7

#sc_Xtest_data = scaler.transform(Xtest_data)

print("\n", linereg.predict(Xtest_data), "\n----------------------")


#------ RFR (Random Forest Regressor) ------#
RFReg = RandomForestRegressor(n_estimators=300)
RFReg.fit(sc_X_train, y_train)
RFReg_predict = RFReg.predict(sc_X_test)

print("\n RFReg:") #print("\n RFReg:", RFReg_predict)
print("Measn squared error:" , mean_squared_error(y_test, RFReg_predict))
print("Variance score:", r2_score(y_test, RFReg_predict))
#--------------------------------------------#

#------- SVR (Support Vector Regressor) -------#
clf = SVR(gamma='scale')
clf.fit(X_train, y_train)
clf_predict = clf.predict(X_test)

print("\n SVReg:") #print("\n SVReg:", clf_predict)
print("Measn squared error:" , mean_squared_error(y_test, clf_predict))
print("Variance score:", r2_score(y_test, clf_predict))
#--------------------------------------------#

#------ MLPR (Neural Network Regressor) -----#
MLPR = MLPRegressor(hidden_layer_sizes=(20, 15, 15 ), max_iter =1000)
MLPR.fit(X_train, y_train)
MLPR_predict = MLPR.predict(X_test)

print("\n NNReg:") #print("\n NNReg:", MLPR_predict)
print("Measn squared error:" , mean_squared_error(y_test, MLPR_predict))
print("Variance score:", r2_score(y_test, MLPR_predict))
#--------------------------------------------#


#------ ABR (Ada Boost Regressor) -----------#
ABR = AdaBoostRegressor(n_estimators=500)
ABR.fit(X_train, y_train)
ABR_predict = ABR.predict(X_test)

print("\n ABReg:") #print("\n ABReg:", ABR_predict)
print("Measn squared error:" , mean_squared_error(y_test, ABR_predict))
print("Variance score:", r2_score(y_test, ABR_predict))

#--------------------------------------------#

##################################################################
import pickle

#----- Saving to pickle -----#
model = RFReg
pickle.dump( [model,scaler] , open('mpg_finalized.pkl', 'wb') )

#----- Loading from pickle -----#
loaded_model, loaded_scaler = pickle.load (open('mpg_finalized.pkl', 'rb'))

Xtest_data = loaded_scaler.transform(Xtest_data)
result = loaded_model.predict(Xtest_data)

print ("\n---------------------\n Loaded Result:", result)



