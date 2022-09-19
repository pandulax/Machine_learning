import pickle
import pandas as pd

data = input("MPG data: ")

autodata = []

#Xtest_data = [[8,350.0,160.0,4456.0,13.5,72,1]] #12
Xtest_data = [[8,390.0,190.0,3850.0,8.5,70,1], [4,79.0,67.0,1963.0,15.5,74,2], [4,97.0,67.0,2065.0,17.8,81,3]] #15, 26, 32.3
#Xtest_data = [[8,400.0,170.0,4746.0,12.0,71,1], [8,305.0,140.0,4215.0,13.0,76,1], [6,258.0,110.0,2962.0,13.5,71,1], [4,97.0,75.0,2265.0,18.2,77,3], [4,121.0,110.0,2600.0,12.8,77,2]] 
#              #13, 17.5, 18, 26, 21.5
#Xtest_data = [[4,120.0,74.0,2635.0,18.3,81,3 ], [4,141.0,80.0,3230.0,20.4,81,2], [6,145.0,76.0,3160.0,19.6,81,2]]  #31.6, 28.1, 30.7


if (data != ''):
  for i in data.split(","):
    autodata.append(float(i))
  
  Xtest_data = [autodata]


#----- Loading from pickle -----#
loaded_model, loaded_scaler = pickle.load(open('mpg_finalized.pkl', 'rb'))

Xtest_data = loaded_scaler.transform(Xtest_data)
result = loaded_model.predict(Xtest_data)

print ("\n---------------------\n Loaded Result:", result)