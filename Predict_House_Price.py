import os
import pandas as pd 
import numpy as np 
from sklearn import linear_model
from sklearn.model_selection import train_test_split

#Get home data from csv file
def getData():
	dataFile=None
	if os.path.exists('home_data.csv'):
		dataFile=pd.read_csv('home_data.csv', skipfooter=1)
	return dataFile

def linearRegressionModel(X_train, Y_train, X_test, Y_test):
	linear=linear_model.LinearRegression()
	linear.fit(X_train, Y_train)
	score_trained=linear.score(X_test, Y_test)
	y_pre=linear.predict(X_test)
	print('Gia du doan ($):', int(y_pre[0]))
	print('Gia thuc te ($):', int (Y_test[0]) )
	return score_trained

if __name__ =="__main__":
	data= getData()
	if data is not None:
		attributes=list(
			[
				'living_area',
				'num_bed',
				'year_built',
				'num_room',
				'num_bath',
				'latitude',
				'longitude',
				'Free_Parking',
				'num_parking',
				'Facilities',
			]
		)
		#Vector price of house
		Y=data['askprice']
		#Vector attributes of house
		X=data[attributes]
		#Train
		X_train, X_test, Y_train, Y_test=train_test_split(np.array(X), np.array(Y), test_size=0.2)
		linearScore=linearRegressionModel(X_train, Y_train, X_test, Y_test)
		print('linearScore: ', linearScore)