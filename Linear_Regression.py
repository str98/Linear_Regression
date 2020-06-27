import os 
import pandas as pd 
import numpy as np 
from sklearn import linear_model 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

def getData():
	#Get home data from CSV file
	data_File=None
	if os.path.exists('home_data.csv'):
		print('home_data.csv found locally')
		data_File=pd.read_csv('home_data.csv', skipfooter=1)
	return data_File


def linear_Regression_Model(X_train, Y_train, X_test, Y_test):
	linear=linear_model.LinearRegression()
	#Training Process
	linear.fit(X_train, Y_train)
	#Evaluting the model
	score_trained=linear.score(X_test, Y_test)
	return score_trained

def lasso_Regression_Model(X_train, Y_train, X_test, Y_test):
	lasso_linear=linear_model.Lasso(alpha=1.0)
	#Training Process
	lasso_linear.fit(X_train, Y_train)
	#Evaluting the model
	score_trained=lasso_linear.score(X_test, Y_test)
	return score_trained

def polynomial_Regression(X_train, Y_train, X_test, Y_test, degree):
	poly_model=Pipeline([('poly',PolynomialFeatures(degree)), ('linear', linear_model.LinearRegression(fit_intercept=False))])
	poly_model=poly_model.fit(X_train,Y_train)
	score_trained=poly_model.score(X_test, Y_test)
	return score_trained

if __name__=="__main__":
	data=getData()
	if data is not None:
		#Select few attributes
		attributes=list(
				[
					'num_bed',
					'year_built',
					'num_room',
					'num_bath',
					'living_area',
				]
			)
		#Vector price of house
		Y=data['askprice']
		#Vector attributes of house
		X=data[attributes]
		#Split data to training test and testing test
		X_train, X_test, Y_train, Y_test = train_test_split(np.array(X), np.array(Y), test_size=0.2)
		#Linear Regression Model
		linearScore=linear_Regression_Model(X_train, Y_train, X_test, Y_test)
		print('Linear Score=', linearScore)

		#Lasso Regression Model
		lassoScore=lasso_Regression_Model(X_train, Y_train, X_test, Y_test)
		print('Lasso Score=', lassoScore)

		#Polynomial Regression Model 
		polyScore=polynomial_Regression(X_train, Y_train, X_test, Y_test, 2)
		print('Polynomial Score=', polyScore)

