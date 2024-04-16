import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from xgboost import XGBRegressor 
import streamlit as st 

calories = pd.read_csv('/Users/jignesh/Desktop/machine learning/project/calories.csv') 
exercise_data = pd.read_csv('/Users/jignesh/Desktop/machine learning/project/exercise.csv') 

# def calories_burn():

#     calories_data = pd.concat([exercise_data, calories['Calories']],axis=1) 

#     calories_data.replace({"Gender":{'male':0,'female':1}},inplace=True) 

#     x=calories_data.drop(columns=['User_ID','Calories'],axis=1) 
#     y=calories_data['Calories']

#     x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=2) 

#     model = XGBRegressor(
#         max_depth=3,       
#         learning_rate=0.1, 
#         n_estimators=100   
#     ) 
#     model.fit(x_train,y_train)
    
#     test_data_prediction = model.predict(x_test)
#     mae = metrics.mean_absolute_error(y_test,test_data_prediction)
#     print("mean absolute error :",mae)

#     return 0

calories_data = pd.concat([exercise_data, calories['Calories']],axis=1) 

calories_data.replace({"Gender":{'male':0,'female':1}},inplace=True) 

x=calories_data.drop(columns=['User_ID','Calories'],axis=1) 
y=calories_data['Calories']

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=2) 

model = XGBRegressor(
    max_depth=3,       
    learning_rate=0.1, 
    n_estimators=100   
) 
model.fit(x_train,y_train)

# test_data_prediction = model.predict(x_test)
# mae = metrics.mean_absolute_error(y_test,test_data_prediction)
# print("mean absolute error :",mae)

st.title('Calories Counter')
# Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp

Gender=st.text_input('Gender ')
Age=st.text_input('Age ')
Height=st.text_input('height ')
Weight=st.text_input('weight ')
Duration=st.text_input('Duration ')
Heart_Rate=st.text_input('Heart Rate ')
Body_Temp=st.text_input('Body temp ')

ans=0

if st.button('Count Calories'):
    # ans = calories_burn([Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp])
    ans = model.predict([Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp])

st.success(ans)