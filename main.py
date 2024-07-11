import numpy as np 
import pandas as pd 
import streamlit as st 
from sklearn.model_selection import train_test_split 
from xgboost import XGBRegressor 
from sklearn import metrics 

calories = pd.read_csv('calories.csv') 
exercise_data = pd.read_csv('exercise.csv') 

def calories_burn(model_data):
    calories_data = pd.concat([exercise_data, calories['Calories']],axis=1) 
    calories_data.replace({"Gender":{'male':0,'female':1}},inplace=True) 

    x = calories_data.drop(columns=['User_ID','Calories'],axis=1) 
    y = calories_data['Calories']

    x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=2) 

    model = XGBRegressor(
        max_depth=3,       
        learning_rate=0.1, 
        n_estimators=100   
    ) 
    model.fit(x_train, y_train)
    
    return model.predict(model_data)

st.set_page_config(page_title="🔥 Calories Counter")
st.title('Calories Counter')


Gender = st.selectbox('Gender', ['male', 'female'])
Age = st.number_input('Age')
Height = st.number_input('Height')
Weight = st.number_input('Weight')
Duration = st.number_input('Duration')
Heart_Rate = st.number_input('Heart Rate')
Body_Temp = st.number_input('Body temp')

# Convert Gender to numeric
Gender = 0 if Gender == 'male' else 1

input_data = [[Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp]]

if st.button('Count Calories'):
    prediction = calories_burn(input_data)
    st.success(f"Predicted calories burnt: {prediction[0]:.2f} calories")
