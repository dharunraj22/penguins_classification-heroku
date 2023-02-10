import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

st.write("""
         # Penguins Classification App
         """)

st.sidebar.header('User Input Features')
st.sidebar.markdown("""
                    Sidebar Markdown
                    """)
                    
uploaded_file=st.sidebar.file_uploader("Upload your csv file", type=["csv"])

if uploaded_file is not None:
    input_df=pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island=st.sidebar.selectbox('Island',('Biscoe','Dream','Torgensen'))
        sex=st.sidebar.selectbox('Sex',('male','female'))
        bill_length_mm=st.sidebar.slider('Bill length (mm)',32.1,59.6,43.9)
        bill_depth_mm=st.sidebar.slider('Bill depth (mm)',13.1,21.5,17.2)
        flipper_length_mm=st.sidebar.slider('Flipper length (mm)',172.0,231.0,201.0)
        body_mass_g=st.sidebar.slider('Body mass (g)',2700.0,6300.0,4207.0)
        data={'island':island,
              'bill_length_mm':bill_length_mm,
              'bill_depth_mm':bill_depth_mm,
              'flipper_length_mm':flipper_length_mm,
              'body_mass_g':body_mass_g,
              'sex':sex
            }
        features=pd.DataFrame(data, index=[0])
        return features
    
    input_df=user_input_features()
    
penguins=pd.read_csv('penguins.csv')
penguins=penguins.drop(["species"],axis=1)
df=pd.concat([input_df,penguins],axis=0)

encode=['sex','island']
for col in encode:
    dummy=pd.get_dummies(df[col],prefix=col)
    df=pd.concat([df,dummy], axis=1)
    del df[col]
df=df[:1]

st.subheader('User Input Features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write("Currently using input parameters")
    st.write(df)
        
load_model=pickle.load(open('penguins_model.pkl','rb'))
prediction_proba=load_model.predict_proba(df)
prediction=load_model.predict(df)

st.subheader('Prediction')
species=np.array(['Adelie','Chinstrap','Gentoo'])
st.write(species[prediction])  

st.subheader("Prediction Probability")
st.write(prediction_proba)      
