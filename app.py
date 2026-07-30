import joblib as jb
import numpy as np
from pathlib import Path
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

segment_map={0:'premium customer', 1:'average customer', 2:'wealthy, conservative spender', 3:'low-value customer', 4:'impulsive spender'}

scaler=jb.load(BASE_DIR/'model'/'std_scaler.joblib')
kmeans=jb.load(BASE_DIR/'model'/'kmeans.joblib')

st.title('Customer Segmentation')

income=st.number_input('Enter the annual income(in k$):', min_value=0, step=1)
spend_score=st.number_input('Enter the spending score(1-100):', min_value=1, max_value=100, step=1)

if st.button('Get customer category'):
    dat=np.array([income, spend_score])
    dat=dat.reshape(1,-1)
    pred=segment_map[kmeans.predict(scaler.transform(dat))[0]]

    st.success(f'The customer belongs to the category of "{pred.upper()}"')
