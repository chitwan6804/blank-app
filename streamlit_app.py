import streamlit as st

st.write('Starting the app...')

@st.cache_data
def load_data():
    return pd.read_csv('heart.csv')

data = load_data()
