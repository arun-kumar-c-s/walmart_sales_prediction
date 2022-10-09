import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

st.title('Walmart Sales Prediction - Arun Kumar C S')

@st.cache
def predict(x_test):
  clf = pickle.load('lgbm.pkl')
  x_desc=pd.read_csv("sales_item_entry.csv")
  y_pred=clf.predict(x_test)
  y_pred = np.reshape(y_pred, (-1, 28),order = 'F')
  y_pred=pd.DataFrame(y_pred)
  pred_df=pd.concat([x_desc,y_pred],axis=1)
  return pred_df

file = st.file_uploader("Choose a file")
if file is not None:

  df=pd.read_csv(file)
  sale=predict(df)
  


  item_id = st.selectbox('Select The Item!',sale['item_id'].unique()) 

  store_id= st.selectbox('Select The Store!',sale['store_id'].unique())

  if item_id and store_id:
  
    sale=sale[(sale['item_id']==item_id) & (sale['store_id']==store_id)]
    a=sale.drop(['item_id','store_id','Unnamed: 0'],axis=1)
    if st.checkbox("Show Raw Data"):
      st.subheader("28 day Forecast")
      st.write(a)
    else:
      st.subheader("Graphical Representation")
      fig=plt.figure(figsize=(15,7))
  
      [y_axis]=a.values
      x_axis=a.columns.tolist()
      plt.grid()
      plt.plot(x_axis,y_axis)
      s=np.arange(1,29)
      plt.xticks(x_axis,s)
      st.pyplot(fig)

