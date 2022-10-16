import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

######## FEATURIZATION ########

def take_last_n_days_from_sales(sales, n):
    product_info_col = list(sales.keys()[:6])
    days_col = list(sales.keys()[6:])
    last_n_days = days_col[-n:]
    sales = sales[product_info_col + last_n_days].copy()
    return sales

def sales_label_encoder_test(sales, sales_label_encoders):
    sales['id']=sales.id.str.replace('_evaluation', '')
    sales['id']=sales.id.str.replace('_validation', '')
    product_info_col = list(sales.keys()[:6])
    for cols in product_info_col:
        sales[cols] = sales_label_encoders[cols].transform(sales[cols])
    return sales

def calendar_label_encode_test(calendar, calendar_label_encoders):
    calendar_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for cols in calendar_cols:
        calendar[cols] = calendar_label_encoders[cols].transform(calendar[cols])
    return calendar

def price_label_encoder_test(prices, price_label_encoders):
    prices_cols = ['store_id', 'item_id']
    for cols in prices_cols:
        prices[cols] = sales_label_encoders[cols].transform(prices[cols])
    return prices

def melt_sales(sales):
    sales = sales.melt(id_vars=list(sales.keys()[:6]), var_name='d', value_name='sales')
    sales['d'] = sales['d'].str.replace('d_', '').astype(np.int16)
    return sales

def merge_data(melted_sales, calendar, prices):
    #calendar['d'] = calendar['d'].str.replace('d_', '').astype(np.int16)
    melted_sales = melted_sales.merge(calendar, how='left', left_on='d', right_on='d')
    melted_sales = melted_sales.merge(prices, how='left', on=['store_id', 'item_id', 'wm_yr_wk'])
    return melted_sales

def add_lags(df, lag_days):
    for lag_day in lag_days:
        df[f"lag_{lag_day}"] = df.groupby(["id"])["sales"].transform(lambda x: x.shift(lag_day))
    return df

def add_rolling_mean(df, window):
    for window in window:
        df[f"rolling_mean_{window}"] = df.groupby(["id"])["sales"].transform(lambda x: x.shift(28).rolling(window).mean())
    return df

def add_rolling_std(df, window):
    for window in window:
        df[f"rolling_std_{window}"] = df.groupby(["id"])["sales"].transform(lambda x: x.shift(28).rolling(window).std())
    return df

def add_price_change(featurized_df):
    featurized_df['weekly_price_change'] = featurized_df['sell_price'].shift(7)
    return featurized_df

def featurize(sales, calendar, prices, sales_label_encoders, calendar_label_encoders, item_id, store_id):
    sales.id = sales.id.str.replace('_validation', '')
    sales.id = sales.id.str.replace('_evaluation', '')
    sales = sales[(sales['item_id']==item_id) & (sales['store_id']==store_id)]
    print(sales)
    sales = take_last_n_days_from_sales(sales, 60)

    # Downcast df for less memory usage
    sales = sales_label_encoder_test(sales, sales_label_encoders)
    melted_sales = melt_sales(sales)

    calendar = calendar.fillna('RegularDay')
    calendar['d'] = calendar['d'].str.replace('d_', '').astype(np.int16)
    calendar.drop(['weekday', 'date'], axis=1, inplace=True)
    #calendar = calendar.drop(['date','wm_yr_wk', 'weekday', 'wday'], axis=1)
    calendar = calendar_label_encode_test(calendar, calendar_label_encoders)
    prices = price_label_encoder_test(prices, sales_label_encoders)
    featurized_df = merge_data(melted_sales, calendar, prices)

    featurized_df = add_lags(featurized_df, range(1,30))
    #featurized_df = add_lags(featurized_df, [35, 42, 49, 56])
    featurized_df = add_rolling_mean(featurized_df, [7, 14, 21, 28, 30])
    featurized_df = add_rolling_std(featurized_df, [7, 14, 21, 28, 30])
    featurized_df = add_price_change(featurized_df)
    featurized_df.dropna(inplace=True)
    get_forecasting_x = lambda x: x[x.d == x.d.max()]
    featurized_df = get_forecasting_x(featurized_df)
    return featurized_df

#### Load Pickle Files ####

F = {}
for f in range(1, 29):
    F[f] = pickle.load(open(f'final_models/F{f}.pkl', 'rb'))

sales_label_encoders = pickle.load(open('pickled_files/sales_label_encoders.pkl', 'rb'))
calendar_label_encoders = pickle.load(open('pickled_files/calendar_label_encoders.pkl', 'rb'))

st.title('Walmart Sales Prediction - Arun Kumar C S')


#### Streamlit ####

sales = st.file_uploader("Choose a sales file")
calendar = st.file_uploader("Choose a calendar file")
sell_prices = st.file_uploader("Choose a sell prices file")


if (sales is not None) and (calendar is not None) and (sell_prices is not None):
  st.write("Files uploaded successfully")
  st.write("Please wait while we process your files")
  sales = pd.read_csv(sales)
  st.write("File processed successfully")
  item_id = st.selectbox('Select The Item!',sales['item_id'].unique()) 
  store_id= st.selectbox('Select The Store!',sales['store_id'].unique())

  if item_id and store_id:

    if st.button('Start Forecasting'):
      print('Forecasting for item_id: ', item_id, ' and store_id: ', store_id)
      prices = pd.read_csv(sell_prices)
      calendar = pd.read_csv(calendar)

      x = featurize(sales, calendar, prices, sales_label_encoders, calendar_label_encoders, item_id, store_id)

      forecasts = []
      for f in range(1, 29):
          F[f] = pickle.load(open(f'final_models/F{f}.pkl', 'rb'))
          forecasts.append(F[f].predict(x))

      forecasts = pd.DataFrame(forecasts).T
      forecasts.columns = [f'F{i}' for i in range(1, 29)]
      forecasts['id']=x.id.values
      forecasts = forecasts[['id']+ [f'F{i}' for i in range(1, 29)]].sort_values('id')
      rounded_forecast = np.around(forecasts).astype(int)

      st.subheader('Forecasted Sales for next 28 days')
      st.write(rounded_forecast.drop('id', axis=1))
      fig = plt.figure(figsize=(10, 5))
      plt.grid()
      plt.plot(list(range(1,29)), rounded_forecast.drop('id', axis=1).values.flatten())
      st.pyplot(fig)

    # sales=sales[(sales['item_id']==item_id) & (sales['store_id']==store_id)]
    # if st.checkbox("Show Raw Data"):
    #   st.subheader("28 day Forecast")
    #   st.write(a)
    # else:
    #   st.subheader("Graphical Representation")
    #   fig=plt.figure(figsize=(15,7))
  
    #   [y_axis]=a.values
    #   x_axis=a.columns.tolist()
    #   plt.grid()
    #   plt.plot(x_axis,y_axis)
    #   s=np.arange(1,29)
    #   plt.xticks(x_axis,s)
    #   st.pyplot(fig)