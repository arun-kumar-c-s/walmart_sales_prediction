from calendar import calendar
from copyreg import pickle
import pandas as pd
from tqdm import tqdm 
from downcast import reduce
import numpy as np

from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.naive import NaiveForecaster
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

feature_names = ['wday', 'month', 'year', 'snap_CA', 'snap_TX', 'snap_WI', 'sell_price',
       'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8',
       'lag_9', 'lag_10', 'lag_11', 'lag_12', 'lag_13', 'lag_14', 'lag_15',
       'lag_16', 'lag_17', 'lag_18', 'lag_19', 'lag_20', 'lag_21', 'lag_22',
       'lag_23', 'lag_24', 'lag_25', 'lag_26', 'lag_27', 'lag_28',
       'event_name_1_encoded', 'event_type_1_encoded', 'event_name_2_encoded',
       'event_type_2_encoded', 'id_encoded', 'item_id_encoded',
       'dept_id_encoded', 'cat_id_encoded', 'store_id_encoded',
       'state_id_encoded']

def featurize(product_id, cat_id, dept_id, state_id):
    calendar = pd.read_csv('calendar.csv')
    calendar = calendar.fillna('RegularDay')
    label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
    calendar = calendar.drop(['date', 'weekday', 'month', 'year'], axis=1)
    prices = pd.read_csv('sell_prices.csv')
    prices = prices[prices.wm_yr_wk == prices.wm_yr_wk.max()]
    prices = prices.drop(['wm_yr_wk'], axis=1)
    standard_scaler = pickle.load(open('standard_scaler.pkl', 'rb'))
    

