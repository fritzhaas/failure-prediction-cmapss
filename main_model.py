import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.preprocessing import Engine, create_sequences, FeatureEngineering
from src.eval import Eval   
from xgboost import XGBRegressor
from src.models import LstmModel
from src.tools.config import CONFIG


######################################################### Load and Process Data
engine = Engine(CONFIG['train_files'], CONFIG['test_files'])
engine.load_train_data()
engine.load_test_data()
df_train = engine.calculate_rul_piecewise_linear(threshold=120) # perfoms better than linear rul
df_test = engine.data_test

######################################################### do some feature engineering
if CONFIG['reduce_by_not_relevant_sensor_data']:
    df_train = FeatureEngineering.reduce_not_relevant_sensor(df_train)
    df_test = FeatureEngineering.reduce_not_relevant_sensor(df_test)

if CONFIG['use_features']:
    df_train = FeatureEngineering.mean_and_std(df_train)
    df_train = FeatureEngineering.ema(df_train)

    df_test = FeatureEngineering.mean_and_std(df_test)
    df_test = FeatureEngineering.ema(df_test)

    if CONFIG['add_special_features']:
        # not used. Decreases RSME. Probably makes no sense without noise reduction (in advance)
        df_train = FeatureEngineering.delta(df_train)
        df_train = FeatureEngineering.slope(df_train)
        df_train = FeatureEngineering.lag_features(df_train)
        df_train = FeatureEngineering.categorical(df_train)

        df_test = FeatureEngineering.delta(df_test)
        df_test = FeatureEngineering.slope(df_test)
        df_test = FeatureEngineering.lag_features(df_test)
        df_test = FeatureEngineering.categorical(df_test)


# Select only sensors as Features, no operation mode
# keeping prio dataframe not necessary
X_train, feature_columns = FeatureEngineering.select_features(df_train)
y_train = df_train['RUL'].values

######################################################### split the data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#########################################################  standardization
scaler = StandardScaler()
if CONFIG['use_standard_scaler']:
    # xgboost does not require standardization. Also reduces RSME.
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

######################################################### set and train model

if CONFIG['model'] == 'xgBoost':

    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    rmse_val = Eval.mse(y_val, y_val_pred)
    print(f"validation RMSE: {rmse_val}")
    
    ######################################################### predict on test
    df_test_last_cycle = df_test.loc[df_test.groupby("unit_number")["time_in_cycles"].idxmax()]
    X_test_last_cycle = df_test_last_cycle[feature_columns]

    # Standardize using the same scaler
    if CONFIG['use_standard_scaler']:
        X_test_last_cycle = scaler.transform(X_test_last_cycle)

    test_pred = model.predict(X_test_last_cycle)

    ######################################################### RMSE for ground truth
    test_rul_ground_truth = pd.read_csv(CONFIG['RUL_files'][0], header=None).values.flatten() 
    rmse_test = Eval.mse(test_rul_ground_truth, test_pred)
    print(f"test RMSE: {rmse_test}")
    print(test_pred)

    # best RMSE: 15.74
    # hint: vald RMSE is higher. THis shouldn be the case. However: makes sense, because we are 
    # training on all 4 datasets and predicting/testing only the first engine

if CONFIG['model'] == 'lstm':
    # tensorboard would be nice here
    # for lstm we learn on sequences with length <sequence_length>
    X_train, y_train = create_sequences(X_train, y_train, CONFIG['sequence_length'])
    X_val, y_val = create_sequences(X_val, y_val, CONFIG['sequence_length'])

    LSTM_model = LstmModel(X_train, X_val, y_train, y_val, feature_columns, 
                           CONFIG['sequence_length'])
    LSTM_model.train()

    y_val_pred = LSTM_model.model.predict(np.nan_to_num(X_val))
    print(np.isnan(y_val_pred).sum())
    rmse_val = Eval.mse(y_val, y_val_pred)
    print(f"validation RMSE: {rmse_val}")

    ######################################################### predict on test
    test_pred_last = Eval.get_last_sequence(df_test, feature_columns, CONFIG['sequence_length'], 
                                            scaler, CONFIG['use_standard_scaler'])
    test_pred_last = LSTM_model.model.predict(test_pred_last)

    ######################################################### RMSE for ground truth
    test_rul_ground_truth = pd.read_csv(CONFIG['RUL_files'][0], header=None).values.flatten()
    test_pred_last = np.nan_to_num(test_pred_last)

    rmse_test = Eval.mse(test_rul_ground_truth, test_pred_last)
    print(f"test RMSE: {rmse_test}")
    print(test_pred_last)

