import os
import pandas as pd
import numpy as np
import warnings

# Load Data and Preprocessing Functions
class Engine:
    def __init__(self, train_files, test_files):
        self.train_files = train_files
        self.test_files = test_files
        self.data_train = None
        self.data_test = None

    def load_raw_data(self, filename, unit_offset):
        path = os.path.join(os.getcwd(), 'data', filename)
        col_header = ['unit_number', 'time_in_cycles',
                      'operational_setting_1', 'operational_setting_2',
                      'operational_setting_3'] + [f'sensor_measurement_{x}' for x in range(1, 22)]
        try:
            # we need unit-offset. otherwise we group different engines (units) into the same group
            # if we use all 4 datasets alltogether
            df = pd.read_csv(path, sep=r'\s+', header=None, engine='python')
            df.columns = col_header
            df['unit_number'] += unit_offset
            return df
        except FileNotFoundError:
            print(f'{filename} not found!')
            return None

    def load_train_data(self):
        train_dfs = [self.load_raw_data(file, i * 1000) for i, file in enumerate(self.train_files)]
        self.data_train = pd.concat(train_dfs, ignore_index=True)

    def load_test_data(self):
        test_dfs = [self.load_raw_data(file, i * 1000) for i, file in enumerate(self.test_files)]
        self.data_test = pd.concat(test_dfs, ignore_index=True)

    def calculate_rul(self):
        if self.data_train is None:
            print('please load data first')
            return None
        max_cycles = self.data_train.groupby('unit_number')['time_in_cycles'].transform('max')
        self.data_train['RUL'] = max_cycles - self.data_train['time_in_cycles']
        return self.data_train
    
    def calculate_rul_piecewise_linear(self, threshold):
        if self.data_train is None:
            print('please load data first')
            return None
        
        max_cycles = self.data_train.groupby('unit_number')['time_in_cycles'].transform('max')
        rul_values = max_cycles - self.data_train['time_in_cycles']
        
        # constant RUL above threshold, linear below
        self.data_train['RUL'] = rul_values
        self.data_train.loc[rul_values > threshold, 'RUL'] = threshold
        
        return self.data_train

class FeatureEngineering():
    
    @staticmethod
    def select_features(df):
        feature_columns = [col for col in df.columns if 'sensor_measurement' in col]
        return df[feature_columns].values, feature_columns

    @staticmethod
    def reduce_not_relevant_sensor(df):
        # include: sensor_measurement_6
        # exclude: sensor_measurement_14 ?
        drop_columns = ['sensor_measurement_1', 'sensor_measurement_5',
                'sensor_measurement_10', 'sensor_measurement_16', 'sensor_measurement_18',
                'sensor_measurement_7'] 
        return df.drop(columns=drop_columns)

    @staticmethod
    def mean_and_std(df, window_size=5):
        sensor_cols = [col for col in df.columns if 'sensor_measurement' in col]
        for col in sensor_cols:
            df[f'{col}_rolling_mean'] = df.groupby('unit_number')[col].transform(lambda x: 
                                                x.rolling(window=window_size, min_periods=1).mean())
            df[f'{col}_rolling_std'] = df.groupby('unit_number')[col].transform(lambda x: 
                                                x.rolling(window=window_size, min_periods=1).std())
        return df
    
    @staticmethod
    def ema(df):
        sensor_cols = [col for col in df.columns if 'sensor_measurement' in col]
        for col in sensor_cols:
            df[f'{col}_ema'] = df.groupby('unit_number')[col].transform(lambda x: x.ewm(span=5)
                                                                        .mean()) 
        return df
    
    @staticmethod
    def categorical(df):
        sensor_cols = [col for col in df.columns if 'sensor_measurement' in col]
        for col in sensor_cols:
            df[f'{col}_min'] = df.groupby('unit_number')[col].transform('min')
            df[f'{col}_max'] = df.groupby('unit_number')[col].transform('max')
            df[f'{col}_range'] = df[f'{col}_max'] - df[f'{col}_min']
        return df
    
    @staticmethod
    def slope(df):
        # use of rolling windows makes probably also sense
        def compute_slope(df, col):
            return df.groupby('unit_number')[col].transform(lambda x: 
                                                            np.polyfit(np.arange(len(x)), x, 1)[0])
        
        sensor_cols = [col for col in df.columns if 'sensor_measurement' in col]
        for col in sensor_cols:
            df[f'{col}_slope'] = compute_slope(df, col)
        return df
    
    def lag_features(df):
        # bad code here ! thus .. we get warnings. 
        # looping over the data is super inefficient here
        warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
        sensor_cols = [col for col in df.columns if 'sensor_measurement' in col]
        for col in sensor_cols:
            for lag in range(1, 4):
                df[f'{col}_lag{lag}'] = df.groupby('unit_number')[col].shift(lag)
        return df
    
    def delta(df):
        # bad code here ! thus .. we get warnings. 
        # looping over the data is super inefficient here
        warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
        sensor_cols = [col for col in df.columns if 'sensor_measurement' in col]
        for col in sensor_cols:
            df[f'{col}_delta'] = df[col] - df.groupby('unit_number')[col].shift(1)
        return df



# Create Sequences
def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i : i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

