import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class LstmModel():
        
    def __init__(self, X_train, X_val, y_train, y_val, feature_columns, sequence_length):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        self.model = None

    def train(self):
        # reshape 
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 
                                                 len(self.feature_columns)))
        self.X_val = np.reshape(self.X_val, (self.X_val.shape[0], self.X_val.shape[1], 
                                             len(self.feature_columns)))

        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 
                                                            len(self.feature_columns))),
                                                            Dropout(0.2),
                                                            LSTM(50),
                                                            Dropout(0.2),
                                                            Dense(1)])

        self.model.compile(optimizer='adam', loss='mse')

        # replace NaNs. Why do we have nans ? Due to standardization ?
        self.X_train = np.nan_to_num(self.X_train)
        self.X_val = np.nan_to_num(self.X_val)

        # Train Model
        self.model.fit(self.X_train, self.y_train, epochs=1, batch_size=32, 
                       validation_data=(self.X_val, self.y_val), verbose=1)
