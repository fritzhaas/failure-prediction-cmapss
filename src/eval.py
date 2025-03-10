import numpy as np
from sklearn.metrics import mean_squared_error

class Eval():
    
    @staticmethod
    def mse(ground_truth,prediction):
        return np.sqrt(mean_squared_error(ground_truth, prediction))

    @staticmethod
    def get_last_sequence(df, feature_columns, sequence_length, scaler, use_standard_scaler):
        last_sequences = []
        for engine_id in df["unit_number"].unique():
            engine_data = df[df["unit_number"] == engine_id][feature_columns].values
            
            if len(engine_data) >= sequence_length:
                last_sequence = engine_data[-sequence_length:]  # take last `sequence_length` rows
            else:
                # pad
                pad = np.zeros((sequence_length - len(engine_data), engine_data.shape[1]))
                last_sequence = np.vstack((pad, engine_data))

            last_sequences.append(last_sequence)

        x_test_last_sequences = np.array(last_sequences, dtype=np.float32)

        # standardize
        if use_standard_scaler:
            x_test_last_sequences = np.array([scaler.transform(seq) for seq in x_test_last_sequences])

        # Reshape for LSTM input
        return x_test_last_sequences.reshape(x_test_last_sequences.shape[0], x_test_last_sequences.shape[1], len(feature_columns))

