
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.models import Model


class BaseModel(Model):
        
    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred 


class LSTMModel(BaseModel):
    def __init__(self, num_timesteps, num_features):
        self.input_layer = Input(shape=(num_timesteps, num_features))
        self.lstm_layer = tf.keras.layers.LSTM(units=64, activation='relu')(self.input_layer)
        self.dense_layer = Dense(units=32, activation='relu')(self.lstm_layer)
        self.output_layer = Dense(units=1)(self.dense_layer)
        self.model = Model(inputs=self.input_layer, outputs=self.output_layer)

    def train (self, X_train, y_train, epochs, batch_size, validation_data, learning_rate=0.001, *args, **kwargs):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss='mse',
                           metrics=[tf.keras.metrics.RootMeanSquaredError()])
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, *args, **kwargs)

class MultioutputModel(BaseModel):

    def __init__(self, num_timesteps, num_features=1):
        self.input_layer = Input(shape=(num_timesteps, num_features))
        self.first_dense = Dense(units=128, activation='relu')(self.input_layer)
        self.first_batch_norm = BatchNormalization(self.first_dense)
        self.second_dense = Dense(units=128, activation='relu')(self.first_batch_norm)
        self.second_batch_norm = BatchNormalization(self.second_dense)
        self.y1_output = Dense(units=1, name='y1_output')(self.second_batch_norm)
        self.third_dense = Dense(units=64, activation='relu')(self.second_batch_norm)

        self.y2_output = Dense(units=1, name='y2_output')(self.third_dense)
        self.model = Model(inputs=self.input_layer, outputs=[self.y1_output, self.y2_output])


    def train(self, X_train, y_train, epochs, batch_size, validation_data, learning_rate=0.0001, *args, **kwargs):
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                            loss={'y1_output': 'mse', 'y2_output': 'mse'},
                            metrics={'y1_output': tf.keras.metrics.RootMeanSquaredError(),
                                     'y2_output': tf.keras.metrics.RootMeanSquaredError()})
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, *args, **kwargs)



class ClusterClassifierModel():
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters

    def train(self, X_train, y_train):
        pass
     
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def eval(self, y_pred, y_test):
        pass
    