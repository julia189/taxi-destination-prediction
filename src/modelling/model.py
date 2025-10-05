
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

class MultioutputModel(Model):

    def __init__(self, num_timesteps, num_features=1):
        self.input_layer = Input(shape=(num_timesteps, num_features))
        self.first_dense = Dense(units=128, activation='relu')(self.input_layer)
        self.second_dense = Dense(units=128, activation='relu')(self.first_dense)
        
        self.y1_output = Dense(units=1, name='y1_output')(self.second_dense)
        self.third_dense = Dense(units=64, activation='relu')(self.second_dense)

        self.y2_output = Dense(units=1, name='y2_output')(self.third_dense)
        self.model = Model(inputs=self.input_layer, outputs=[self.y1_output, self.y2_output])


    def train(self, X_train, y_train, epochs, batch_size, validation_data, learning_rate=0.0001, *args, **kwargs):
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                            loss={'y1_output': 'mse', 'y2_output': 'mse'},
                            metrics={'y1_output': tf.keras.metrics.RootMeanSquaredError(),
                                     'y2_output': tf.keras.metrics.RootMeanSquaredError()})
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, *args, **kwargs)

    
    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred 

    
    def eval(self, y_pred, y_test):
        pass


class ClusterClassifierModel():
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters

    def train(self, X_train, y_train):
        pass

    def predict(self, X_test):
        pass
     
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def eval(self, y_pred, y_test):
        pass
    