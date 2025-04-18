import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np

latitudes = np.array([37.77, 37.78, 37.79])
longitudes = np.array([-122.41, -122.42, -122.43])

coords = np.column_stack((latitudes, longitudes))
scaler = StandardScaler()
normalized_coords = scaler.fit_transform(coords)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
    ])

model.compile(optimizer='adam', loss='mse')

y = np.array([25.0, 30.0, 15.0])

model.fit(normalized_coords, y, epochs=10)

prediction = model.predict(np.array([[35.0, -120.0]])
                           )
print(prediction)