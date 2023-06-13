import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Загрузка данных из CSV-файла
data = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSrnwmLNVTFvicLvdQAt-gRb4vRaEixCgAxsHln-lFipcVErUhaN5XfPuPWOlBH6dFn7vLSh4uK47Yi/pub?gid=241995259&single=true&output=csv')

# Определение входных и выходных данных
input_data = data[['Pressure', 'Frequency']]
output_data = data[['Electric field strength', 'Magnetic field strength']]

# Нормализация данных
scaler_input = MinMaxScaler()
scaler_output = MinMaxScaler()
input_data = scaler_input.fit_transform(input_data)
output_data = scaler_output.fit_transform(output_data)

test_features = input_data
test_labels = output_data


# Разделение данных на тренировочный и тестовый наборы
train_input, test_input, train_output, test_output = train_test_split(input_data, output_data, test_size=0.2)


# Создание модели нейронной сети
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2)
])

# Компиляция модели
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
history = model.fit(train_input, train_output, epochs=100, batch_size=32, validation_data=(test_features, test_labels))

# Сохранение весов модели
model.save_weights('model_weights.h5')

# Сохранение графика ошибки на тестовой выборке
# Сохранение графика ошибки на тестовой выборке
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.savefig('loss_graph.png')

loss = model.evaluate(test_input, test_output)
print('Test loss:', loss)