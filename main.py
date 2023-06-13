import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Загрузка модели нейронной сети
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(2)
])
model.load_weights('model_weights.h5')

# Функция для построения графиков
def plot_graph(x, y, title):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel('Parameter')
    plt.ylabel('Prediction')
    plt.show()

# Ввод пользовательских данных
selected_param = input("Выберите параметр (Pressure/Frequency): ")
start_range = float(input("Введите начальное значение диапазона выбранного параметра: "))
end_range = float(input("Введите конечное значение диапазона выбранного параметра: "))
other_param = float(input("Введите значение другого параметра: "))

# Создание массива значений выбранного параметра
x_values = np.linspace(start_range, end_range, 100)

# Создание массива значений другого параметра
if selected_param.lower() == 'pressure':
    other_values = np.full_like(x_values, other_param)
elif selected_param.lower() == 'frequency':
    other_values = np.full_like(x_values, other_param)
else:
    print("Некорректный выбор параметра.")
    exit()

# Подготовка входных данных для модели
input_data = np.column_stack((x_values, other_values))

# Получение предсказаний нейронной сети
predictions = model.predict(input_data)

# Разделение предсказаний на отдельные значения
electric_field_predictions = predictions[:, 0]
magnetic_field_predictions = predictions[:, 1]

# Построение графиков предсказаний
plot_graph(x_values, electric_field_predictions, 'Electric Field Strength')
plot_graph(x_values, magnetic_field_predictions, 'Magnetic Field Strength')
