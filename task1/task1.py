# Імпортуємо необхідні бібліотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Бібліотеки для машинного навчання
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Імпортуємо Keras та оптимізатори
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import legacy as optimizers_legacy

data = pd.read_csv('data.csv')
print("\nІнформація про датасет:")
print(data.info())

print("\nОписова статистика:")
print(data.describe())

X = data.drop('wage_per_hour', axis=1)
y = data['wage_per_hour']

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42
)

val_size = 0.20 / 0.90
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_size, random_state=42
)

print(f'\nРозмір тренувального набору: {X_train.shape[0]}')
print(f'Розмір валідаційного набору: {X_val.shape[0]}')
print(f'Розмір тестового набору: {X_test.shape[0]}')


scaler_standard = StandardScaler()

X_train_standard = scaler_standard.fit_transform(X_train)
X_val_standard = scaler_standard.transform(X_val)
X_test_standard = scaler_standard.transform(X_test)

input_dim = X_train_standard.shape[1]

def create_model(input_dim, activation='relu', optimizer='adam'):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation=activation))
    model.add(Dropout(0.3))
    model.add(Dense(32, input_dim=input_dim, activation=activation))
    model.add(Dropout(0,3))
    model.add(Dense(16, input_dim=input_dim, activation=activation))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mse', metrics=['mape'])
    return model

optimizers = {
    'SGD': optimizers_legacy.SGD(),
    'RMSprop': optimizers_legacy.RMSprop(),
    'Adam': optimizers_legacy.Adam(),
    'Adadelta': optimizers_legacy.Adadelta(),
    'Adagrad': optimizers_legacy.Adagrad(),
    'Adamax': optimizers_legacy.Adamax(),
    'Nadam': optimizers_legacy.Nadam()
}

activations = ['relu', 'sigmoid', 'tanh']
results = {}

for optimizer_name, optimizer in optimizers.items():
    for activation in activations:
        print(f'\nНавчання з оптимізатором {optimizer_name} та активацією {activation}')
        model = create_model(input_dim, activation=activation, optimizer=optimizer)
        history = model.fit(
            X_train_standard, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val_standard, y_val),
            verbose=0
        )

        val_mape = history.history['mape'][-1]
        results[(optimizer_name, activation)] = val_mape
        print(f'Точність на валідації: {100 - val_mape:.4f}')

results_df = pd.DataFrame(list(results.items()), columns=['Optimizer_Activation', 'Val_mape'])
results_df[['Optimizer', 'Activation']] = pd.DataFrame(results_df['Optimizer_Activation'].tolist(),index=results_df.index)
results_df = results_df.drop('Optimizer_Activation', axis=1)
print(results_df.head())

pivot_table = results_df.pivot(index="Activation", columns="Optimizer", values="Val_mape")
print(pivot_table)

plt.figure(figsize=(14, 10))
sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="YlGnBu")
plt.title('Точність на валідації для різних оптимізаторів та активаційних функцій')
plt.show()

best_row = results_df.loc[results_df['Val_mape'].idxmin()]
best_optimizer = best_row['Optimizer']
best_activation = best_row['Activation']
best_val_mae = best_row['Val_mape']

print(f'\nНайкраща модель: Оптимізатор = {best_optimizer}, Активація = {best_activation}, mape на валідації = {best_val_mae:.4f}')

optimizer_instance = optimizers[best_optimizer]
best_model = create_model(input_dim, best_activation, optimizer_instance)
history_best = best_model.fit(
    X_train_standard, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_standard, y_val),
    verbose=1
)

test_loss, test_mape = best_model.evaluate(X_test_standard, y_test, verbose=0)
print(f'\nmape на тестовому наборі: {test_mape:.4f}')

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(history_best.history['loss'], label='Тренувальні втрати (loss)')
plt.plot(history_best.history['val_loss'], label='Валідаційні втрати (loss)')
plt.title('Втрати під час навчання')
plt.xlabel('Епоха')
plt.ylabel('Втрати (mape)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_best.history['mape'], label='Тренувальна mape')
plt.plot(history_best.history['val_mape'], label='Валідаційна mape')
plt.title('mape під час навчання')
plt.xlabel('Епоха')
plt.ylabel('mape')
plt.legend()

plt.tight_layout()
plt.show()

train_pred = best_model.predict(X_train_standard)
train_mape = mean_absolute_percentage_error(y_train, train_pred)

val_pred = best_model.predict(X_val_standard)
val_mape = mean_absolute_percentage_error(y_val, val_pred)

print(f'\nmape на тренувальному наборі: {train_mape:.4f}')
print(f'mape на валідаційному наборі: {val_mape:.4f}')
