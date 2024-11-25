import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import legacy as optimizers_legacy

data = pd.read_csv('data.csv')
print(data.head())
X = data.drop('diabetes', axis=1)
y = data['diabetes']

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)

val_size = 0.20 / 0.90
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=42,
                                                  stratify=y_train_val)

print(f'Розмір тренувального набору: {X_train.shape[0]}')
print(f'Розмір валідаційного набору: {X_val.shape[0]}')
print(f'Розмір тестового набору: {X_test.shape[0]}')

scaler_standard = StandardScaler()
X_train_standard = scaler_standard.fit_transform(X_train)
X_val_standard = scaler_standard.transform(X_val)
X_test_standard = scaler_standard.transform(X_test)

input_dim = X_train.shape[1]


def create_model(input_dim, activation='relu', optimizer='adam'):
    model = Sequential()
    model.add(Dense(16, input_dim=input_dim, activation=activation))
    model.add(Dense(8, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
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

activations = ['relu', 'tanh', 'sigmoid']
results = {}

for optimizer_name, optimizer in optimizers.items():
    for activation in activations:
        print(f'Навчання з оптимізатором {optimizer_name} та активацією {activation}')
        model = create_model(input_dim, activation=activation, optimizer=optimizer)
        history = model.fit(
            X_train_standard, y_train,
            epochs=50,
            batch_size=16,
            validation_data=(X_val_standard, y_val),
            verbose=0
        )
        val_accuracy = history.history['val_accuracy'][-1]
        results[(optimizer_name, activation)] = val_accuracy
        print(f'Точність на валідації: {val_accuracy:.4f}')

results_df = pd.DataFrame(list(results.items()), columns=['Optimizer_Activation', 'Val_Accuracy'])
results_df[['Optimizer', 'Activation']] = pd.DataFrame(results_df['Optimizer_Activation'].tolist(),
                                                       index=results_df.index)
results_df = results_df.drop('Optimizer_Activation', axis=1)

print(results_df.head())

pivot_table = results_df.pivot(index="Activation", columns="Optimizer", values="Val_Accuracy")
print(pivot_table)

plt.figure(figsize=(14, 10))
sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="YlGnBu")
plt.title('Точність на валідації для різних оптимізаторів та активаційних функцій')
plt.show()

best_params = results_df.loc[results_df['Val_Accuracy'].idxmax()]
best_optimizer = best_params['Optimizer']
best_activation = best_params['Activation']
best_val_accuracy = best_params['Val_Accuracy']

print(f'Найкраща модель: Оптимізатор = {best_optimizer}, Активація = {best_activation}, Точність на валідації = {best_val_accuracy:.4f}')

best_model = create_model(input_dim, activation=best_activation, optimizer=best_optimizer)
history = best_model.fit(
    X_train_standard, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_val_standard, y_val),
    verbose=0
)

test_loss, test_accuracy = best_model.evaluate(X_test_standard, y_test, verbose=0)
print(f'Точність на тестовому наборі: {test_accuracy:.4f}')

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Тренувальна точність')
plt.plot(history.history['val_accuracy'], label='Валідаційна точність')
plt.title('Точність під час навчання')
plt.xlabel('Епоха')
plt.ylabel('Точність')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Тренувальні втрати')
plt.plot(history.history['val_loss'], label='Валідаційні втрати')
plt.title('Втрати під час навчання')
plt.xlabel('Епоха')
plt.ylabel('Втрати')
plt.legend()

plt.show()

train_pred = best_model.predict(X_train_standard)
train_pred = (train_pred > 0.5).astype(int)
train_accuracy = accuracy_score(y_train, train_pred)

val_pred = best_model.predict(X_val_standard)
print(val_pred)
val_pred = (val_pred > 0.5).astype(int)
val_accuracy = accuracy_score(y_val, val_pred)

print(f'Точність на тренувальному наборі: {train_accuracy:.4f}')
print(f'Точність на валідаційному наборі: {val_accuracy:.4f}')
