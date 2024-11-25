import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

#ohttps://github.com/plotly/datasets/blob/master/iris.csv
df = pd.read_csv('data.csv')

print(df.head())

label_encoder = LabelEncoder()
df['Name_encoded'] = label_encoder.fit_transform(df['Name'])


X = df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']].values
y = df['Name_encoded'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_categorical = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y)

print(f"Розмір навчальної вибірки: {X_train.shape}")
print(f"Розмір тестової вибірки: {X_test.shape}")

model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Тестова втрата: {test_loss:.4f}")
print(f"Тестова точність: {test_accuracy:.4f}")

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Навчальна втрата')
plt.plot(history.history['val_loss'], label='Валідаційна втрата')
plt.title('Втрати під час навчання')
plt.xlabel('Епоха')
plt.ylabel('Втрата')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Навчальна точність')
plt.plot(history.history['val_accuracy'], label='Валідаційна точність')
plt.title('Точність під час навчання')
plt.xlabel('Епоха')
plt.ylabel('Точність')
plt.legend()

plt.show()

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Матриця сплутування
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.ylabel('Справжній клас')
plt.xlabel('Передбачений клас')
plt.title('Матриця сплутування')
plt.show()

