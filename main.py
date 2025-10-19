import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# https://www.kaggle.com/datasets/primaryobjects/voicegender
# Veri setini okuma
df = pd.read_csv("voice.csv")

# Etiketleri sayısal hale getirme
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])  # female=0, male=1

# Özellikleri ve hedefi ayırma
X = df.drop('label', axis=1).values
y = df['label'].values

# Veri ölçekleme
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Eğitim ve test bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLP modeli
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Derleme
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Eğitim
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# Sonuç Değerlendirme
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test doğruluğu: {acc*100:.2f}%")

result = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
print(result)

import matplotlib.pyplot as plt

# Eğitim geçmişinden değerleri al
train_loss = history.history['loss']
train_acc = history.history['accuracy']
epochs = range(1, len(train_loss) + 1)

# --- Eğitim Kaybı Grafiği ---
plt.figure(figsize=(8,5))
plt.plot(epochs, train_loss, color='red', linewidth=2)
plt.title('Epoch’a Göre Eğitim Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp (Loss)')
plt.grid(True)
plt.show()

# --- Eğitim Doğruluğu Grafiği ---
plt.figure(figsize=(8,5))
plt.plot(epochs, train_acc, color='blue', linewidth=2)
plt.title('Epoch’a Göre Eğitim Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk (Accuracy)')
plt.grid(True)
plt.show()
