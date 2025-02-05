import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Buat data sintetis gaya berjalan
data = {
    'Step_Speed': np.random.uniform(0.5, 2.0, 100),  # Kecepatan langkah (m/s)
    'Step_Length': np.random.uniform(0.3, 1.2, 100), # Panjang langkah (m)
}
df = pd.DataFrame(data)

# Label gaya berjalan berdasarkan kecepatan langkah
conditions = [
    (df['Step_Speed'] < 0.9),
    (df['Step_Speed'] >= 0.9) & (df['Step_Speed'] < 1.5),
    (df['Step_Speed'] >= 1.5)
]
categories = ['Lambat', 'Normal', 'Cepat']
df['Gait_Type'] = np.select(conditions, categories)

# Pisahkan fitur dan label
X = df[['Step_Speed', 'Step_Length']]
y = df['Gait_Type']

# Split data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')

# Fungsi deteksi gaya berjalan
def detect_gait(speed, length):
    prediction = model.predict([[speed, length]])
    return f"Gaya berjalan: {prediction[0]}"

# Contoh penggunaan
print(detect_gait(1.2, 0.8))
