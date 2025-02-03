import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Buat data sintetis ukuran kaki dan kategori sepatu
data = {
    'Foot_Length': np.random.uniform(22, 30, 100),  # Panjang kaki dalam cm
    'Foot_Width': np.random.uniform(8, 12, 100),   # Lebar kaki dalam cm
    'Shoe_Size': np.random.randint(38, 46, 100)    # Ukuran sepatu Eropa
}
df = pd.DataFrame(data)

# Label kategori sepatu berdasarkan ukuran
conditions = [
    (df['Shoe_Size'] < 40),
    (df['Shoe_Size'] >= 40) & (df['Shoe_Size'] < 43),
    (df['Shoe_Size'] >= 43)
]
categories = ['Small', 'Medium', 'Large']
df['Category'] = np.select(conditions, categories)

# Pisahkan fitur dan label
X = df[['Foot_Length', 'Foot_Width']]
y = df['Category']

# Split data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Fungsi rekomendasi sepatu
def recommend_shoe(length, width):
    prediction = knn.predict([[length, width]])
    return f"Rekomendasi kategori sepatu: {prediction[0]}"

# Contoh penggunaan
print(recommend_shoe(26, 10))
