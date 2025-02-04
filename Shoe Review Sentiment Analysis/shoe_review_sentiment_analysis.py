import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Data sintetis ulasan sepatu
data = {
    'Review': [
        "Sepatunya sangat nyaman dan keren!",
        "Bahan kurang bagus, agak mengecewakan.",
        "Modelnya stylish, suka sekali!",
        "Kurang nyaman dipakai lama.",
        "Kualitasnya mantap, bakal beli lagi!",
        "Tidak sesuai ekspektasi, solnya tipis.",
        "Sepatu terbaik yang pernah saya beli!",
        "Warnanya pudar setelah beberapa minggu.",
        "Pas di kaki, nyaman dipakai seharian!",
        "Tidak nyaman, ukurannya aneh."
    ],
    'Sentiment': ['Positif', 'Negatif', 'Positif', 'Negatif', 'Positif', 'Negatif', 'Positif', 'Negatif', 'Positif', 'Negatif']
}
df = pd.DataFrame(data)

# Pisahkan data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['Sentiment'], test_size=0.2, random_state=42)

# Pipeline TF-IDF + Naïve Bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Evaluasi model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')

# Fungsi deteksi sentimen
def detect_sentiment(review):
    prediction = model.predict([review])
    return f"Sentimen: {prediction[0]}"

# Contoh penggunaan
print(detect_sentiment("Sepatunya bagus dan nyaman!"))
