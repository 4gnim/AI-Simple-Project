# Shoe Review Sentiment Analysis 🏷️📊

Shoe Review Sentiment Analysis is a machine learning project that classifies customer shoe reviews into Positive or Negative sentiment using Natural Language Processing (NLP).

## Features 🚀

- Analyzes customer reviews and predicts sentiment.
- Uses a Naïve Bayes classifier for text classification.
- Implements TF-IDF vectorization for text preprocessing.
- Provides an easy-to-use function for sentiment detection.

## How It Works ⚙️

1. The program takes a dataset of shoe reviews labeled as Positive or Negative.
2. It converts the text data into numerical features using TF-IDF.
3. A Naïve Bayes classifier is trained to classify sentiments.
4. Users can input a new review to get its sentiment prediction.

## Requirements 🛠️

Ensure you have Python and the following libraries installed:

- pandas
- scikit-learn

Install the dependencies using the following command:

```sh
pip install pandas scikit-learn
```

## Usage 🏃‍♂️

Import the function and analyze a review:

```python
from shoe_review_sentiment import detect_sentiment

print(detect_sentiment("Sepatunya sangat nyaman dan keren!"))  # Example usage
```
