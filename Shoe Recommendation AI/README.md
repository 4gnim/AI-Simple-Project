# Shoe Recommendation AI 👟🔍

Shoe Recommendation AI is a simple machine learning project using Python that recommends shoe sizes based on foot length and width. It utilizes the K-Nearest Neighbors (KNN) algorithm to classify shoe sizes into different categories.

## Features 🚀

- Predicts shoe category based on foot measurements.
- Uses a synthetic dataset for training.
- Implements the K-Nearest Neighbors (KNN) algorithm for classification.
- Easy-to-use function for shoe recommendation.

## How It Works ⚙️

1. The program generates a synthetic dataset of foot length, foot width, and shoe sizes.
2. It categorizes shoe sizes into three groups: Small, Medium, and Large.
3. The model is trained using K-Nearest Neighbors (KNN).
4. Users can input foot measurements to receive a recommended shoe category.

## Requirements 🛠️

Ensure you have Python and the following libraries installed:

- pandas
- numpy
- scikit-learn

Install the dependencies using the following command:

```sh
pip install pandas numpy scikit-learn
```

## Usage 🏃‍♂️

Import the function and get a recommendation:

```python
from shoe_recommendation import recommend_shoe

print(recommend_shoe(26, 10))  # Example usage
```
