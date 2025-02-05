# Gait Recognition AI 🚶‍♂️🔍

Gait Recognition AI is a machine learning project that classifies walking styles (Gait Types) based on step speed and step length using a Random Forest classifier.

## Features 🚀

- Analyzes walking speed and step length to determine gait type.
- Uses a synthetic dataset for training.
- Implements a Random Forest classifier for accurate classification.
- Provides an easy-to-use function for gait recognition.

## How It Works ⚙️

1. The program generates a dataset of step speed and step length.
2. It categorizes walking styles into three types: Slow, Normal, and Fast.
3. A Random Forest classifier is trained to classify gait types.
4. Users can input step measurements to receive a predicted gait type.

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

Import the function and analyze walking style:

```python
from gait_recognition import detect_gait

print(detect_gait(1.2, 0.8))  # Example usage
```
