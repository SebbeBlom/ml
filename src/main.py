import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
)

train_df = pd.read_csv("data/splits/train.csv")
test_df  = pd.read_csv("data/splits/test_github.csv")

print("=" * 30)
print("TRAIN SET")
print("=" * 30)
print(f"Shape : {train_df.shape}\n")
print(f"Column names:\n{train_df.columns.tolist()}\n")
print(f"Data types:\n{train_df.dtypes}\n")
print(f"First 5 rows:\n{train_df.head()}\n")

print("=" * 30)
print("TARGET")
print("=" * 30)
print(train_df["readmitted"].value_counts())
print(train_df["readmitted"].value_counts(normalize=True))