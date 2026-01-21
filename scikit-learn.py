#git clone https://github.com/Majdawad88/LDR-scikit-learn.git

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Load data
df = pd.read_csv('light_data.csv')
X = df[['Time']] # Feature
y = df['LightLevel'] # Target

# Train Model
model = LinearRegression()
model.fit(X, y)

print(f"Model trained! Slope: {model.coef_[0]}", flush=True)
