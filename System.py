import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('homeprices.csv')
print(df.head())

plt.figure(figsize=(8, 5))
plt.scatter(df['area'], df['price'], color='red', marker='+')
plt.title('House Prices vs Area')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price ($)')
plt.grid(True)
plt.show()

X = df[['area']]  
y = df['price']    

model = LinearRegression()
model.fit(X, y)

area_to_predict = 3300
predicted_price = model.predict([[area_to_predict]])
print(f"\nPredicted price for {area_to_predict} sq ft: ${predicted_price[0]:,.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(df['area'], df['price'], color='red', marker='+', label='Actual data')
plt.plot(X, model.predict(X), color='blue', linewidth=1, label='Regression line')
plt.title('Linear Regression: House Prices')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()
