# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset into a DataFrame and explore its contents to understand the data structure.
2.Separate the dataset into independent (X) and dependent (Y) variables, and split them into training and testing sets.
3.Create a linear regression model and fit it using the training data.
4.Predict the results for the testing set and plot the training and testing sets with fitted lines.

## Program:

/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SARAVANAN K
RegisterNumber:  25013282
*/
~~~
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks_Scored": [35, 40, 50, 55, 60, 65, 70, 80, 85, 95]
}
df = pd.DataFrame(data)


print("Dataset:\n", df.head())
df


X = df[["Hours_Studied"]]   # Independent variable
y = df["Marks_Scored"]     # Dependent variable


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("\nModel Parameters:")
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])

print("\nEvaluation Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(8,6))
plt.scatter(X, y, label="Actual Data")
plt.plot(X, model.predict(X), linewidth=2, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression: Predicting Marks")
plt.legend()
plt.grid(True)
plt.show()

hours = 7.5
predicted_marks = model.predict([[hours]])
print(f"\nPredicted marks for {hours} hours of study = {predicted_marks[0]:.2f}")
~~~

## Output:
Model Parameters:

Intercept (b0): 28.663793103448278
Slope (b1): 6.379310344827586

Evaluation Metrics:

Mean Squared Error: 1.5922265160523277
R² Score: 0.9968548612028596

<img width="1074" height="804" alt="image" src="https://github.com/user-attachments/assets/fd2f8fe9-5788-456c-a655-8ff96ecdb720" />
<img width="1041" height="64" alt="image" src="https://github.com/user-attachments/assets/2872deb7-a3bc-45ee-afb5-d9a4e60645ed" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
