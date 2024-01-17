# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('advertising.csv')

# Display the first few rows of the dataset
print(data.head())

# Select features and target variable
features = data[['TV', 'Radio', 'Newspaper']]
target = data['Sales']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Visualize the results
plt.scatter(X_test['TV'], y_test, color='black', label='Actual Sales')
plt.scatter(X_test['TV'], predictions, color='blue', label='Predicted Sales')
plt.xlabel('TV Advertising Budget')
plt.ylabel('Sales')
plt.title('Sales Prediction using TV Advertising Budget')
plt.legend()
plt.show()
