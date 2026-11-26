from sklearn.tree import DecisionTreeRegressor

# Dataset
X = [[800, 1], [1000, 2], [1200, 2],
     [1400, 3], [1600, 3], [1800, 4]]

y = [20, 25, 30, 35, 40, 45]

# Create model
model = DecisionTreeRegressor(criterion='squared_error', max_depth=1)

# Train
model.fit(X, y)

# Prediction
print("Prediction:", model.predict([[1300, 2]]))
