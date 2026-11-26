from sklearn.tree import DecisionTreeClassifier

# Dataset
X = [[1, 0], [2, 0], [2, 1], [3, 0], [3, 1],
     [4, 0], [5, 1], [6, 0], [7, 1], [8, 1]]

y = ['Fail', 'Fail', 'Fail', 'Fail', 'Pass',
     'Pass', 'Pass', 'Pass', 'Pass', 'Pass']

# Create model
model = DecisionTreeClassifier(criterion='gini', max_depth=1)

# Train
model.fit(X, y)

# Prediction
print("Prediction:", model.predict([[3, 1]]))
