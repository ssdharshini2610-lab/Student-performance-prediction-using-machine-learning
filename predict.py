import pandas as pd
from sklearn.linear_model import LinearRegression

# Read dataset
data = pd.read_csv("student_performance_prediction/student_data.csv")

# Features and target
X = data[['attendance', 'assignments', 'previous marks']]
y = data['finalscore']

# Train model
model = LinearRegression()
model.fit(X, y)

# New student data for prediction
new_data = pd.DataFrame({
    'attendance': [90],
    'assignments': [8],
    'previous marks': [80]
})

# Predict
prediction = model.predict(new_data)

print("Predicted Final Score:", round(prediction[0], 2))