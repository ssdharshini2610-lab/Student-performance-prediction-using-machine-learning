import pandas as pd
from sklearn.linear_model import LinearRegression

# Read dataset
data = pd.read_csv("student_performance_prediction/student_data.csv")

# Input features
X = data[['attendance', 'assignments', 'previous marks']]

# Output column
y = data['finalscore']

# Train model
model = LinearRegression()
model.fit(X, y)

print("Model trained successfully!")