# Netflix User Analytics Assignment

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Q1 Load Dataset
df = pd.read_csv("Dataset 2.csv")

print("First Five Records:")
print(df.head())

# Q2 Number of Rows and Columns
print("\nShape of Dataset:")
print(df.shape)

# Q3 Column Names
print("\nColumn Names:")
print(df.columns)

# Q4 Numerical and Categorical Features
print("\nNumerical Features:")
print(df.select_dtypes(include=['int64','float64']).columns)

print("\nCategorical Features:")
print(df.select_dtypes(include=['object']).columns)

# Q5 Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Q6 Average Age
print("\nAverage Age:", df['Age'].mean())

# Q7 Average Watch Hours Per Week
print("Average Watch Hours:", df['WatchHoursPerWeek'].mean())

# Q8 Average Monthly Spending
print("Average Monthly Spending:", df['MonthlySpend'].mean())

# Q9 Subscription Category Count
print("\nSubscription Type Counts:")
print(df['SubscriptionType'].value_counts())

# Q10 Percentage of Renewed Subscriptions
renewed_percentage = (df['SubscriptionRenewed'] == 'Yes').mean() * 100
print("\nRenewed Percentage:", renewed_percentage)

# Q11 Convert Categorical Features into Numerical Form
encoder = LabelEncoder()

for col in ['Gender', 'SubscriptionType', 'FavoriteGenre', 'SubscriptionRenewed']:
    df[col] = encoder.fit_transform(df[col])

# Q12 Define X and y
X = df.drop(['SubscriptionRenewed', 'UserID'], axis=1)
y = df['SubscriptionRenewed']

# Q13 Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Q14 Train Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Q15 Accuracy
y_pred_dt = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)

print("\nDecision Tree Accuracy:", dt_accuracy)

# Q16 Confusion Matrix
cm = confusion_matrix(y_test, y_pred_dt)

print("\nConfusion Matrix:")
print(cm)

# Q17 Train KNN (K=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Q18 Compare Accuracy
y_pred_knn = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)

print("\nKNN Accuracy:", knn_accuracy)

if knn_accuracy > dt_accuracy:
    print("KNN performed better.")
else:
    print("Decision Tree performed better.")

# Q19 Linear Regression
X_reg = df.drop(['MonthlySpend', 'UserID'], axis=1)
y_reg = df['MonthlySpend']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train_reg, y_train_reg)

# Q20 Predict Monthly Spending for New User
new_user = [[25, 1, 2, 15, 2, 1, 10, 1]]

prediction = lr.predict(new_user)

print("\nPredicted Monthly Spending:", prediction[0])
