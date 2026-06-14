import pandas as pd

# Load dataset
df = pd.read_csv("Netflix_User_Analytics.csv")

# Q1
print(df.head())

# Q2
print("Rows and Columns:", df.shape)

# Q3
print(df.columns)

# Q4
print("Numerical Features:")
print(df.select_dtypes(include=['int64','float64']).columns)

print("Categorical Features:")
print(df.select_dtypes(include=['object']).columns)

# Q5
print(df.isnull().sum())

# Q6
print("Average Age:", df['Age'].mean())

# Q7
print("Average Watch Hours:", df['WatchHoursPerWeek'].mean())

# Q8
print("Average Monthly Spending:", df['MonthlySpend'].mean())

# Q9
print(df['SubscriptionType'].value_counts())

# Q10
renewed_percent = (df['SubscriptionRenewed'].value_counts(normalize=True)['Yes']) * 100
print("Renewed Percentage:", renewed_percent)

#Q11–Q13
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

X = df.drop(['SubscriptionRenewed','UserID'], axis=1)
y = df['SubscriptionRenewed']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#Q14–Q16
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt)

cm = confusion_matrix(y_test, y_pred_dt)
print("Confusion Matrix:")
print(cm)


#Q17–Q18
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", accuracy_knn)

if accuracy_knn > accuracy_dt:
    print("KNN performed better.")
else:
    print("Decision Tree performed better.")
    

#Q19–Q20
from sklearn.linear_model import LinearRegression

X_reg = df.drop(['MonthlySpend','UserID'], axis=1)
y_reg = df['MonthlySpend']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train_reg, y_train_reg)

# Example new user
new_user = [[25,1,2,15,2,1,3,1]]
prediction = lr.predict(new_user)

print("Predicted Monthly Spending:", prediction[0])