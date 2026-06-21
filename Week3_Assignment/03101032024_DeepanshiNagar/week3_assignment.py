import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load Dataset
df = pd.read_csv("agriculture_yield_dataset.csv")

# ---------------- Q1 ----------------
print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 10 Records:")
print(df.head(10))

# ---------------- Q2 ----------------
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# ---------------- Q3 ----------------
print("\nDescriptive Statistics:")
print(df.describe())

print("\nFeature with Highest Mean:")
print(df.mean(numeric_only=True).idxmax())

print("\nFeature with Highest Standard Deviation:")
print(df.std(numeric_only=True).idxmax())

# ---------------- Q4 ----------------
cols = ["rainfall_mm", "temperature_c",
        "fertilizer_kg", "yield_ton_per_hectare"]

for col in cols:
    plt.figure(figsize=(6,4))
    plt.hist(df[col], bins=20)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# ---------------- Q5 ----------------
print("\nCrop Type Frequency:")
print(df["crop_type"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x="crop_type", data=df)
plt.title("Crop Type Count")
plt.show()

print("Most Frequent Crop:")
print(df["crop_type"].value_counts().idxmax())

# ---------------- Q6 ----------------
print("\nSoil Type Frequency:")
print(df["soil_type"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x="soil_type", data=df)
plt.title("Soil Type Count")
plt.show()

print("Most Common Soil Type:")
print(df["soil_type"].value_counts().idxmax())

# ---------------- Q7 ----------------
plt.figure(figsize=(6,4))
plt.hist(df["yield_ton_per_hectare"], bins=20)
plt.title("Yield Distribution")
plt.xlabel("Yield")
plt.ylabel("Frequency")
plt.show()

# ---------------- Q8 ----------------
plt.figure(figsize=(6,4))
plt.scatter(df["rainfall_mm"],
            df["yield_ton_per_hectare"])
plt.xlabel("Rainfall (mm)")
plt.ylabel("Yield")
plt.title("Rainfall vs Yield")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df["fertilizer_kg"],
            df["yield_ton_per_hectare"])
plt.xlabel("Fertilizer (kg)")
plt.ylabel("Yield")
plt.title("Fertilizer vs Yield")
plt.show()

# ---------------- Q9 ----------------
corr_matrix = df.select_dtypes(include=np.number).corr()

print("\nCorrelation Matrix:")
print(corr_matrix)

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix,
            annot=True,
            cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("\nCorrelation with Yield:")
print(
    corr_matrix["yield_ton_per_hectare"]
    .sort_values(ascending=False)
)

# ---------------- Q10 ----------------
crop_yield = df.groupby("crop_type")[
    "yield_ton_per_hectare"
].mean()

soil_yield = df.groupby("soil_type")[
    "yield_ton_per_hectare"
].mean()

print("\nAverage Yield by Crop Type:")
print(crop_yield)

print("\nAverage Yield by Soil Type:")
print(soil_yield)

print("\nHighest Yield Crop:")
print(crop_yield.idxmax())

print("\nHighest Yield Soil:")
print(soil_yield.idxmax())

# ---------------- Q11 ----------------
categorical_cols = ["crop_type", "soil_type"]

df_encoded = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True
)

print("\nEncoded Dataset:")
print(df_encoded.head())

# ---------------- Q12 ----------------
X = df_encoded.drop(
    "yield_ton_per_hectare",
    axis=1
)

y = df_encoded["yield_ton_per_hectare"]

print("\nTarget Variable:")
print("yield_ton_per_hectare")

# ---------------- Q13 ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nX_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)
print("y_train Shape:", y_train.shape)
print("y_test Shape:", y_test.shape)

# ---------------- Q14 ----------------
model = LinearRegression()
model.fit(X_train, y_train)

print("\nIntercept:")
print(model.intercept_)

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nModel Coefficients:")
print(coef_df)

print("\nHighest Positive Coefficient:")
print(
    coef_df.loc[
        coef_df["Coefficient"].idxmax()
    ]
)