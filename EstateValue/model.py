import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Load dataset
df = pd.read_csv("data/house_pricing.csv")

X = df[["area_sqft", "bedrooms"]]
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========== Linear Regression ==========
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

joblib.dump(linear_model, "house_price_model.pkl")

# ========== Scaling ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== Ridge ==========
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
joblib.dump(ridge_model, "house_price_ridge.pkl")

# ========== Lasso ==========
lasso_model = Lasso(alpha=0.5)
lasso_model.fit(X_train_scaled, y_train)
joblib.dump(lasso_model, "house_price_lasso.pkl")

# ========== Coefficients ==========
coef_df = pd.DataFrame({
    "Linear": linear_model.coef_,
    "Ridge": ridge_model.coef_,
    "Lasso": lasso_model.coef_
}, index=X.columns)

print(coef_df)

# ========== Plot ==========
coef_df.plot(kind="bar", figsize=(8, 5))
plt.title("Linear vs Ridge vs Lasso Coefficient Comparison")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()
plt.show()
input("Press Enter to close the plot...")
