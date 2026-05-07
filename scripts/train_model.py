import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import spearmanr
import joblib

# -----------------------
# LOAD DATA
# -----------------------
X = np.load("../data/X.npy")
y = np.load("../data/y.npy")

print("Dataset shape:", X.shape)

# -----------------------
# NORMALIZE FEATURES
# -----------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------
# SPLIT DATA
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# -----------------------
# MODEL (XGBoost)
# -----------------------
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# -----------------------
# TRAIN
# -----------------------
print("Training XGBoost model...")
model.fit(X_train, y_train)

# -----------------------
# PREDICT
# -----------------------
preds = model.predict(X_test)

# -----------------------
# EVALUATE
# -----------------------
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)
spearman_corr, _ = spearmanr(y_test, preds)

print("\nResults:")
print("MSE:", mse)
print("R2 Score:", r2)
print("Spearman Correlation:", spearman_corr)

import matplotlib.pyplot as plt

plt.scatter(y_test, preds)
plt.xlabel("True MOS")
plt.ylabel("Predicted MOS")
plt.title("XGBoost Predictions")
plt.show()

joblib.dump(model, "../models/model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")
