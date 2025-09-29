from .train import TrainConfig, train
from .data_preprocessing import split_data, preprocessing
from .model import create_XGBoost

import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


data_path = "data/historical_flights.csv"

config = TrainConfig(flight_csv = data_path, 
                         model_name = "xg", 
                         model_function = create_XGBoost)
model = train(config)

df = pd.read_csv("data/historical_flights.csv")

df = preprocessing(df)
_x, _y, X_test, y_test = split_data(df)

y_pred = model.predict(X_test)

# Calculate AUC-ROC
auc_score = roc_auc_score(y_test, y_pred)
print(f"AUC-ROC Score: {auc_score:.4f}")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0,1], [0,1], linestyle='--', color='grey')  # baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")
print("ROC curve saved to roc_curve.png")