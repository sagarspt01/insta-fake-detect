import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

def train_baseline():
    # Load processed data
    X_train = pd.read_csv(PROCESSED_DIR/"X_train.csv")
    X_test  = pd.read_csv(PROCESSED_DIR/"X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR/"y_train.csv").squeeze()
    y_test  = pd.read_csv(PROCESSED_DIR/"y_test.csv").squeeze()

    # Balance with SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Train model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_res, y_train_res)

    # Evaluate
    y_pred = model.predict(X_test)

    print("✅ Baseline RandomForest Results:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))

if __name__ == "__main__":
    train_baseline()
