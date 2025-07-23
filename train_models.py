import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO/WARNINGS

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Load preprocessed data
X_train, X_test, y_train, y_test = joblib.load("dataset.pkl")

# ------------------ Random Forest ------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
print(f"Random Forest Accuracy: {rf_acc:.4f}")
joblib.dump(rf_model, "rf_model.pkl")

# ------------------ XGBoost ------------------
xgb_model = XGBClassifier(eval_metric='logloss', n_estimators=100)
xgb_model.fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
print(f"XGBoost Accuracy: {xgb_acc:.4f}")
joblib.dump(xgb_model, "xgb_model.pkl")

# ------------------ Logistic Regression ------------------
log_model = LogisticRegression(max_iter=300, solver='liblinear')
log_model.fit(X_train, y_train)
log_acc = accuracy_score(y_test, log_model.predict(X_test))
print(f"Logistic Regression Accuracy: {log_acc:.4f}")
joblib.dump(log_model, "log_model.pkl")

# ------------------ Feedforward Neural Network ------------------
ffnn_model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

ffnn_model.compile(optimizer=Adam(learning_rate=0.001),
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

ffnn_model.fit(X_train, y_train,
               epochs=20,
               batch_size=64,
               validation_data=(X_test, y_test),
               callbacks=[early_stop],
               verbose=0)

ffnn_acc = ffnn_model.evaluate(X_test, y_test, verbose=0)[1]
print(f"Feedforward Neural Network Accuracy: {ffnn_acc:.4f}")
ffnn_model.save("ffnn_model.keras", include_optimizer=False)
