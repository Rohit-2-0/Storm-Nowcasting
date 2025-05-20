# ⛈️ Storm Nowcasting with LSTM, Random Forest, and XGBoost

This project builds predictive models to forecast storm occurrences at two time horizons: **1 hour** and **3 hours** in advance. The goal is to support early warning systems using a combination of deep learning (LSTM) and traditional machine learning models (Random Forest, XGBoost).

---

## 📁 Files

- `Storm_Nowcasting_updated_ipynb_...ipynb` – Jupyter Notebook containing full data preprocessing, model training, and evaluation pipeline.
- `train.csv` – Dataset with storm records, used for supervised learning.

---

## 📊 Features Used

The following features are extracted from storm data:

- **lat** – Latitude  
- **lon** – Longitude  
- **intensity** – Storm intensity metric  
- **size** – Storm size  
- **distance** – Distance from target zone  

---

## 🧪 Target Variables

Two separate binary classification targets:

- `Storm_NosyBe_1h` – Predicts whether a storm will occur 1 hour later  
- `Storm_NosyBe_3h` – Predicts whether a storm will occur 3 hours later  

---

## ⚙️ Technologies & Libraries

- `pandas`, `numpy` – Data handling  
- `scikit-learn` – Preprocessing, train-test split, evaluation  
- `tensorflow.keras` – Deep learning (LSTM)  
- `xgboost` – Gradient Boosted Decision Trees  

---

## 🧠 Model Architectures

### 🧬 LSTM Neural Network

Custom LSTM model built with Keras:

```python
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=input_shape))
model.add(LSTM(32, return_sequences=False, activation='tanh'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```
🌲 Random Forest Classifier

Used as a traditional baseline model:

from sklearn.ensemble import RandomForestClassifier

```
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_flat, y_train)
```
⚡ XGBoost Classifier

A powerful gradient boosting model for tabular classification:
```
from xgboost import XGBClassifier

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_flat, y_train)
```
📈 Evaluation Metrics

    Accuracy – For overall performance comparison

    ROC AUC Score – To assess the model's ability to separate classes

    ROC Curves – Plotted for LSTM, Random Forest, and XGBoost

    Probability outputs – From predict_proba or sigmoid activation

    (Optional) Confusion Matrix, Precision, Recall for additional diagnostics
