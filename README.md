[SUPPLY_CHAIN_README (1).md](https://github.com/user-attachments/files/28039131/SUPPLY_CHAIN_README.1.md)
# 🚚 Supply Chain Delivery Risk Prediction System

> Binary and multi-class delivery risk classification on the DataCo Supply Chain dataset (180,519 orders) using Random Forest — deployed as an interactive **Streamlit web application**.

---

## 📌 Overview

Late deliveries are a major pain point in e-commerce and supply chain management. This project builds an end-to-end ML pipeline to predict delivery risk from shipment configuration features, and deploys the trained model as a live Streamlit app where users can interactively predict delivery outcomes.

---

## 📊 Dataset

| Property | Details |
|---|---|
| Source | DataCo Smart Supply Chain Dataset |
| Total Records | 180,519 orders |
| Features Used | 8 (after leakage removal & preprocessing) |
| Target (EDA) | 3-class: `On-Time` · `At-Risk` · `Delayed` |
| Target (App) | Binary: `On-Time` · `Late Delivery Risk` |

**Key Features:**
`Shipping Mode` · `Market` · `Order Region` · `Customer Segment` · `Order Quantity` · `Sales` · `Discount Rate` · `Scheduled Shipment Days`

---

## 🏗️ ML Pipeline

```
DataCoSupplyChainDataset.csv (180,519 rows)
          │
          ▼
   Data Cleaning & EDA
   (drop leakage cols, encode, handle nulls)
          │
          ▼
  ColumnTransformer Preprocessing
  ├── Numeric: SimpleImputer (median) + StandardScaler
  └── Categorical: SimpleImputer (mode) + OneHotEncoder
          │
          ▼
  RandomForestClassifier (150 trees, random_state=42)
          │
          ▼
  Evaluation (Accuracy · Precision · Recall · F1)
          │
          ▼
  joblib.dump → model.pkl → Streamlit App
```

---

## 📈 Results

### Multi-Class (EDA — 3 classes)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| At Risk | 0.98 | 0.77 | 0.87 |
| Delayed | 0.90 | 0.76 | 0.82 |
| On-Time | 0.77 | 0.96 | 0.86 |
| **Overall Accuracy** | | | **85%** |

### Algorithms Compared

| Model | Accuracy |
|---|---|
| Random Forest (150 trees) | **85%** |
| Gradient Boosting | 66% |
| XGBoost | 66% |

---

## 🖥️ Streamlit App

The trained model is deployed as an interactive web app (`app.py`) where users configure shipment parameters and get an instant delivery risk prediction with confidence score.

**App Inputs:**
- Market, Shipping Mode, Order Region
- Customer Segment, Order Quantity
- Scheduled Shipment Days, Product Price, Discount Rate

**App Output:**
- ✅ On-Time Delivery (with % confidence)
- ⚠️ Late Delivery Risk (with % confidence)
- Probability bar chart for both classes

**Run locally:**
```bash
pip install streamlit scikit-learn pandas numpy joblib
streamlit run app.py
```

---

## 📁 Repository Structure

```
supply-chain-delivery-risk/
│
├── app.py                          # Streamlit web application
├── Data.py                         # Model training pipeline script
├── eda.ipynb                       # EDA, feature engineering & model evaluation
├── model.pkl                       # Serialised trained pipeline
└── README.md
```

> **Dataset:** The `DataCoSupplyChainDataset.csv` is not included due to file size. Download from [Kaggle — DataCo Smart Supply Chain](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis).

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

- **ML:** Scikit-learn (Random Forest, Gradient Boosting, Pipeline, ColumnTransformer)
- **App:** Streamlit
- **Data:** Pandas, NumPy
- **Serialisation:** Joblib

---

## 👤 Author

Developed as part of **Thapar Institute of Engineering & Technology coursework (2025–26)**.

| Member | Roll No. |
|---|---|
| Arnav Gupta| 1024030780 |
| Divyam Mittal | 1024030008 |
| Paarth Mendiratta | 1024030030 |


