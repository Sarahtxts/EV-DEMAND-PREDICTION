
# ⚡ EV-DEMAND-PREDICTION

This repository contains a project for forecasting the demand for electric vehicles (EVs) using historical data and machine learning models.

---

## 📁 Repository Contents

- **EV_dataset.csv** — Historical EV demand data.
- **ev.ipynb** — Jupyter Notebook for data analysis, preprocessing, model training, and forecasting.
- **forecasting_ev_model.pkl** — Trained machine learning model saved as a pickle file.

---

## 📌 Project Overview

The project aims to predict the demand for electric vehicles by analyzing past data and applying regression-based forecasting models.  
The notebook `ev.ipynb` demonstrates the full pipeline:
- Data loading
- Exploratory Data Analysis
- Feature engineering
- Model training and evaluation
- Forecasting future EV demand
- Saving the trained model for reuse

---

## 🛠 Requirements

- Python 3.x
- Jupyter Notebook

### 📦 Python Libraries
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

(If other libraries like `statsmodels` or `prophet` are used in your notebook, add them accordingly.)

---

## 🚀 How to Use

1. **Clone the repository:**
```bash
git clone https://github.com/Sarahtxts/EV-DEMAND-PREDICTION.git
cd EV-DEMAND-PREDICTION
```

2. **Open the notebook:**
```bash
jupyter notebook ev.ipynb
```

3. **Run all cells** to:
   - Load and explore the dataset
   - Preprocess the data
   - Train the model and evaluate its performance
   - Save and load the forecast model
   - Predict EV demand for future periods

---

## 🧠 Model Usage (in any script)

```python
import pickle

with open('forecasting_ev_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Example: prediction = model.predict(new_data)
```

---

## 🤝 Contributions

Contributions are welcome!  
Fork the repository, make your changes, and submit a pull request.

---

## 📜 License

This project currently has **no license** specified.

---

## 📬 Contact

For queries or feedback, open an issue in this repository  
or contact the author via GitHub [@Sarahtxts](https://github.com/Sarahtxts).
