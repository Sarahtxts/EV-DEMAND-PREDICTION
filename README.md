# EV-DEMAND-PREDICTION

This repository contains a project for forecasting the demand for electric vehicles (EVs) using historical data and machine learning models.

---

## Repository Contents

- **EV_dataset.csv**: Dataset containing historical EV demand data.
- **ev.ipynb**: Jupyter Notebook containing the data analysis, preprocessing, model training, and forecasting steps.
- **forecasting_ev_model.pkl**: Trained machine learning model saved as a pickle file.

---

## Project Overview

The project aims to predict the demand for electric vehicles by analyzing past demand data and employing forecasting techniques. The provided notebook (`ev.ipynb`) demonstrates the complete workflow starting from loading the dataset, exploring and preprocessing the data, building a forecasting model, and saving the trained model for future predictions.

---

## Requirements

- Python 3.x
- Jupyter Notebook
- Libraries used (likely to include):
  - pandas
  - numpy
  - scikit-learn
  - matplotlib / seaborn
  - statsmodels / prophet (depending on the forecasting method used)
  - joblib or pickle (for saving/loading models)

# You can install the required packages using pip:
pip install pandas numpy scikit-learn matplotlib seaborn


(Add other packages depending on the notebook details)

---

## How to Use
1. Clone the repository:
   git clone https://github.com/Sarahtxts/EV-DEMAND-PREDICTION.git
   cd EV-DEMAND-PREDICTION
  
2. Open the Jupyter notebook:
   jupyter notebook ev.ipynb

   
3. Run the notebook cells sequentially to:
   - Load and understand the dataset (`EV_dataset.csv`)
   - Explore and preprocess the data
   - Train the forecasting model
   - Save the model into a `.pkl` file
   - Use the model to make predictions on future EV demand

---

## Model Usage

You can load the saved model `forecasting_ev_model.pkl` in your Python scripts to predict EV demand without retraining:
import pickle

with open('forecasting_ev_model.pkl', 'rb') as file:
model = pickle.load(file)


---

## Contributions

Contributions are welcome! If you would like to contribute improvements or new features, please fork the repository and submit a pull request.

---

## License

This project currently has no license specified.

---

## Contact

For questions, you can open an issue in this repository or contact the author.





