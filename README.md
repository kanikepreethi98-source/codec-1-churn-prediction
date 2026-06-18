# Customer Churn Prediction

## 📌 Project Overview

Customer Churn Prediction is a machine learning application that predicts whether a customer is likely to leave a service based on historical customer data. By analyzing customer attributes and behavior, the model helps businesses identify at-risk customers and take proactive retention measures.

The project includes a trained classification model, a user-friendly Streamlit web application, and a complete machine learning pipeline for preprocessing, training, evaluation, and prediction.

---

## 🚀 Features

* Predicts whether a customer is likely to churn or stay.
* Interactive Streamlit web application.
* Clean and responsive user interface.
* Data preprocessing and feature engineering.
* Multiple classification algorithms with best model selection.
* Model persistence using Joblib.
* Displays prediction confidence.
* Feature importance analysis for explainability.
* Modular and well-documented codebase.
* Ready for GitHub and Streamlit Community Cloud deployment.

---

## 📂 Project Structure

```text
customer-churn-prediction/
│── app.py
│── train_model.py
│── predict.py
│── requirements.txt
│── README.md
│── .gitignore
│── model.pkl
│── scaler.pkl
│── feature_columns.pkl
│── dataset/
│     └── customer_churn.csv
│── notebooks/
│     └── EDA.ipynb
│── assets/
│     └── logo.png
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Joblib
* Matplotlib
* Plotly (Optional)

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Train the Model

Run the training script:

```bash
python train_model.py
```

This generates:

* `model.pkl`
* `scaler.pkl`
* `feature_columns.pkl`

---

## ▶️ Run the Streamlit App

Start the web application with:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (typically `http://localhost:8501`) in your browser.

---

## 🧠 How It Works

1. Load and preprocess customer data.
2. Encode categorical features and scale numerical values.
3. Train classification models.
4. Save the best-performing model.
5. Accept user input through the Streamlit interface.
6. Predict whether the customer is likely to churn.
7. Display the prediction and confidence score.

---

## 📈 Model Evaluation

The training pipeline reports metrics such as:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC (when applicable)
* Confusion Matrix

These metrics help evaluate how well the model distinguishes customers who are likely to churn from those who are likely to stay.

---

## 🌐 Deploy on Streamlit Community Cloud

1. Push the project to a GitHub repository.
2. Sign in to Streamlit Community Cloud.
3. Create a new app and connect your GitHub repository.
4. Select:

   * Repository: `customer-churn-prediction`
   * Branch: `main`
   * Main file path: `app.py`
5. Deploy the application.

---

## 💡 Future Improvements

* Hyperparameter tuning for improved accuracy.
* SHAP or LIME visualizations for model explainability.
* Support for additional machine learning algorithms.
* Batch prediction using uploaded CSV files.
* Model retraining from the web interface.
* Docker containerization and CI/CD integration.

---

## 🤝 Contributing

Contributions, feature requests, and bug reports are welcome. Feel free to fork the repository, create a new branch, and submit a pull request.

---

## 📄 License

This project is released under the MIT License. You are free to use, modify, and distribute it for educational and commercial purposes.

---

## ⭐ Acknowledgments

* Scikit-learn for machine learning tools.
* Streamlit for rapid web app development.
* The open-source community for datasets and supporting libraries.
