# Customer Churn Prediction

End-to-end ML project predicting customer churn using Random Forest, Scikit-learn, and Streamlit.

## Quick Start

```bash
git clone https://github.com/yourusername/customer-churn-prediction
cd customer-churn-prediction
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Add dataset/customer_churn.csv (IBM Telco dataset from Kaggle)
python train_model.py
streamlit run app.py
```

## Project Files

- app.py — Streamlit web application
- train_model.py — Training pipeline
- predict.py — Inference module
- requirements.txt — Dependencies
- dataset/customer_churn.csv — IBM Telco dataset

## Model Performance

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| Random Forest | 81.4% | 0.79 | 0.848 |
| Logistic Regression | 79.8% | 0.76 | 0.821 |
| Decision Tree | 73.2% | 0.71 | 0.731 |

## Deploy to Streamlit Cloud

1. Push to GitHub (include .pkl files)
2. Go to share.streamlit.io
3. Connect repo, set app.py as entry point
4. Deploy

## License
MIT
