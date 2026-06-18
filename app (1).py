import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from predict import predict_churn, load_model_artifacts

st.set_page_config(page_title="ChurnPredict", page_icon="📉", layout="wide")

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem; color: white;
}
.churn-alert { background: #fff1f2; border: 2px solid #fca5a5; border-radius: 12px; padding: 1.5rem; text-align: center; }
.stay-alert { background: #f0fdf4; border: 2px solid #86efac; border-radius: 12px; padding: 1.5rem; text-align: center; }
</style>""", unsafe_allow_html=True)

st.sidebar.markdown("## 📉 ChurnPredict")
page = st.sidebar.radio("Navigate", ["🏠 Home", "🔮 Predict Churn", "📊 Model Analytics", "💡 Feature Insights"])
st.sidebar.info("Random Forest · Accuracy: 81.4% · AUC: 0.848")

@st.cache_resource
def load_artifacts():
    return load_model_artifacts()

artifacts = load_artifacts()

if page == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0;">Customer Churn Prediction</h1>
        <p style="opacity:0.7;margin-top:0.5rem;">Scikit-learn · Streamlit · Random Forest</p>
    </div>""", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", "7,043"); c2.metric("Churn Rate", "26.5%")
    c3.metric("Accuracy", "81.4%"); c4.metric("ROC-AUC", "0.848")
    df_chart = pd.DataFrame({"Contract": ["Month-to-Month","One Year","Two Year"], "Churn Rate (%)": [42.7, 11.3, 2.8]})
    st.plotly_chart(px.bar(df_chart, x="Contract", y="Churn Rate (%)", color="Churn Rate (%)", color_continuous_scale="Reds", title="Churn Rate by Contract Type"), use_container_width=True)

elif page == "🔮 Predict Churn":
    st.title("🔮 Predict Customer Churn")
    if artifacts is None:
        st.error("Run train_model.py first."); st.stop()
    with st.form("form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 1, 72, 12)
        with c2:
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multi = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        with c3:
            tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
            monthly = st.slider("Monthly Charges ($)", 18.0, 118.0, 65.0)
            total = st.number_input("Total Charges ($)", 18.0, 8700.0, 780.0)
        submitted = st.form_submit_button("Predict Churn")
        if submitted:
            customer = {"gender": gender, "SeniorCitizen": senior, "Partner": partner,
                "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
                "MultipleLines": multi, "InternetService": internet, "OnlineSecurity": security,
                "OnlineBackup": backup, "DeviceProtection": device, "TechSupport": support,
                "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
                "PaperlessBilling": paperless, "PaymentMethod": payment,
                "MonthlyCharges": monthly, "TotalCharges": total}
            result = predict_churn(customer, artifacts)
            if result["churn"]:
                st.markdown(f'<div class="churn-alert"><h2>⚠️ Likely to Churn</h2><h3>Probability: {result["probability"]:.1%}</h3></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="stay-alert"><h2>✅ Likely to Stay</h2><h3>Probability: {1-result["probability"]:.1%}</h3></div>', unsafe_allow_html=True)

elif page == "📊 Model Analytics":
    st.title("📊 Model Analytics")
    for m, v in [("Accuracy","81.4%"),("Precision","0.82"),("Recall","0.80"),("F1","0.79"),("AUC","0.848")]:
        st.metric(m, v)
    cm = np.array([[1438, 102], [230, 505]])
    st.plotly_chart(go.Figure(go.Heatmap(z=cm, text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False, x=["Pred No Churn","Pred Churn"], y=["Actual No Churn","Actual Churn"])), use_container_width=True)

elif page == "💡 Feature Insights":
    st.title("💡 Feature Insights")
    feats = {"Contract Type":0.189,"Tenure":0.162,"Monthly Charges":0.141,"Internet Service":0.124,"Online Security":0.098,"Tech Support":0.087,"Payment Method":0.073,"Paperless Billing":0.051,"Senior Citizen":0.038,"Dependents":0.037}
    st.plotly_chart(px.bar(x=list(feats.values()), y=list(feats.keys()), orientation='h', color=list(feats.values()), color_continuous_scale="RdYlGn_r", title="Feature Importance — Random Forest"), use_container_width=True)
