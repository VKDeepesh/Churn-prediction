from flask import Flask, render_template, request, url_for, jsonify
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
app = Flask(__name__)

# Load test data and models
X_test, y_test = joblib.load("xgb_test_data.pkl")
ann_model = tf.keras.models.load_model('ann_model.h5')
with open('xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict')
def predict():
    try:
        ann_pred = (ann_model.predict(X_test) > 0.5).astype(int).ravel()
        xgb_pred = xgb_model.predict(X_test)

        ann_acc = accuracy_score(y_test, ann_pred)
        xgb_acc = accuracy_score(y_test, xgb_pred)

        return jsonify({
            "ann_accuracy": round(ann_acc * 100, 2),
            "xgb_accuracy": round(xgb_acc * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/plot/<plot_type>')
def plot(plot_type):
    df = pd.read_csv("customer_churn.csv")
    os.makedirs('static/images', exist_ok=True)
    img_path = "static/images/plot.png"

    if plot_type == "tenure":
        no = df[df.Churn=="No"]['tenure']
        yes = df[df.Churn=="Yes"]['tenure']
        plt.figure(figsize=(10,6))
        plt.hist([yes, no], rwidth=0.9, color=['green','red'],
                 label=['Churn=Yes','Churn=No'])
        plt.title("Customer Churn vs Tenure")
        plt.xlabel("Tenure"); plt.ylabel("Number of Customers")
        plt.legend()

    elif plot_type == "monthly":
        no = df[df.Churn=="No"]['MonthlyCharges']
        yes = df[df.Churn=="Yes"]['MonthlyCharges']
        plt.figure(figsize=(10,6))
        plt.hist([yes, no], rwidth=0.9, color=['green','red'],
                 label=['Churn=Yes','Churn=No'])
        plt.title("Customer Churn vs Monthly Charges")
        plt.xlabel("Monthly Charges"); plt.ylabel("Number of Customers")
        plt.legend()

    elif plot_type == "heatmap":
        # Recreate features for correlation
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = df.dropna(subset=['TotalCharges'])

        multi_no_cols = [
            'OnlineSecurity','OnlineBackup','DeviceProtection',
            'TechSupport','StreamingTV','StreamingMovies'
        ]
        for c in multi_no_cols:
            df[c] = df[c].replace('No internet service', 'No')

        binary_cols = [
            'Partner','Dependents','PhoneService','MultipleLines',
            'OnlineSecurity','OnlineBackup','DeviceProtection',
            'TechSupport','StreamingTV','StreamingMovies',
            'PaperlessBilling'
        ]
        df[binary_cols] = df[binary_cols].applymap(lambda x: 1 if x=='Yes' else 0)
        df['gender'] = df['gender'].map({'Female':0, 'Male':1})

        X_base = df.drop(columns=['customerID', 'Churn', 'InternetService','Contract','PaymentMethod'])
        X_dummies = pd.get_dummies(df[['InternetService','Contract','PaymentMethod']], drop_first=False)
        X_corr = pd.concat([X_base, X_dummies], axis=1)

        corr = X_corr.corr()
        plt.figure(figsize=(20,16))
        sn.heatmap(
            corr, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, cbar_kws={"shrink": .8}
        )
        plt.title("Correlation Heatmap")
        plt.xticks(rotation=90); plt.yticks(rotation=0)

    else:
        plt.figure(figsize=(4,2))
        plt.text(0.5, 0.5, "Unknown plot", ha='center', va='center')

    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()
    return render_template("index.html", image_path=url_for('static', filename='images/plot.png'))

@app.route('/matrix', methods=['POST'])
def matrix():
    model_choice = request.form.get('model', '')

    if model_choice not in ('ann', 'xgb'):
        return render_template(
            "index.html",
            error="Please select a model before generating the confusion matrix.",
            model_choice=model_choice
        )

    if model_choice == 'ann':
        y_pred = (ann_model.predict(X_test) > 0.5).astype(int).ravel()
    else:
        y_pred = xgb_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))
    sn.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_choice.upper()}')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()

    img_path = "static/images/plot.png"
    plt.savefig(img_path)
    plt.close()

    return render_template(
        "index.html",
        image_path=url_for('static', filename='images/plot.png'),
        model_choice=model_choice
    )

if __name__ == '__main__':
    app.run(debug=True)
