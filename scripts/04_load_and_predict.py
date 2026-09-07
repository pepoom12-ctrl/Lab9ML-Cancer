import mlflow
import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_and_predict():
    MODEL_NAME = "cancer-classifier-prod"
    MODEL_ALIAS = "staging"

    print(f"Loading model '{MODEL_NAME}' with alias '@{MODEL_ALIAS}'...")

    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    except mlflow.exceptions.MlflowException as e:
        print(f"Error loading model: {e}")
        return

    # โหลดชุดข้อมูลเพื่อดึงชื่อคลาสคำอ่านและตัวอย่าง 2 ราย
    breast_cancer_data = load_breast_cancer(as_frame=True)
    df = breast_cancer_data.frame
    target_names = breast_cancer_data.target_names  # ['malignant', 'benign']

    # ดึงรายแรกของแต่ละคลาส (target 0 และ 1)
    sample_0 = df[df['target'] == 0].iloc[0]
    sample_1 = df[df['target'] == 1].iloc[0]

    samples_df = pd.DataFrame([sample_0, sample_1])
    X_samples = samples_df.drop(columns=['target'])
    y_true = samples_df['target'].values

    predictions = model.predict(X_samples)

    print("-" * 50)
    for i, (true_val, pred_val) in enumerate(zip(y_true, predictions)):
        true_label = target_names[int(true_val)]
        pred_label = target_names[int(pred_val)]
        status = "Correct" if true_val == pred_val else "Incorrect"
        print(f"Sample {i+1}: True = {true_label}, Predicted = {pred_label} ({status})")
    print("-" * 50)


if __name__ == "__main__":
    load_and_predict()