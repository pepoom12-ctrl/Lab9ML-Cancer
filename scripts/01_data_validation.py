import mlflow
from sklearn.datasets import load_breast_cancer


def validate_data():
    """
    Loads the breast cancer dataset, performs basic validation checks,
    and logs the results to MLflow.
    """
    # Set the experiment name for this step
    mlflow.set_experiment("Breast Cancer - Data Validation")

    with mlflow.start_run():
        print("Starting data validation run...")
        mlflow.set_tag("ml.step", "data_validation")

        breast_cancer_data = load_breast_cancer(as_frame=True)
        df = breast_cancer_data.frame
        print("Data loaded successfully.")

        num_rows, num_cols = df.shape
        num_classes = df['target'].nunique()
        missing_values = df.isnull().sum().sum()

        class_counts = df['target'].value_counts(normalize=True)
        min_class_ratio = class_counts.min()

        mlflow.log_metric("num_rows", num_rows)
        mlflow.log_metric("num_cols", num_cols)
        mlflow.log_metric("missing_values", missing_values)
        mlflow.log_metric("class_balance", min_class_ratio) 
        mlflow.log_param("num_classes", num_classes)

        # เกณฑ์: ไม่ขาดหาย, มี 2 คลาสพอดี, และคลาสน้อยสุดต้อง >= 20% (0.20)
        validation_status = "Success"
        if missing_values > 0 or num_classes != 2 or min_class_ratio < 0.20:
            validation_status = "Failed"

        mlflow.log_param("validation_status", validation_status)
        
        print(f"{num_rows} rows, {num_cols} columns / Number of classes: {num_classes} / Validation status: {validation_status}")

        if validation_status == "Failed":
            raise SystemExit("Data validation failed — หยุด pipeline ไม่ให้ไปขั้นถัดไป")

        print("Data validation run finished.")


if __name__ == "__main__":
    validate_data()