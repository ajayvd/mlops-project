import dagshub
import mlflow
import logging
import os
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


mlflow.set_tracking_uri('https://dagshub.com/ajayvd/mlops-project.mlflow')
dagshub.init(repo_owner='ajayvd', repo_name='mlops-project', mlflow=True)

mlflow.set_experiment('Baseline Logistic Regression Model') # Set the experiment name to credit-fraud-detection



df = pd.read_csv('data.csv') # Load the new balanced dataframe into a pandas dataframe : df

X = df.drop('Class', axis = 1) # Get all the columns except the Class column
y = df['Class'] # Get the Class column

print("Shape of X : ",X.shape) # Display the shape of X
print("Shape of y : ",y.shape) # Display the shape of y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

print("Shape of X_train : ",X_train.shape) # Display the shape of X_train
print("Shape of y_train : ",y_train.shape) # Display the shape of y_train
print("Shape of X_test : ",X_test.shape) # Display the shape of X_test
print("Shape of y_test : ",y_test.shape) # Display the shape of y_test

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logging.info("Starting MLflow run...")

with mlflow.start_run():
    start_time = time.time()
    
    try:
        logging.info("Logging preprocessing parameters...")
        mlflow.log_param("test_size", 0.3)

        logging.info("Initializing Logistic Regression model...")
        model = LogisticRegression(max_iter=1000)  # Increase max_iter to prevent non-convergence issues

        logging.info("Fitting the model...")
        model.fit(X_train, y_train)
        logging.info("Model training complete.")

        logging.info("Logging model parameters...")
        mlflow.log_param("model", "Logistic Regression")

        logging.info("Making predictions...")
        y_pred = model.predict(X_test)

        logging.info("Calculating evaluation metrics...")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logging.info("Logging evaluation metrics...")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        logging.info("Saving and logging the model...")
        mlflow.sklearn.log_model(model, "model")

        # Log execution time
        end_time = time.time()
        logging.info(f"Model training and logging completed in {end_time - start_time:.2f} seconds.")

        # Save and log the notebook
        # notebook_path = "exp1_baseline_model.ipynb"
        # logging.info("Executing Jupyter Notebook. This may take a while...")
        # os.system(f"jupyter nbconvert --to notebook --execute --inplace {notebook_path}")
        # mlflow.log_artifact(notebook_path)

        # logging.info("Notebook execution and logging complete.")

        # Print the results for verification
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"F1 Score: {f1}")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)