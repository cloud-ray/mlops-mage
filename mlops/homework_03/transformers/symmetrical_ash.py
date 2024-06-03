import pandas as pd
from typing import Tuple
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# Set the MLflow tracking URI to your MLflow server
mlflow.set_tracking_uri("http://localhost:5000")

# Set the MLflow experiment
mlflow.set_experiment("yellow-taxi")

@transformer
def train_model(
    df: pd.DataFrame, **kwargs
) -> Tuple[DictVectorizer, LinearRegression]:
    categorical = ['PULocationID', 'DOLocationID']
    df.loc[:, categorical] = df.loc[:, categorical].astype(str)
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    print(f'Feature matrix size: {X_train.shape}')

    target = 'duration'
    y_train = df[target].values
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_train)
    print(f'Train RMSE: {mean_squared_error(y_train, y_pred, squared=False)}')

    print(f'Intercept: {lr.intercept_}')  # Remove the index

    # Log the model with MLflow
    with mlflow.start_run():
        mlflow.sklearn.log_model(lr, "linear_regression_model")
        mlflow.sklearn.log_model(dv, "dict_vectorizer")
        
    return dv, lr
