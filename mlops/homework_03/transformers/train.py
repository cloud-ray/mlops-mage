from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from typing import Tuple

import pandas as pd

from mlops.utils.data_preparation.cleaning import clean
from mlops.utils.data_preparation.feature_engineering import combine_features
from mlops.utils.data_preparation.feature_selector import select_features
from mlops.utils.data_preparation.splitters import split_on_value

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def transform(
    df: pd.DataFrame, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, DictVectorizer, LinearRegression]:
    df = pd.DataFrame(df)

    split_on_feature = kwargs.get('split_on_feature')
    split_on_feature_value = kwargs.get('split_on_feature_value')
    target = kwargs.get('target')

    df = clean(df)
    df = combine_features(df)
    df = select_features(df, features=[split_on_feature, target])

    df_train, df_val = split_on_value(
        df,
        split_on_feature,
        split_on_feature_value,
    )

    # Fit a dict vectorizer
    dv = DictVectorizer()
    pickup_dicts = df_train['PULocationID'].to_dict(orient='records')
    dropoff_dicts = df_train['DOLocationID'].to_dict(orient='records')
    X_train_pickup = dv.fit_transform(pickup_dicts)
    X_train_dropoff = dv.transform(dropoff_dicts)
    X_train = pd.concat([X_train_pickup, X_train_dropoff], axis=1)

    # Train a linear regression with default parameters
    y_train = df_train[target].values
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Print the intercept of the model
    print("Intercept:", lr.intercept_)

    return df, df_train, df_val, dv, lr
