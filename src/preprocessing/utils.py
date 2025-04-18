import datetime
import json
from datetime import datetime

import numpy as np
import pandas as pd


def print_sample_size(data, add_text=""):
    print(f"{add_text} Size data samples {data.shape[0]}")


def convert_string_to_json(value: str) -> str:
    json_formatted_string = json.loads(value)
    return json_formatted_string


def convert_datatypes(data: pd.DataFrame, columns_datatypes_dict: dict) -> pd.DataFrame:
    for key, value in columns_datatypes_dict.items():
        if value == "int":
            data[key] = data[key].astype(int)
        elif value == "object":
            data[key] = data[key].apply(str)
        elif value == "datetime":
            data[key] = data[key].apply(
                lambda value_unix: datetime.fromtimestamp(value_unix)
            )
    return data


def extend_timestamps(data, timestamp_column:str) -> pd.DataFrame:
    data["timestamp_month"] = pd.to_datetime(data[timestamp_column]).dt.month
    data["timestamp_year"] = pd.to_datetime(data[timestamp_column]).dt.year
    data["year_month"] = (
        data["timestamp_year"].astype(str) + "_" + data["timestamp_month"].astype(str)
    )
    return data


def reduce_high_cardinality(data, columns=[]):
    for column_ in columns:
        vc = data[str(column_)].value_counts()
        median_freq = vc.median()
        majority_freq = vc[vc >= median_freq].index
        data[str(column_) + "_agg"] = data.apply(
            lambda row: row[str(column_)]
            if row[str(column_)] in majority_freq
            else "OTHER",
            axis=1,
        )
        data[str(column_) + "_agg"] = data[str(column_) + "_agg"].astype(str)
    return data


def feature_encoding_oe(data, categories_oe):
    from sklearn.preprocessing import OrdinalEncoder

    data_encoded = pd.DataFrame()
    for attribute_ in categories_oe:
        enc = OrdinalEncoder()
        fenc = enc.fit_transform(X=data[str(attribute_)].values.reshape(-1, 1))
        # print(fenc)
        df_fenc = pd.DataFrame(fenc, columns=[str(attribute_) + "_OE"])
        data_encoded = pd.concat([df_fenc, data_encoded], axis=1)
    return data_encoded


def feature_encoding_oh(data, categories_oh):
    from sklearn.preprocessing import OneHotEncoder

    data_encoded = pd.DataFrame()
    for attribute_ in categories_oh:
        enc = OneHotEncoder()
        fenc = enc.fit_transform(
            X=data[str(attribute_)].values.reshape(-1, 1)
        ).toarray()

        df_fenc = pd.DataFrame(fenc, columns=enc.categories_[0])
        data_encoded = pd.concat([df_fenc, data_encoded], axis=1)
    return data_encoded


def add_binary_features(train_data, test_data):
    missing_features_test = train_data.columns.difference(test_data.columns)
    missing_features_train = test_data.columns.difference(train_data.columns)
    zero_data_test = np.zeros(shape=(test_data.shape[0], len(missing_features_test)))
    dummy_data_test = pd.DataFrame(zero_data_test, columns=missing_features_test)
    zero_data_train = np.zeros(shape=(train_data.shape[0], len(missing_features_train)))
    dummy_data_train = pd.DataFrame(zero_data_train, columns=missing_features_train)

    return pd.concat([test_data, dummy_data_test], axis=1), pd.concat(
        [train_data, dummy_data_train], axis=1
    )
