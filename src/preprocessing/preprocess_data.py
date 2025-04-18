from data_cleaning import (
    convert_polyline_to_geojson_format,
    convert_string_to_geojson,
    filter_invalid_trips,
)
from geo_spatial import calculate_polyline_features
import pandas as pd
import numpy as np
from utils import convert_datatypes

class MulitoutputNNPreprocessor():
    def __init__(self, n_min_points: int, id_column: str, model_columns: list):
        self.n_min_points = n_min_points
        self.id_column = id_column
        self.model_columns = model_columns

    def preprocess_data(self, data: pd.DataFrame, is_train_data: bool) -> pd.DataFrame:
        data = convert_polyline_to_geojson_format(data)
        data = calculate_polyline_features(data)
        data = filter_invalid_trips(data, n_points=self.n_min_points, id_column=self.id_column,
                                    column_coordinate_points='n_coordinate_points')
        data['sequence'] = data['polyline'].apply(lambda row: np.hstack(row))
        if is_train_data:
            data = data[(data.n_coordinate_points <= data.n_coordinate_points.quantile(0.90))
                 & (data.total_distance_km <= data.total_distance_km.quantile(0.90))]
            data = data[data.missing_data == False]
        model_columns = self.model_columns.append(self.id_column)
        data = data[[model_columns]]
        return data
    
    