from math import radians

import geojson
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances



def split_lat_lon(data, lon_sequence_column: str, lat_sequence_column: str, sequence_column: str) -> pd.DataFrame:
    modified_data = data.copy()
    modified_data[lon_sequence_column] = modified_data[sequence_column].apply(
        lambda sequence_: np.array(
            [value_[1] for value_ in enumerate(sequence_) if value_[0] % 2 == 0]
        )
    )
    modified_data[lat_sequence_column] = modified_data[sequence_column].apply(
        lambda sequence_: np.array(
            [value_[1] for value_ in enumerate(sequence_) if value_[0] % 2 != 0]
        )
    )
    return modified_data


def create_fix_length_sequences(data, n_limited, start_sequence_column: str, stop_sequence_column: str, sequence_column: str) -> pd.DataFrame:
    modified_data = data.copy()
    modified_data[start_sequence_column] = modified_data[sequence_column].apply(
        lambda sequence: sequence[0 : 2 * n_limited]
    )
    modified_data[stop_sequence_column] = modified_data[sequence_column].apply(
        lambda sequence: sequence[-2 * n_limited :]
    )
    return modified_data


def filter_invalid_trips(data: pd.DataFrame, column_coordinate_points: str, id_column:str, n_points: int) -> pd.DataFrame:
    """
    filters trips with less than n_points coordinate points and takes data sample with longest
    POLYLINE for duplicated TRIP IDs
    """
    modified_data = data.copy()
    modified_data = modified_data[modified_data[(column_coordinate_points)] >= n_points]
    duplicated_ids = modified_data[modified_data[id_column].map(modified_data[id_column].value_counts()) > 1][id_column].unique()
    if len(duplicated_ids) > 0:
        duplicated_data = modified_data[modified_data[id_column].isin(duplicated_ids)]
        valid_data = modified_data[~modified_data[id_column].isin(duplicated_ids)]
        filtered_data = duplicated_data.groupby([id_column]).apply(
            lambda data_chunk: data_chunk[
                data_chunk[column_coordinate_points] == data_chunk[column_coordinate_points].max()
            ]
        )
        modified_data = pd.concat([filtered_data, valid_data], axis=0)
    return modified_data
