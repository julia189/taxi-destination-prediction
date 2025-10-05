import math 

import geojson
import pandas as pd
import logging 
from sklearn.metrics.pairwise import haversine_distances


def convert_string_to_geojson(value: str) -> list:
    json_string = geojson.loads(value)
    return json_string


def convert_polyline_to_geojson_format(
    data: pd.DataFrame, name_column: str
) -> pd.DataFrame:
    current_df = data.copy()
    current_df[name_column] = current_df[name_column].apply(
        lambda row_: convert_string_to_geojson(row_)
    )
    return current_df


def calculate_polyline_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    calculates length of polyline as n_coordinate_points
    calculates total flight time in seconds and minutes as total_flight_time_seconds and total_flight_time_minutes
    """
    modified_data = data.copy()
    modified_data["n_coordinate_points"] = modified_data["polyline"].apply(
        lambda value: len(value[:-1])
    )
    # total flight time
    modified_data["total_flight_time_seconds"] = modified_data.apply(
        lambda row: (row.n_coordinate_points - 1) * 15, axis=1
    )
    modified_data["total_flight_time_minutes"] = (
        modified_data.total_flight_time_seconds / 60
    )
    return modified_data

def extract_lat_lon(data, column_):
    try:
        data[column_+'_lon'] = data[column_].apply(lambda value: value[0])
        data[column_+'_lat'] = data[column_].apply(lambda value: value[1])
    except Exception as e:
        logging.info(e)
        logging.info("Try converting to geojson format")
    return data

def haversine_distance(lat1, lat2, lon1, lon2) -> float:
    """
    :param lat1: Latitude of first point
    :param lat2: Latitude of second point
    :param lon1: Longitude of first point
    :param lon2: Longitude of second point
    :return: Method returns haversine distance between geo coordinates of two points
    """
    # analog to following source -> https://scikit-
    # learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html

    point_1 = [lon1, lat1]
    point_2 = [lon2, lat2]
    point1_in_radians = [math.radians(_) for _ in point_1]
    point2_in_radians = [math.radians(_) for _ in point_2]
    result = haversine_distances([point1_in_radians, point2_in_radians])
    result = result * 6371000 / 1000
    return result[0][1]

def pair_wise_haversine_distance(point_sequence :list) -> list:
    """Haversine distance between pair wise points in a sequence
    """
    distance_array = []
    prev_point = None
    for current_point in point_sequence:
        if prev_point != None:
            current_distance = haversine_distance(lat1=prev_point[0],lat2=current_point[0], lon1=prev_point[1],lon2=current_point[1])
            distance_array = np.append(current_distance, distance_array)
        prev_point = current_point
    return distance_array


def calculate_bearing(point_A, point_B):
    deg2rad = math.pi / 180
    latA = point_A[0] * deg2rad 
    latB = point_B[0] * deg2rad 
    lonA = point_A[1] * deg2rad 
    lonB = point_B[1] * deg2rad 

    delta_ratio = math.log(math.tan(latB/ 2 + math.pi / 4) / math.tan(latA/ 2 + math.pi / 4))
    delta_lon = abs(lonA - lonB)

    delta_lon %= math.pi
    bearing = math.atan2(delta_lon, delta_ratio)/deg2rad
    return bearing



def calculate_total_distance(data) -> pd.DataFrame:
    modified_data = data.copy()
    modified_data["start_point"] = modified_data["polyline"].apply(
        lambda value: value[0]
    )
    modified_data["dest_point"] = modified_data["polyline"].apply(
        lambda value: value[-1]
    )
    # point before last point as last point will be base for label
    modified_data["final_point"] = modified_data["polyline"].apply(
        lambda value: value[-2]
    )
    modified_data["total_distance_km"] = modified_data.apply(
        lambda row: haversine_distance(
            lat1=row.start_point[0],
            lat2=row.final_point[0],
            lon1=row.start_point[1],
            lon2=row.final_point[1],
        ),
        axis=1,
    )
    return modified_data


