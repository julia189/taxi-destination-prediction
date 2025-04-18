import dask.array as da
from dask_ml import cluster
import logging
import numpy as np
import pandas as pd
import random 
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from data_cleaning import (
    convert_polyline_to_geojson_format,
    convert_string_to_geojson,
    filter_invalid_trips,
)
from geo_spatial import calculate_polyline_features, extract_lat_lon
from utils import convert_datatypes, extend_timestamps, reduce_high_cardinality, feature_encoding_oh, feature_encoding_oe


class BasePreprocessor():
    def __init__(self, n_min_points: int, id_column: str, model_columns: list):
        self.n_min_points = n_min_points
        self.id_column = id_column
        self.model_columns = model_columns

    def preprocess_data(self, data: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        data = convert_polyline_to_geojson_format(data)
        data = calculate_polyline_features(data)
        data = filter_invalid_trips(data, n_points=self.n_min_points, id_column=self.id_column,
                                    column_coordinate_points='n_coordinate_points')
        data['sequence'] = data['polyline'].apply(lambda row: np.hstack(row))
        if is_training:
            data = data[(data.n_coordinate_points <= data.n_coordinate_points.quantile(0.90))
                 & (data.total_distance_km <= data.total_distance_km.quantile(0.90))]
            data = data[data.missing_data == False]
        model_columns = self.model_columns.append(self.id_column)
        data = data[[model_columns]]
        return data
    
    def encode_features(self, data: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        data = extend_timestamps(data, timestamp_column='timestamp')
        data = reduce_high_cardinality(data, columns=['origin_stand'])
        features_oh = ['call_type','origin_stand_agg','year_month']
        df_fenc_oh = feature_encoding_oh(data, features_oh)
        data = pd.concat([data, df_fenc_oh],axis=1)
        #TODO: fix 
        #if not is_training:
         #   data = data.add_binary_features(data, test_data)
            
        # non_features = [non_feature.lower() for non_feature in non_features]
        return data 

class MulitoutputNNPreprocessor(BasePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess_data(self, data: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        data = super().preprocess_data(data.copy(), is_training=is_training)
        return data
    

class ClusteringPreprocessor(BasePreprocessor):
    def __init__(self,n_clusters: int, pipe: Pipeline, centers, cluster_target,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cluster_target = cluster_target
        self.n_clusters = n_clusters or None
        self.pipe = pipe or None 
        self.centers = centers or None 

    def preprocess_data(self, data: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        data = super().preprocess_data(data.copy(), is_training=is_training)
        data = convert_polyline_to_geojson_format(data=data, name_column='dest_point')
        data = extract_lat_lon(data,'dest_point')
        data = self.preprocess_cluster(data=data.copy(), n_clusters=self.n_clusters, col_name = self.cluster_target, 
                                       centers = self.centers, is_training=is_training)
        return data 

    def preprocess_cluster(self, data: pd.DataFrame, n_clusters: int, col_name: str, centers: pd.DataFrame, is_training: bool):
        random_state=random.seed(5)
        X = np.vstack(X[col_name])
        X_da = da.from_array(X)
        if not n_clusters:
            inertia_per_k = [(k, KMeans(n_clusters=k, init='k-means++', verbose=True, random_state=1).fit(X_da).inertia_)
             for k in range(1000,5000,1000)]
            n_clusters = np.argmax(np.abs(np.gradient(inertia_per_k)))
        
        if is_training:
            pipe = Pipeline([('scaler', StandardScaler()),
                         ('clustering', KMeans(n_clusters=n_clusters,
                                               init = 'k-means++',
                                               n_init=1,
                                               verbose=True,
                                               random_state=random_state
                                               )
                        )])
            cluster_label = pipe.fit_predict(X_da)
            centers_scaled = pipe[1].cluster_centers_
            centers = pipe[0].inverse_transform(centers_scaled)
            centers = pd.DataFrame(centers, columns=['center_lon','center_lat'])
            centers = centers.reset_index()
            assert(centers.index.max() == data.cluster.max())

        else:
            cluster_label = pipe.predict(X_da)
        
        data['cluster_label'] = cluster_label.reshape(-1,1)
        data = pd.merge(data, centers[['index','center_lon','center_lat']], 
                           how='left', 
                           left_on='cluster_label',
                           right_on='index'
                        )
        return data
