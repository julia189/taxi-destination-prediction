if __name__ == "__main__":
    import pandas as pd
    from modelling.model import MultioutputModel
    from preprocessing.geo_spatial import calculate_bearing
    
    #X_train = pd.read_csv('data/processed/X_train.csv', header=0, index_col=False).to_numpy()
    #X_test = pd.read_csv('data/processed/X_test.csv', header=0, index_col=False).to_numpy()
    
    #y_train = pd.read_csv('data/processed/y_train.csv', header=0, index_col=False)
    #y_test = pd.read_csv('data/processed/y_test.csv', header=0, index_col=False)

    
    #model = MultioutputModel(num_features=2, num_samples=X_train.shape[0])
    #print(model.model.summary())

    #model.train(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test))



    g_point1 = (51.51154, -0.0029163)
    g_point2 = (35.6709056, 139.7577372)

    print(calculate_bearing(g_point1, g_point2))