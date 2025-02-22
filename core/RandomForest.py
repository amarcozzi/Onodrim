import polars as pl
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from DataFrames import create_polars_dataframe
from core.DataFrames import create_avg_polars_dataframe

MAX_DEPTH = 6
data_dir = "./data"

def main():
    #fit_random_forest()
    validate_datasets()



def fit_random_forest():
    plot_data = create_polars_dataframe()

    #features and target data
    #excluded DIA_SQR_SUM from data
    X = plot_data[['TREE_COUNT',
                   'LIVE_CANOPY_CVR_PCT',
                   'TPA_UNADJ',
                   'MAX_HT',
                   'AVG_HT',
                   'BALIVE',
                   'ELEV',
                   'SLOPE',
                   'ASPECT',
                   'LAT',
                   'LON'
                   ]]
    y = plot_data['PLT_CN']

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2)



    #initialize RandomForestClassifier
    rf_classifier = RandomForestClassifier(n_estimators=100)

    print("Fitting RandomForestClassifier to dataset")
    #fit classifier to data
    rf_classifier.fit(X_train,y_train)

    #make prediction
    y_pred = rf_classifier.predict(X_test)
    y_pred = pl.from_numpy(y_pred, orient='col')
    y_test = y_test.to_frame()
    #X_test = X_test.to_frame()

    #save our data
    save_data(os.path.join(data_dir, "y_pred.csv"), y_pred)
    save_data(os.path.join(data_dir, "X_test.csv"), X_test)
    save_data(os.path.join(data_dir, "y_test.csv"), y_test)
    save_data(os.path.join(data_dir, "plot_data.csv"), plot_data)

    #validate our results
    validate_results(y_pred, X_test,y_test,plot_data)

def validate_datasets():
    y_pred = pl.read_csv(os.path.join(data_dir, "y_pred.csv"))
    y_test = pl.read_csv(os.path.join(data_dir, "y_test.csv"))
    X_test = pl.read_csv(os.path.join(data_dir, "X_test.csv"))
    plot_data = pl.read_csv(os.path.join(data_dir, "plot_data.csv"))
    validate_results(y_pred, X_test, y_test, plot_data)

def validate_results(y_pred, X_test, y_test, plot_data):
    print("Validating results: ")

    #organize our predictions into a data frame
    #   then connect it to our X testing data
    #   This forms our predictions of test data
    y_pred = y_pred.rename({"column_0": "PLT_CN"})
    pred = pl.concat([y_pred, X_test], how="horizontal")
    pred = pred.sort("PLT_CN")
    n = len(pred) #get our total number of plots
    print(pred)

    TEST = pl.sql(
        query='''
            SELECT 
            pred.PLT_CN,
            pred.BALIVE
            FROM pred
            '''
    ).collect()
    print(TEST)

    #Now grab our original plot data based on predicted PLT_CN
    PLT = pl.sql(
        query='''
            SELECT 
            y_pred.PLT_CN,
            plot_data.BALIVE
            FROM y_pred NATURAL INNER JOIN plot_data
            '''
    ).collect()
    PLT = PLT.sort("PLT_CN")
    print(PLT)




    SUB = TEST - PLT
    print(SUB)

    SUMSQ= pl.sql(
        query='''
            SELECT 
            SUM(POW(SUB.BALIVE,2))
            FROM SUB
            '''
    ).collect()

    #final step of calculating RMSE
    RMSE = np.sqrt(SUMSQ.item()/n)
    print(f"RMSE: {RMSE}")

def save_data(path, frame):
    frame.write_csv(path, separator=",")


if __name__ == "__main__":
    main()