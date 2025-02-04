import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from DataFrames import create_polars_dataframe

MAX_DEPTH = 6

def main():
    FitRandomForest()



def FitRandomForest():
    plot_data = create_polars_dataframe()

    #features and target data
    #excluded DIA_SQR_SUM from data
    X = plot_data[['TREE_COUNT',
                   'LIVE_CANOPY_CVR_PCT',
                   'TPA_UNADJ',
                   'MAX_HT',
                   'AVG_HT',
                   'QMD',
                   'ELEV',
                   'SLOPE',
                   'ASPECT',
                   'LAT',
                   'LON'
                   ]]
    y = plot_data['PLT_CN']

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5)

    #initialize RandomForestClassifier
    rf_classifier = RandomForestClassifier(n_estimators=100)

    print("Fitting RandomForestClassifier to dataset")
    #fit classifier to data
    rf_classifier.fit(X_train, y_train)

    #make prediction
    y_pred = rf_classifier.predict(X_test)

    #get accuracy and classification reports
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    #print
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", classification_rep)


if __name__ == "__main__":
    main()