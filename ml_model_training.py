import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

def linear_regression(df):

    if df.columns[0].startswith("Unnamed: 0"):
        df = df.drop(df.columns[0],axis=1)


    linear_regression_model = LinearRegression()

    X = df.drop("median_house_value",axis=1)
    y = df["median_house_value"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

    linear_regression_model.fit(X_train,y_train)
    y_predictions = linear_regression_model.predict(X_test)

    mse = mean_squared_error(y_test,y_predictions)
    mae = mean_absolute_error(y_test,y_predictions)
    r_score = r2_score(y_test,y_predictions)

    print("MSE : ",mse)
    print()
    print("MAE : ",mae)
    print()
    print("R2 Score : ",r_score)

    return (mse,mae,r_score)


def random_forest_regressor(df,num):

    if df.columns[0].startswith("Unnamed: 0"):
        df = df.drop(df.columns[0],axis=1)


    random_forest_regressor_model = RandomForestRegressor()

    X = df.drop("median_house_value",axis=1)
    y = df["median_house_value"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

    random_forest_regressor_model.fit(X_train,y_train)
    y_predictions = random_forest_regressor_model.predict(X_test)

    mse = mean_squared_error(y_test,y_predictions)
    mae = mean_absolute_error(y_test,y_predictions)
    r_score = r2_score(y_test,y_predictions)

    print("MSE : ",mse)
    print()
    print("MAE : ",mae)
    print()
    print("R2 Score : ",r_score)

    importances = random_forest_regressor_model.feature_importances_

    important_dict = {"Features":X.columns,"Importance":importances}

    feature_importances = pd.DataFrame(important_dict).sort_values(by="Importance",ascending=False)

    if num==0:
        return (mse,mae,r_score)
    else:
        return feature_importances


def train_save_random_forest_model(df):

    if df.columns[0].startswith("Unnamed: 0"):
        df = df.drop(df.columns[0],axis=1)

    random_forest_regressor_model = RandomForestRegressor()

    X = df.drop("median_house_value",axis=1)
    y = df["median_house_value"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

    random_forest_regressor_model.fit(X_train,y_train)
    y_predictions = random_forest_regressor_model.predict(X_test)

    r_score = r2_score(y_test,y_predictions)

    with open ("random_forest_regressor.pkl","wb") as f:
        pickle.dump(random_forest_regressor_model,f)

    return r_score


def predict_price(input_list):

    with open ("random_forest_regressor.pkl","rb") as f:
        model = pickle.load(f)

    input_array = np.array(input_list).reshape(1,-1)
    prediction = model.predict(input_array)
    return prediction


    
if __name__ == "__main__":
    df = pd.read_csv("housing_clean.csv")
    linear_regression(df)
    print("----------------------------------")
    random_forest_regressor(df,0)
