import pandas as pd
import numpy as np

def cleaning_dataset(df):

    df.columns = df.columns.str.strip()

    numerical_columns = df.select_dtypes(include=["int64","float64"]).columns
    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

    nominal_columns = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df,columns=nominal_columns)

    print(df.info())
    df.to_csv("housing_clean.csv")

    return df


if __name__ == "__main__":
    df = pd.read_csv("housing.csv")
    cleaning_dataset(df)
