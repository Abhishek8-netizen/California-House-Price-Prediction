import streamlit as st
import pandas as pd
import numpy as np
import io
import data_cleaning
import ml_model_training

st.set_page_config(layout="wide")

st.title("California House Price Prediction Model")

tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["Upload dataset","Preprocessed dataset","Summary Statistics","Training ML models",
                                         "Evaluation","Prediction"])

if "cleaned_dataset" not in st.session_state:
    st.session_state.cleaned_dataset = None


with tab1:
    file = st.file_uploader("Upload CSV file",type="csv")

    if file is None:
        file = "housing.csv"
        df = pd.read_csv("housing.csv")
        st.info("Using default dataset")
    else:
        df = pd.read_csv(file)

    st.dataframe(df)
    st.session_state.cleaned_dataset = data_cleaning.cleaning_dataset(df)

with tab2:
    if st.session_state.cleaned_dataset is not None:
        st.dataframe(st.session_state.cleaned_dataset)

with tab3:
    if st.session_state.cleaned_dataset is not None:

        buffer = io.StringIO()
        st.session_state.cleaned_dataset.info(buf=buffer)
        info_value = buffer.getvalue()

        st.subheader("Summary of data")
        st.code(info_value)

        st.markdown("<br><br>",unsafe_allow_html=True)

        st.subheader("Summary statistics of numerical columns")
        st.write(st.session_state.cleaned_dataset.describe())

with tab4:
    col1,col2,col3 = st.columns([3,1,3])

    if "linear_regression_results" not in st.session_state:
        st.session_state.linear_regression_results = None

    if "random_forest_regression_results" not in st.session_state:
        st.session_state.random_forest_regression_results = None


    with col1:
        clicked = st.button("Train Linear Regression Model")

        if clicked:
            if st.session_state.cleaned_dataset is not None:

                mse,mae,r_score = ml_model_training.linear_regression(st.session_state.cleaned_dataset)
                st.session_state.linear_regression_results = (mse,mae,r_score)

        if st.session_state.linear_regression_results is not None:
            mse,mae,r_score = st.session_state.linear_regression_results

            st.code(f"Mean Squared Error : {mse:.2f}")
            st.markdown("<br>",unsafe_allow_html=True)
            st.code(f"Mean Absolute Error : {mae:.2f}")
            st.markdown("<br>",unsafe_allow_html=True)
            st.code(f"R\u00B2 Score : {r_score:.2f}")
            st.markdown("<br>",unsafe_allow_html=True)

            st.write(f"""The Mean Absolute Error (MAE) represents the average absolute difference between predicted and actual values.
                     On average, the model is off by about ${mae:.2f}.""")

            st.markdown("<br>",unsafe_allow_html=True)

            st.write(f"""The R\u00B2 score indicates that approximately {(r_score * 100):.2f}% of the variance in house prices is explained by
                    the model. While this suggests that the model captures a significant portion of the underlying patterns, there is still room
                    for improvement, likely due to non-linear relationships in the data""")
            

    with col3:
        clicked = st.button("Train Random Forest Regressor Model")

        if clicked:
            if st.session_state.cleaned_dataset is not None:

                mse,mae,r_score = ml_model_training.random_forest_regressor(st.session_state.cleaned_dataset,0)
                st.session_state.random_forest_regression_results = (mse,mae,r_score)

        if st.session_state.random_forest_regression_results is not None:
            mse,mae,r_score = st.session_state.random_forest_regression_results

            st.code(f"Mean Squared Error : {mse:.2f}")
            st.markdown("<br>",unsafe_allow_html=True)
            st.code(f"Mean Absolute Error : {mae:.2f}")
            st.markdown("<br>",unsafe_allow_html=True)
            st.code(f"R2 Score : {r_score:.2f}")

            st.markdown("<br>",unsafe_allow_html=True)
            
            st.write(f"""The Mean Absolute Error (MAE) represents the average absolute difference between predicted and actual values.
                     Here, on average,the model is off by about ${mae:.2f}.""")

            st.markdown("<br>",unsafe_allow_html=True)

            st.write(f"""The R\u00B2 score indicates that approximately {(r_score * 100):.2f}% of the variance in house prices is explained by
                    the model. This suggests that the Random Forest Regressor model captures the complex, non-linear relationships in the
                    data more effectively""")


with tab5:
    if st.session_state.cleaned_dataset is not None:

        clicked = st.button("Generate feature importance")
        
        if clicked:
            feature_importances = ml_model_training.random_forest_regressor(st.session_state.cleaned_dataset,1)
            st.dataframe(feature_importances)


with tab6:
    if st.session_state.cleaned_dataset is not None:

        df = pd.read_csv("housing.csv")

        clicked = st.button("Train and save Random Forest Regressor Model")

        if clicked:
            r_score = ml_model_training.train_save_random_forest_model(st.session_state.cleaned_dataset)
            st.code(f"R\u00B2 Score : {r_score:.2f}")
            st.success("Model trained and saved successfully")


        longitude = st.slider("Select longitude",st.session_state.cleaned_dataset["longitude"].min(),
                              st.session_state.cleaned_dataset["longitude"].max())
        
        st.markdown("<br>",unsafe_allow_html=True)
        latitude = st.slider("Select latitude",st.session_state.cleaned_dataset["latitude"].min(),
                             st.session_state.cleaned_dataset["latitude"].max())

        st.markdown("<br>",unsafe_allow_html=True)
        age = st.slider("Select House Median Age : ",1,100)
        
        st.markdown("<br>",unsafe_allow_html=True)
        rooms = st.number_input("Enter total number of rooms",1,st.session_state.cleaned_dataset["total_rooms"].max())

        st.markdown("<br>",unsafe_allow_html=True)
        bed_rooms = st.number_input("Enter total number bedrooms",1.0,st.session_state.cleaned_dataset["total_bedrooms"].max())

        st.markdown("<br>",unsafe_allow_html=True)
        population = st.number_input("Enter total population",1,st.session_state.cleaned_dataset["population"].max())

        st.markdown("<br>",unsafe_allow_html=True)
        households = st.number_input("Enter total households",1,st.session_state.cleaned_dataset["households"].max())
        
        st.markdown("<br>",unsafe_allow_html=True)
        
        median_income = st.slider("Select Median Income",st.session_state.cleaned_dataset["median_income"].min(),
                  st.session_state.cleaned_dataset["median_income"].max())
        
        st.markdown("<br>",unsafe_allow_html=True)
        ocean_proximity = st.radio("Ocean Proximity",df["ocean_proximity"].unique())

        ocean_categories = [
            "<1H OCEAN",
            "INLAND",
            "ISLAND",
            "NEAR BAY",
            "NEAR OCEAN"]

        encoded_ocean = []

        for cat in ocean_categories:
            if ocean_proximity == cat:
                encoded_ocean.append(1)
            else:
                encoded_ocean.append(0)
            
        st.markdown("<br>",unsafe_allow_html=True)

        input_list = [longitude,latitude,age,rooms,bed_rooms,population,households,median_income,*encoded_ocean]

        predict_button = st.button("Predict House Price")

        if predict_button:
            prediction = float(ml_model_training.predict_price(input_list))
            st.success(f"Predicted House Price : ${prediction}")
                
