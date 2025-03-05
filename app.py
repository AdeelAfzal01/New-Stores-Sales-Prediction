import pandas as pd
import streamlit as st
import numpy as np
import pickle 
import sklearn
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from utils import ImputeMissingValues, FeatureSelector, StandarizeParking, MapColumns, KernelPCAFeatureSelection,\
      OutlierRemover, CustomScaler, FeatureEngineeringTransformer

st.set_page_config(layout="wide")
col_left, right_col = st.columns([1,4])

with col_left:
      input_method = st.radio("Choose Data Input Method:",("Manual Data Entry", "Upload Excel File"))

## Loading the serialized trained model
model = pickle.load(open("Model_Experiment_04.pkl", 'rb'))

## Loading the serialized scikit learn pipeline
pipeline = pickle.load(open("Pipeline_Experiment_04.pkl", 'rb'))


with right_col:
      ## App main page title
      st.title("New Store Sales Prediction App")
      year = '2023'

      if input_method == "Manual Data Entry":
                  
            # Province
            province = st.selectbox('Province', ['Saskatchewan', 'Alberta', 'British Columbia', 'Manitoba', 'Ontario'])

            # 2023 Total Population
            total_pop_1km = st.number_input(f"{year} Total Population 1km", min_value=0.0, format="%.1f")
            total_pop_3km = st.number_input(f"{year} Total Population 3km", min_value=0.0, format="%.1f")
            total_pop_5km = st.number_input(f"{year} Total Population 5km", min_value=0.0, format="%.1f")

            # 2023 Total Population Median Age
            total_pop_med_age_1km = st.number_input(f"{year} Total Population Median Age 1km", min_value=0.0, format="%.1f")
            total_pop_med_age_3km = st.number_input(f"{year} Total Population Median Age 3km", min_value=0.0, format="%.1f")
            total_pop_med_age_5km = st.number_input(f"{year} Total Population Median Age 5km", min_value=0.0, format="%.1f")

            # 2023 Daytime Population
            daytime_pop_1km = st.number_input(f"{year} Daytime Population 1km", min_value=0.0, format="%.1f")
            daytime_pop_3km = st.number_input(f"{year} Daytime Population 3km", min_value=0.0, format="%.1f")
            daytime_pop_5km = st.number_input(f"{year} Daytime Population 5km", min_value=0.0, format="%.1f")

            # 2023 Median Household Income 1km
            median_household_inc_1km = st.number_input(f"{year} Median Household Income 1km", min_value=0.0, format="%.1f")
            median_household_inc_3km = st.number_input(f"{year} Median Household Income 3km", min_value=0.0, format="%.1f")
            median_household_inc_5km = st.number_input(f"{year} Median Household Income 5km", min_value=0.0, format="%.1f")

            # 2023 Occupations Unique to Manufacture and Utilities 1km 
            occupation_manu_utilities_1km = st.number_input(f"{year} Occupations Unique to Manufacture and Utilities 1km", min_value=0.0, format="%.1f")
            occupation_manu_utilities_3km = st.number_input(f"{year} Occupations Unique to Manufacture and Utilities 3km", min_value=0.0, format="%.1f")
            occupation_manu_utilities_5km = st.number_input(f"{year} Occupations Unique to Manufacture and Utilities 5km", min_value=0.0, format="%.1f")

            # 2023 Occupations in Trades, Transport, Operators 1km
            occupation_trades_trans_oper_1km = st.number_input(f"{year} Occupations in Trades, Transport, Operators 1km", min_value=0.0, format="%.1f")
            occupation_trades_trans_oper_3km = st.number_input(f"{year} Occupations in Trades, Transport, Operators 3km", min_value=0.0, format="%.1f")
            occupation_trades_trans_oper_5km = st.number_input(f"{year} Occupations in Trades, Transport, Operators 5km", min_value=0.0, format="%.1f")

            # 2023 Occupation Management 1km
            occupation_management_1km = st.number_input(f"{year} Occupation Management 1km", min_value=0.0, format="%.1f")
            occupation_management_3km = st.number_input(f"{year} Occupation Management 3km", min_value=0.0, format="%.1f")
            occupation_management_5km = st.number_input(f"{year} Occupation Management 5km", min_value=0.0, format="%.1f")

            # 2023 Tobacco Products, Alcoholic Beverages 1km
            tobacco_products_alcohol_bev_1km = st.number_input(f"{year} Tobacco Products, Alcoholic Beverages 1km", min_value=0.0, format="%.1f")
            tobacco_products_alcohol_bev_3km = st.number_input(f"{year} Tobacco Products, Alcoholic Beverages 3km", min_value=0.0, format="%.1f")
            tobacco_products_alcohol_bev_5km = st.number_input(f"{year} Tobacco Products, Alcoholic Beverages 5km", min_value=0.0, format="%.1f")

            # 2023 - Dominant PRIZM Segment Number 1km
            dominant_PRIZM_segment_num_1km = st.number_input(f"{year} Dominant PRIZM Segment Number 1km", min_value=0.0, format="%.1f")
            dominant_PRIZM_segment_num_3km = st.number_input(f"{year} Dominant PRIZM Segment Number 3km", min_value=0.0, format="%.1f")
            dominant_PRIZM_segment_num_5km = st.number_input(f"{year} Dominant PRIZM Segment Number 5km", min_value=0.0, format="%.1f")

            # 2023 Male Population 20 to 24 Years 1km
            male_pop_20_to_24_1km = st.number_input(f"{year} Male Population 20 to 24 Years 1km", min_value=0.0, format="%.1f")
            male_pop_20_to_24_3km = st.number_input(f"{year} Male Population 20 to 24 Years 3km", min_value=0.0, format="%.1f")
            male_pop_20_to_24_5km = st.number_input(f"{year} Male Population 20 to 24 Years 5km", min_value=0.0, format="%.1f")

            # 2023 Male Population 25 to 29 Years 1km
            male_pop_25_to_29_1km = st.number_input(f"{year} Male Population 25 to 29 Years 1km", min_value=0.0, format="%.1f")
            male_pop_25_to_29_3km = st.number_input(f"{year} Male Population 25 to 29 Years 3km", min_value=0.0, format="%.1f")
            male_pop_25_to_29_5km = st.number_input(f"{year} Male Population 25 to 29 Years 5km", min_value=0.0, format="%.1f")

            # 2023 Male Population 25 to 29 Years 1km
            male_pop_30_to_34_1km = st.number_input(f"{year} Male Population 30 to 34 Years 1km", min_value=0.0, format="%.1f")
            male_pop_30_to_34_3km = st.number_input(f"{year} Male Population 30 to 34 Years 3km", min_value=0.0, format="%.1f")
            male_pop_30_to_34_5km = st.number_input(f"{year} Male Population 30 to 34 Years 5km", min_value=0.0, format="%.1f")

            # 2023 Female Population 20 to 24 Years 1km
            female_pop_20_to_24_1km = st.number_input(f"{year} Female Population 20 to 24 Years 1km", min_value=0.0, format="%.1f")
            female_pop_20_to_24_3km = st.number_input(f"{year} Female Population 20 to 24 Years 3km", min_value=0.0, format="%.1f")
            female_pop_20_to_24_5km = st.number_input(f"{year} female Population 20 to 24 Years 5km", min_value=0.0, format="%.1f")

            # 2023 Female Population 25 to 29 Years 1km
            female_pop_25_to_29_1km = st.number_input(f"{year} Female Population 25 to 29 Years 1km", min_value=0.0, format="%.1f")
            female_pop_25_to_29_3km = st.number_input(f"{year} Female Population 25 to 29 Years 3km", min_value=0.0, format="%.1f")
            female_pop_25_to_29_5km = st.number_input(f"{year} Female Population 25 to 29 Years 5km", min_value=0.0, format="%.1f")

            # 2023 Female Population 25 to 29 Years 1km
            female_pop_30_to_34_1km = st.number_input(f"{year} Female Population 30 to 34 Years 1km", min_value=0.0, format="%.1f")
            female_pop_30_to_34_3km = st.number_input(f"{year} Female Population 30 to 34 Years 3km", min_value=0.0, format="%.1f")
            female_pop_30_to_34_5km = st.number_input(f"{year} Female Population 30 to 34 Years 5km", min_value=0.0, format="%.1f")

            # Binary Features
            restaurant = st.number_input('Number of Restaurants', min_value=0)
            stores = st.number_input('Number of Stores', min_value=0)
            banks = st.number_input('Number of Banks', min_value=0)
            offices = st.number_input('Number of Offices', min_value=0)
            crowd_score = st.selectbox('Crowd Score', ['No', 'Yes'])
            visibility_score = st.number_input('Visibility Score', min_value=0.0, format="%.1f")
            grocery_store = st.selectbox('Grocery Stores', ['No', 'Yes'])
            parking = st.selectbox('Parking', ['no parking', 'street', 'parking'])
            gas_station = st.selectbox('Gas Station', ['No', 'Yes'])

            # Competitors
            competitors_1km = st.number_input('Competitors 1km', min_value=0.0, format="%.1f")
            competitors_3km = st.number_input('Competitors 3km', min_value=0.0, format="%.1f")
            # cumulative_competitors_3km = st.number_input('Cumulative Competitors 3km', min_value=0.0, format="%.1f")
            cumulative_competitors_3km = competitors_1km + competitors_3km



            input_df = pd.DataFrame({
                  "Store Internal Name": "NA",
                  "Province" : [province],
                  "2023 Total Population 1km":[total_pop_1km],
                  "2023 Total Population 3km":[total_pop_3km],
                  "2023 Total Population 5km":[total_pop_5km],
                  "2023 Total Population Median Age 1km":[total_pop_med_age_1km],
                  "2023 Total Population Median Age 3km":[total_pop_med_age_3km], 
                  "2023 Total Population Median Age 5km":[total_pop_med_age_5km],
                  "2023 Daytime Population 1km":[daytime_pop_1km],
                  "2023 Daytime Population 3km":[daytime_pop_3km],
                  "2023 Daytime Population 5km":[daytime_pop_5km],
                  "2023 Median Household Income 1km":[median_household_inc_1km],
                  "2023 Median Household Income 3km":[median_household_inc_3km],
                  "2023 Median Household Income 5km":[median_household_inc_5km],
                  "2023 Occupations Unique to Manufacture and Utilities 1km":[occupation_manu_utilities_1km],
                  "2023 Occupations Unique to Manufacture and Utilities 3km":[occupation_manu_utilities_3km],
                  "2023 Occupations Unique to Manufacture and Utilities 5km":[occupation_manu_utilities_5km],
                  "2023 Occupations in Trades, Transport, Operators 1km":[occupation_trades_trans_oper_1km],
                  "2023 Occupations in Trades, Transport, Operators 3km":[occupation_trades_trans_oper_3km],
                  "2023 Occupations in Trades, Transport, Operators 5km":[occupation_trades_trans_oper_5km],	
                  "2023 Occupation Management 1km":[occupation_management_1km],	
                  "2023 Occupation Management 3km":[occupation_management_3km],
                  "2023 Occupation Management 5km":[occupation_management_5km],
                  "2023 Tobacco Products, Alcoholic Beverages 1km":[tobacco_products_alcohol_bev_1km],
                  "2023 Tobacco Products, Alcoholic Beverages 3km":[tobacco_products_alcohol_bev_3km],	
                  "2023 Tobacco Products, Alcoholic Beverages 5km":[tobacco_products_alcohol_bev_5km],	
                  "2023 - Dominant PRIZM Segment Number 1km":[dominant_PRIZM_segment_num_1km],	
                  "2023 - Dominant PRIZM Segment Number 3km":[dominant_PRIZM_segment_num_3km],	
                  "2023 - Dominant PRIZM Segment Number 5km":[dominant_PRIZM_segment_num_5km],
                  "2023 Male Population 20 to 24 Years 1km":[male_pop_20_to_24_1km],	
                  "2023 Male Population 20 to 24 Years 3km":[male_pop_20_to_24_3km],	
                  "2023 Male Population 20 to 24 Years 5km":[male_pop_20_to_24_5km],	
                  "2023 Male Population 25 to 29 Years 1km":[male_pop_25_to_29_1km],	
                  "2023 Male Population 25 to 29 Years 3km":[male_pop_25_to_29_3km],	
                  "2023 Male Population 25 to 29 Years 5km":[male_pop_25_to_29_5km],	
                  "2023 Male Population 30 to 34 Years 1km":[male_pop_30_to_34_1km],	
                  "2023 Male Population 30 to 34 Years 3km":[male_pop_30_to_34_3km],	
                  "2023 Male Population 30 to 34 Years 5km":[male_pop_30_to_34_5km],
                  "2023 Female Population 20 to 24 Years 1km":[female_pop_20_to_24_1km],	
                  "2023 Female Population 20 to 24 Years 3km":[female_pop_20_to_24_3km],	
                  "2023 Female Population 20 to 24 Years 5km":[female_pop_20_to_24_5km],	
                  "2023 Female Population 25 to 29 Years 1km":[female_pop_25_to_29_1km],	
                  "2023 Female Population 25 to 29 Years 3km":[female_pop_25_to_29_3km],	
                  "2023 Female Population 25 to 29 Years 5km":[female_pop_25_to_29_5km],	
                  "2023 Female Population 30 to 34 Years 1km":[female_pop_30_to_34_1km],
                  "2023 Female Population 30 to 34 Years 3km":[female_pop_30_to_34_3km],	
                  "2023 Female Population 30 to 34 Years 5km":[female_pop_30_to_34_5km],	
                  "Restaurant":[restaurant],	
                  "Stores":[stores],	
                  "Banks":[banks],	
                  "Offices":[offices],	
                  "Crowd_score":[crowd_score],
                  "Visibility_score":[visibility_score],
                  "Grocery Stores":[grocery_store], 
                  "Parking":[parking],	
                  "Gas Station":[gas_station], 
                  "1km Competitors":[competitors_1km],	
                  "3km Competitors":[competitors_3km],	
                  "Cumulative 3km Competitors":[cumulative_competitors_3km],
                  '3 Months Cumulative Sales':[0],
                  '6 Months Cumulative Sales':[0], 
                  '9 Months Cumulative Sales':[0], 
                  '12 Months Cumulative Sales':[0]
            })

      elif input_method == "Upload Excel File":
            uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
            pipeline_columns = ['Store Internal Name', 'Province', '2023 Total Population 1km', '2023 Total Population 3km',
                              '2023 Total Population 5km', '2023 Total Population Median Age 1km', '2023 Total Population Median Age 3km',
                              '2023 Total Population Median Age 5km', '2023 Daytime Population 1km', '2023 Daytime Population 3km',
                              '2023 Daytime Population 5km', '2023 Median Household Income 1km', '2023 Median Household Income 3km',
                              '2023 Median Household Income 5km', '2023 Occupations Unique to Manufacture and Utilities 1km',
                              '2023 Occupations Unique to Manufacture and Utilities 3km', '2023 Occupations Unique to Manufacture and Utilities 5km',
                              '2023 Occupations in Trades, Transport, Operators 1km', '2023 Occupations in Trades, Transport, Operators 3km',
                              '2023 Occupations in Trades, Transport, Operators 5km', '2023 Occupation Management 1km', '2023 Occupation Management 3km',
                               '2023 Occupation Management 5km', '2023 Tobacco Products, Alcoholic Beverages 1km', '2023 Tobacco Products, Alcoholic Beverages 3km',
                              '2023 Tobacco Products, Alcoholic Beverages 5km', '2023 - Dominant PRIZM Segment Number 1km', '2023 - Dominant PRIZM Segment Number 3km',
                              '2023 - Dominant PRIZM Segment Number 5km', '2023 Male Population 20 to 24 Years 1km', '2023 Male Population 20 to 24 Years 3km',
                              '2023 Male Population 20 to 24 Years 5km', '2023 Male Population 25 to 29 Years 1km', '2023 Male Population 25 to 29 Years 3km',
                              '2023 Male Population 25 to 29 Years 5km', '2023 Male Population 30 to 34 Years 1km', '2023 Male Population 30 to 34 Years 3km',
                              '2023 Male Population 30 to 34 Years 5km', '2023 Female Population 20 to 24 Years 1km', '2023 Female Population 20 to 24 Years 3km',
                              '2023 Female Population 20 to 24 Years 5km', '2023 Female Population 25 to 29 Years 1km', '2023 Female Population 25 to 29 Years 3km',
                              '2023 Female Population 25 to 29 Years 5km', '2023 Female Population 30 to 34 Years 1km', '2023 Female Population 30 to 34 Years 3km',
                              '2023 Female Population 30 to 34 Years 5km', 'Restaurant', 'Stores', 'Banks', 'Offices', 'Crowd_score', 'Visibility_score',
                              'Grocery Stores', 'Parking', 'Gas Station', '1km Competitors', '3km Competitors', 'Cumulative 3km Competitors',
                              '3 Months Cumulative Sales', '6 Months Cumulative Sales', '9 Months Cumulative Sales', '12 Months Cumulative Sales']
            
            if uploaded_file is not None:
                  input_df = pd.read_excel(uploaded_file, index_col=0)
                  if not 'Store Internal Name' in input_df.columns:
                        input_df['Store Internal Name'] = "NA"

                  for col in pipeline_columns:
                        if col not in input_df.columns:
                              input_df[col] = 0
                  input_df = input_df[pipeline_columns]

            else:
                  st.write("Incorrect File Format")

      if st.button("Predict"):

            # Save the Stores Names
            stores_name = input_df['Store Internal Name']

            # Preprocess the input data
            processed_data = pipeline.transform(input_df)            

            # Dropping target cols
            target_cols = ["3 Months Cumulative Sales", "6 Months Cumulative Sales", "9 Months Cumulative Sales", "12 Months Cumulative Sales", "Store Internal Name"]
            processed_data.drop(target_cols, axis=1, inplace=True)

            # Make prediction
            prediction = model.predict(processed_data)

            # Display the prediction
            # st.header('Prediction')
            if input_df.shape[0]==1:
                  st.write(f'3 Months Cumulative Sales: {prediction[0]:.2f}')

            elif input_df.shape[0]>1:
                  prediction_list = [int(pre) for pre in prediction.tolist()]
                  prediction_df = pd.DataFrame({
                              "Store Name" : stores_name,
                              "3 Months Cumulative Sales" : prediction_list}).sort_values(by='Store Name').reset_index(drop=True)
                  
                  st.subheader("Predictions Table")
                  with st.container(border=True):
                        st.write(prediction_df)

                  # st.divider()
                  # st.balloons()

                  st.subheader("Predictions Line Chart")
                  with st.container(border=True):
                        st.line_chart(prediction_df, x='Store Name', y='3 Months Cumulative Sales', height=500)