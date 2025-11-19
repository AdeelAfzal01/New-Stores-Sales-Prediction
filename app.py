import pandas as pd
import streamlit as st
import numpy as np
import pickle 
import sklearn
import altair as alt
import os
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from utils import ImputeMissingValues, FeatureSelector, StandarizeParking, MapColumns, KernelPCAFeatureSelection,\
      OutlierRemover, CustomScaler, FeatureEngineeringTransformer



st.set_page_config(layout="wide")

## Add logo

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image("Logo Updated.png")

st.markdown(f"""
    <div style='text-align: center; margin-bottom: 20px;'>
        <img src='data:image/png;base64,{logo_base64}' width='150'/>
    </div>
""", unsafe_allow_html=True)

col_left, right_col = st.columns([1,5])


with col_left:
      input_method = st.radio("Choose Data Input Method:",("Manual Data Entry", "Upload Excel File"))

## Loading the serialized trained model
model = pickle.load(open("Latest Best NN Model.pkl", 'rb'))

## Loading the serialized scikit learn pipeline
pipeline = pickle.load(open("Latest Best NN Pipeline.pkl", 'rb'))


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

            ## Square Footage	
            square_footage = st.number_input('Square Footage', min_value=0.0, format="%.1f")

            # Locale Type
            locale_type = st.selectbox('Locale Type', ['Rural', 'Suburban', 'Urban'])




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
                  "Square Footage":[square_footage],
                  "Locale Type":[locale_type],
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
                              "Square Footage","Locale Type",
                              '3 Months Cumulative Sales', '6 Months Cumulative Sales', '9 Months Cumulative Sales', '12 Months Cumulative Sales']
            
            selected_columns = ['Age_25_34_Ratio_1km', 'Weighted 2023 Female Population 25 to 29 Years', 'Income Tier',
                              'Weighted 2023 Occupations in Trades, Transport, Operators', 'Restaurant', 'Stores', 'YoungAdults_per_sqft',
                              'Gas Station', 'Weighted 2023 Total Population Median Age', 'Locale Type', 'Amenity_per_sqft', 'Province',
                              'Daytime Population Range', 'Weighted 2023 Median Household Income', 'Income_per_sqft', 'Banks', 'Offices',
                              'Grocery Stores', 'Young_Adults_Ratio_3km', 'Weighted 2023 Occupation Management', 'Crowd_score',
                              'Weighted 2023 Female Population 30 to 34 Years', 'Crowd_per_sqft', 'Visibility_score',
                              'Weighted 2023 Female Population 20 to 24 Years', 'Pop_per_sqft', 'Amenity_Score', 'DaytimePop_per_sqft',
                              'Weighted 2023 Total Population', 'Weighted 2023 Daytime Population', 'Comp_1km_to_Pop',
                              'Weighted 2023 Male Population 20 to 24 Years', 'Weighted 2023 Tobacco Products, Alcoholic Beverages',
                              'Income per Capita 1km', 'Cumulative 3km Competitors', 'Visibility_per_sqft', 'Population_per_Bank',
                              'Crowd_Visibility_Interaction', 'Parking', 'Competitor_3km_per_sqft', 'Comp_3km_to_Pop']

            
            if uploaded_file is not None:
                  input_df = pd.read_excel(uploaded_file, index_col=0)
                  if not 'Store Internal Name' in input_df.columns:
                        input_df['Store Internal Name'] = "NA"

                  for col in input_df.columns:
                        if 'parking' in col.lower():
                              input_df.rename(columns={col : 'Parking'}, inplace=True)
                              break

                  for col in pipeline_columns:
                        if col not in input_df.columns:
                              input_df[col] = 0
                  input_df = input_df[pipeline_columns]

            else:
                  st.write("Incorrect File Format")

      if st.button("Predict"):

      # Save the Stores Names
            stores_name = input_df['Store Internal Name'].tolist()

            # Preprocess the input data
            processed_data = pipeline.transform(input_df)            

            # Dropping target cols
            target_cols = ["3 Months Cumulative Sales", "6 Months Cumulative Sales", "9 Months Cumulative Sales", "12 Months Cumulative Sales", "Store Internal Name"]
            processed_data.drop(target_cols, axis=1, inplace=True)

            # Make prediction
            prediction = model.predict(processed_data[selected_columns])

            if input_df.shape[0] == 1:
                  # st.subheader("Predictions for Store 🏪")
                  prediction_df = pd.DataFrame({
                        "3 Months Cumulative Sales": [prediction[0][0]], 
                        "6 Months Cumulative Sales": [prediction[0][1]],
                        "9 Months Cumulative Sales": [prediction[0][2]],
                        "12 Months Cumulative Sales": [prediction[0][3]]
                        }).astype(int)

            elif input_df.shape[0] > 1:

                  # Create prediction DataFrame
                  pred3 = []
                  pred6 = []
                  pred9 = []
                  pred12 = []
                  for row in prediction:
                        pred3.append(row[0])
                        pred6.append(row[1])
                        pred9.append(row[2])
                        pred12.append(row[3])
                  prediction_df = pd.DataFrame({
                        "3 Months Cumulative Sales": pred3, 
                        "6 Months Cumulative Sales": pred6,
                        "9 Months Cumulative Sales": pred9,
                        "12 Months Cumulative Sales": pred12
                        }).astype(int)
            prediction_df.insert(0, "Store Name", stores_name)
            prediction_df = prediction_df.sort_values(by="Store Name").reset_index(drop=True)

            st.subheader("Predictions Table 🏪")
            with st.container(border=True):
                  st.write(prediction_df)

            # Show individual line charts
            st.subheader("Individual Store Line Charts 📈")
            for idx, row in prediction_df.iterrows():
                  store_name = row["Store Name"]
                  sales_data = pd.DataFrame({
                  "Month": ["3 Months", "6 Months", "9 Months", "12 Months"],
                  "Cumulative Sales": [row["3 Months Cumulative Sales"],
                                          row["6 Months Cumulative Sales"],
                                          row["9 Months Cumulative Sales"],
                                          row["12 Months Cumulative Sales"]]
                  })

                  sales_data["Month"] = pd.Categorical(
                        sales_data["Month"],
                        categories=["3 Months", "6 Months", "9 Months", "12 Months"],
                        ordered=True
                  )

                  # Altair line chart with point labels
                  chart = alt.Chart(sales_data).mark_line(point=True).encode(
                                          x=alt.X("Month", 
                                                      sort=["3 Months", "6 Months", "9 Months", "12 Months"], 
                                                      axis=alt.Axis(labelAngle=0)),
                                          y="Cumulative Sales"
                  ).properties(
                        height=400,
                        title=store_name
                  )

                  # Add text labels on each point
                  text = chart.mark_text(
                        align='right',
                        baseline='bottom',
                        dx=-10,  # Shift text to the right of the point
                        size=12,
                        color='yellow'
                  ).encode(
                        text="Cumulative Sales"
                  )

                  # Combine and configure title style
                  final_chart = (chart + text).configure_title(
                        fontSize=20,
                        font='Helvetica-Bold',
                        anchor='middle'
                  ).configure_axis(
                        labelFontSize=10,
                        titleFontSize=12
    )

                  with st.container(border=True):
                        st.altair_chart(final_chart + text, use_container_width=True)