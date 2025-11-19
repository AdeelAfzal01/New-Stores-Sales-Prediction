import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

# Custom Transformer for Imputation
# Custom Transformer for Imputation
class ImputeMissingValues(BaseEstimator, TransformerMixin):
    def __init__(self, num_columns, cat_columns):
        self.num_columns = num_columns
        self.cat_columns = cat_columns
        self.median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        self.frequent_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    def fit(self, X, y=None):
        self.median_imputer.fit(X[self.num_columns])
        self.frequent_imputer.fit(X[self.cat_columns])
        return self
    
    def transform(self, X, y=None):
        # X.ffill(inplace=True)
        # X.bfill(inplace=True)
        X = X.copy()
        X[self.num_columns] = self.median_imputer.transform(X[self.num_columns])
        X[self.cat_columns] = self.frequent_imputer.transform(X[self.cat_columns])
        return X
    
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selected_features, target_cols):

        self.selected_features = selected_features
        self.target_cols = target_cols

    def fit(self, X, y=None):
        # No fitting needed; just storing selected feature names
        return self  

    def transform(self, X, y=None):
        # X = X.copy()
        return X[self.selected_features + self.target_cols]  # Select only the specified columns
    
class StandarizeParking(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        # X['Parking'] = X['Parking(street, No Parking, Parking lot)'].apply(self._standardize_parking)
        X['Parking'] = X['Parking'].apply(self._standardize_parking)
        # X = X[X.columns.difference(['Parking(street, No Parking, Parking lot)'])]
        return X
    
    def _standardize_parking(self, value):
        if isinstance(value, str):
            value = value.lower()
            if 'no parking' in value:
                return 'no parking'
            elif 'parking lot' in value or 'parking' in value:
                return 'parking'
            elif 'street' in value:
                return 'street'  
        return 'unknown'
   
class MapColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.binary_mapping = {'No': 0, 'Yes': 1}
        self.parking_mapping = {'no parking': 0, 'street': 1, 'parking': 2}
        self.province_mapping =  {'Saskatchewan': 1, 'Alberta': 2, 'British Columbia': 3, 'Manitoba': 4, 'Ontario': 5}
        self.locale_mapping = {'Rural': 1, 'Urban': 2, 'Suburban': 3}
        # self.locale_mapping = {'Rural': 1445502, 'Urban': 1937581, 'Suburban': 2281160}        
        self.default_value = -1
        self.encoder_locale = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
        self.encoder_province = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')


    def fit(self, X, y=None):
        # self.encoder_locale.fit(X[['Locale Type']])
        # self.encoder_province.fit(X[['Province']])
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        # Apply mapping with a fallback default for unseen values
        X['Crowd_score'] = X['Crowd_score'].map(lambda val: self.binary_mapping.get(val, self.default_value))
        X['Grocery Stores'] = X['Grocery Stores'].map(lambda val: self.binary_mapping.get(val, self.default_value))
        X['Gas Station'] = X['Gas Station'].map(lambda val: self.binary_mapping.get(val, self.default_value))
        X['Parking'] = X['Parking'].map(lambda val: self.parking_mapping.get(val, self.default_value))
        X['Province'] = X['Province'].map(lambda val: self.province_mapping.get(val, self.default_value))
        X['Locale Type'] = X['Locale Type'].map(lambda val: self.locale_mapping.get(val, self.default_value))

        # one_hot_encoded_province = self.encoder_province.transform(X[['Province']])
        # one_hot_df_province = pd.DataFrame(one_hot_encoded_province,
        #                         columns=self.encoder_province.get_feature_names_out()).astype(int)
        # X = pd.concat([X.reset_index(drop=True), one_hot_df_province.reset_index(drop=True)], axis=1)
        # X = X.drop(['Province'], axis=1)
        
        # one_hot_encoded = self.encoder_locale.transform(X[['Locale Type']])
        # one_hot_df = pd.DataFrame(one_hot_encoded,
        #                         columns=self.encoder_locale.get_feature_names_out()).astype(int)
        # X = pd.concat([X.reset_index(drop=True), one_hot_df.reset_index(drop=True)], axis=1)
        # X = X.drop(['Locale Type'], axis=1)
        return X
    
class KernelPCAFeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, n_components, kernel, target_cols, non_similar_cols):
        self.n_components = n_components
        self.kernel = kernel
        self.target_cols = target_cols
        self.non_similar_cols = non_similar_cols
        self.pca = KernelPCA(n_components=self.n_components, kernel=self.kernel)

    def fit(self, X, y=None):
        # self.pca.fit(X.drop(self.target_cols + self.non_similar_cols, axis=1))
        self.pca.fit(X.drop(self.target_cols, axis=1))
        return self

    def transform(self, X, y=None):
        X = X.copy()
        pca_data = self.pca.transform(X.drop(self.target_cols, axis=1))
        pca_df = pd.DataFrame(pca_data, columns=self.pca.get_feature_names_out().tolist())
        pca_df[self.target_cols] = X[self.target_cols]
        return pca_df.reset_index(drop=True)
    
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.lower_thresholds = {}
        self.upper_thresholds = {}
        self.medians = {}
        self.q90 = {}

    def fit(self, X, y=None):
        if isinstance(self.cols, list):
            for col in self.cols:
                q1, q3 = X[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                self.lower_thresholds[col] = q1 - (1.5 * iqr)
                self.upper_thresholds[col] = q3 + (1.5 * iqr)
                self.medians[col] = X[col].median()
                self.q90[col] = X[col].quantile(0.90)
        elif isinstance(self.cols, str):
            col = self.cols
            q1, q3 = X[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.lower_thresholds[col] = q1 - (1.5 * iqr)
            self.upper_thresholds[col] = q3 + (1.5 * iqr)
            self.medians[col] = X[col].median()
            self.q90[col] = X[col].quantile(0.90)
            
        return self

    def transform(self, X):
        X = X.copy()
        if isinstance(self.cols, list):
            for col in self.cols:
                # X.loc[X[col] < self.lower_thresholds[col], col] = self.medians[col]
                X.loc[X[col] > self.upper_thresholds[col], col] = self.q90[col]
        elif isinstance(self.cols, str):
            col = self.cols
            X.loc[X[col] > self.upper_thresholds[col], col] = self.q90[col]
        return X

class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_keep):
        # self.columns_to_scale = columns_to_scale
        self.columns_to_keep = columns_to_keep
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        # Fit StandardScaler on specified columns
        # self.scaler.fit(X[self.columns_to_scale])
        self.scaler.fit(X[X.drop(columns=self.columns_to_keep, axis=1).columns.tolist()])
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        # Scale specified columns
        # scaled_values = self.scaler.transform(X[self.columns_to_scale])
        # scaled_df = pd.DataFrame(scaled_values, columns=self.columns_to_scale, index=X.index)
        scaled_values = self.scaler.transform(X[X.drop(columns=self.columns_to_keep, axis=1).columns.tolist()])
        scaled_df = pd.DataFrame(scaled_values, columns=X.drop(columns=self.columns_to_keep, axis=1).columns.tolist(), index=X.index)
        
        # Combine scaled columns with the unchanged columns
        result_df = pd.concat([scaled_df, X[self.columns_to_keep]], axis=1)
        return result_df.reset_index(drop=True)
    
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # No fitting required for feature engineering
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. Aggregation Features
        # X['Total Population Mean'] = X[['2023 Total Population 1km', 
        #                                  '2023 Total Population 3km', 
        #                                  '2023 Total Population 5km']].mean(axis=1)

        X['Daytime Population Range'] = X[['2023 Daytime Population 1km', 
                                           '2023 Daytime Population 3km', 
                                           '2023 Daytime Population 5km']].max(axis=1) - \
                                         X[['2023 Daytime Population 1km', 
                                            '2023 Daytime Population 3km', 
                                            '2023 Daytime Population 5km']].min(axis=1)

        # 2. Ratios and Percentages
        # X['Male-to-Female Ratio 1km'] = X['2023 Male Population 20 to 24 Years 1km'] / \
        #                                 (X['2023 Female Population 20 to 24 Years 1km'] + 1e-9)

        # X['Young Adult Ratio 1km'] = (X['2023 Male Population 20 to 24 Years 1km'] + 
        #                               X['2023 Female Population 20 to 24 Years 1km']) / \
        #                              (X['2023 Total Population 1km'] + 1e-9)

        # X['Daytime-to-Total Population Ratio 1km'] = X['2023 Daytime Population 1km'] / \
        #                                              (X['2023 Total Population 1km'] + 1e-9)

        X['Income per Capita 1km'] = X['2023 Median Household Income 1km'] / \
                                     (X['2023 Total Population 1km'] + 1e-9)

        # 3. Binned Features
        income_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        X['Income Tier'] = pd.cut(X['2023 Median Household Income 1km'], bins=[0, 50000, 100000, np.inf], 
                                  labels=['Low', 'Medium', 'High'])
        X['Income Tier'] = X['Income Tier'].map(income_mapping)
        X['Income Tier'] = X['Income Tier'].fillna(0)
        X['Income Tier'] = X['Income Tier'].astype(int)

        # X['Population Density Category'] = pd.cut(X['2023 Total Population 1km'], bins=[0, 1000, 5000, np.inf], 
        #                                           labels=['Low', 'Medium', 'High'])
        # X['Population Density Category'] = X['Population Density Category'].map(income_mapping)
        # X['Population Density Category'] = X['Population Density Category'].fillna(0)
        # X['Population Density Category'] = X['Population Density Category'].astype(int)

        # Weighted Average
        X['Weighted 2023 Total Population'] = (0.5 * X['2023 Total Population 1km'] + 
                                    0.3 * X['2023 Total Population 3km'] + 
                                    0.2 * X['2023 Total Population 5km'])

        X['Weighted 2023 Total Population Median Age'] = (0.5 * X['2023 Total Population Median Age 1km'] + 
                                    0.3 * X['2023 Total Population Median Age 3km'] + 
                                    0.2 * X['2023 Total Population Median Age 5km'])

        X['Weighted 2023 Daytime Population'] = (0.5 * X['2023 Daytime Population 1km'] + 
                                    0.3 * X['2023 Daytime Population 3km'] + 
                                    0.2 * X['2023 Daytime Population 5km'])

        X['Weighted 2023 Median Household Income'] = (0.5 * X['2023 Median Household Income 1km'] + 
                                    0.3 * X['2023 Median Household Income 3km'] + 
                                    0.2 * X['2023 Median Household Income 5km'])

        X['Weighted 2023 Occupations Unique to Manufacture and Utilities'] = (0.5 * X['2023 Occupations Unique to Manufacture and Utilities 1km'] + 
                                    0.3 * X['2023 Occupations Unique to Manufacture and Utilities 3km'] + 
                                    0.2 * X['2023 Occupations Unique to Manufacture and Utilities 5km'])

        X['Weighted 2023 Occupations in Trades, Transport, Operators'] = (0.5 * X['2023 Occupations in Trades, Transport, Operators 1km'] + 
                                    0.3 * X['2023 Occupations in Trades, Transport, Operators 3km'] + 
                                    0.2 * X['2023 Occupations in Trades, Transport, Operators 5km'])

        X['Weighted 2023 Occupation Management'] = (0.5 * X['2023 Occupation Management 1km'] + 
                                    0.3 * X['2023 Occupation Management 3km'] + 
                                    0.2 * X['2023 Occupation Management 5km'])

        X['Weighted 2023 Tobacco Products, Alcoholic Beverages'] = (0.5 * X['2023 Tobacco Products, Alcoholic Beverages 1km'] + 
                                    0.3 * X['2023 Tobacco Products, Alcoholic Beverages 3km'] + 
                                    0.2 * X['2023 Tobacco Products, Alcoholic Beverages 5km'])

        # X['Weighted 2023 - Dominant PRIZM Segment Number'] = (0.5 * X['2023 - Dominant PRIZM Segment Number 1km'] + 
        #                             0.3 * X['2023 - Dominant PRIZM Segment Number 3km'] + 
        #                             0.2 * X['2023 - Dominant PRIZM Segment Number 5km'])

        X['Weighted 2023 Male Population 20 to 24 Years'] = (0.5 * X['2023 Male Population 20 to 24 Years 1km'] + 
                                    0.3 * X['2023 Male Population 20 to 24 Years 3km'] + 
                                    0.2 * X['2023 Male Population 20 to 24 Years 5km'])

        X['Weighted 2023 Male Population 25 to 29 Years'] = (0.5 * X['2023 Male Population 25 to 29 Years 1km'] + 
                                    0.3 * X['2023 Male Population 25 to 29 Years 3km'] + 
                                    0.2 * X['2023 Male Population 25 to 29 Years 5km'])

        X['Weighted 2023 Male Population 30 to 34 Years'] = (0.5 * X['2023 Male Population 30 to 34 Years 1km'] + 
                                    0.3 * X['2023 Male Population 30 to 34 Years 3km'] + 
                                    0.2 * X['2023 Male Population 30 to 34 Years 5km'])

        X['Weighted 2023 Female Population 20 to 24 Years'] = (0.5 * X['2023 Female Population 20 to 24 Years 1km'] + 
                                    0.3 * X['2023 Female Population 20 to 24 Years 3km'] + 
                                    0.2 * X['2023 Female Population 20 to 24 Years 5km'])

        X['Weighted 2023 Female Population 25 to 29 Years'] = (0.5 * X['2023 Female Population 25 to 29 Years 1km'] + 
                                    0.3 * X['2023 Female Population 25 to 29 Years 3km'] + 
                                    0.2 * X['2023 Female Population 25 to 29 Years 5km'])

        X['Weighted 2023 Female Population 30 to 34 Years'] = (0.5 * X['2023 Female Population 30 to 34 Years 1km'] + 
                                    0.3 * X['2023 Female Population 30 to 34 Years 3km'] + 
                                    0.2 * X['2023 Female Population 30 to 34 Years 5km'])
        ## Latest Addition
        X['Young_Adults_Ratio_3km'] = (X['2023 Male Population 20 to 24 Years 3km'] + X['2023 Female Population 20 to 24 Years 3km']) / X['2023 Total Population 3km']

        X['Age_25_34_Ratio_1km'] = (X['2023 Male Population 25 to 29 Years 1km'] + X['2023 Female Population 25 to 29 Years 1km'] +
                                    X['2023 Male Population 30 to 34 Years 1km'] + X['2023 Female Population 30 to 34 Years 1km']) / X['2023 Total Population 1km']

        X['Age_25_34_Ratio_1km'] = X['Age_25_34_Ratio_1km'].fillna(0)

        X['Population_per_Store'] = X['2023 Total Population 1km'] / (X['Stores'] + 1)
        X['Population_per_Bank'] = X['2023 Total Population 1km'] / (X['Banks'] + 1)

        X['Comp_1km_to_Pop'] = X['1km Competitors'] / (X['2023 Total Population 1km'] + 1)
        X['Comp_3km_to_Pop'] = X['3km Competitors'] / (X['2023 Total Population 3km'] + 1)


        X['Amenity_Score'] = (
            (X['Restaurant'] > 0).astype(int) + (X['Stores'] > 0).astype(int) +
            (X['Banks'] > 0).astype(int) + (X['Offices'] > 0).astype(int) +
            (X['Grocery Stores'] > 0).astype(int) + (X['Gas Station'] > 0).astype(int) +
            (X['Parking'] > 0).astype(int)
        )

        X['Crowd_Visibility_Interaction'] = X['Crowd_score'] * X['Visibility_score']

        # X['Pop_per_sqft'] = X['Weighted 2023 Total Population'] / X['Square Footage']
        X['Pop_per_sqft'] = X['Weighted 2023 Total Population'] / X['Square Footage']
        X['DaytimePop_per_sqft'] = X['Weighted 2023 Daytime Population'] / X['Square Footage']

        X['Income_per_sqft'] = X['Weighted 2023 Median Household Income'] / X['Square Footage']

        X['Competitor_1km_per_sqft'] = X['1km Competitors'] / X['Square Footage']
        X['Competitor_3km_per_sqft'] = X['3km Competitors'] / X['Square Footage']
        X['Amenity_per_sqft'] = X['Amenity_Score'] / X['Square Footage']

        X['Crowd_per_sqft'] = X['Crowd_score'] / X['Square Footage']
        X['Visibility_per_sqft'] = X['Visibility_score'] / X['Square Footage']

        X['YoungAdults_per_sqft'] = (
            X['Weighted 2023 Male Population 25 to 29 Years'] +
            X['Weighted 2023 Male Population 30 to 34 Years'] +
            X['Weighted 2023 Female Population 25 to 29 Years'] +
            X['Weighted 2023 Female Population 30 to 34 Years']
        ) / X['Square Footage']

        X = X.drop(['2023 Total Population 1km', '2023 Total Population 3km', '2023 Total Population 5km',
        '2023 Total Population Median Age 1km', '2023 Total Population Median Age 3km', '2023 Total Population Median Age 5km',
        '2023 Daytime Population 1km', '2023 Daytime Population 3km', '2023 Daytime Population 5km',
        '2023 Median Household Income 1km', '2023 Median Household Income 3km', '2023 Median Household Income 5km',
        '2023 Occupations Unique to Manufacture and Utilities 1km', '2023 Occupations Unique to Manufacture and Utilities 3km', '2023 Occupations Unique to Manufacture and Utilities 5km',
        '2023 Occupations in Trades, Transport, Operators 1km', '2023 Occupations in Trades, Transport, Operators 3km', '2023 Occupations in Trades, Transport, Operators 5km',
        '2023 Occupation Management 1km', '2023 Occupation Management 3km', '2023 Occupation Management 5km',
        '2023 Tobacco Products, Alcoholic Beverages 1km', '2023 Tobacco Products, Alcoholic Beverages 3km', '2023 Tobacco Products, Alcoholic Beverages 5km',
        '2023 - Dominant PRIZM Segment Number 1km', '2023 - Dominant PRIZM Segment Number 3km', '2023 - Dominant PRIZM Segment Number 5km',
        '2023 Male Population 20 to 24 Years 1km', '2023 Male Population 20 to 24 Years 3km', '2023 Male Population 20 to 24 Years 5km',
        '2023 Male Population 25 to 29 Years 1km', '2023 Male Population 25 to 29 Years 3km', '2023 Male Population 25 to 29 Years 5km',
        '2023 Male Population 30 to 34 Years 1km', '2023 Male Population 30 to 34 Years 3km', '2023 Male Population 30 to 34 Years 5km',
        '2023 Female Population 20 to 24 Years 1km', '2023 Female Population 20 to 24 Years 3km', '2023 Female Population 20 to 24 Years 5km',
        '2023 Female Population 25 to 29 Years 1km', '2023 Female Population 25 to 29 Years 3km', '2023 Female Population 25 to 29 Years 5km',
        '2023 Female Population 30 to 34 Years 1km', '2023 Female Population 30 to 34 Years 3km', '2023 Female Population 30 to 34 Years 5km',
        'Square Footage'], axis=1)
        
        X['3 Months Cumulative Sales'] = X['3 Months Cumulative Sales'].astype(int)
        X['6 Months Cumulative Sales'] = X['6 Months Cumulative Sales'].astype(int)
        X['9 Months Cumulative Sales'] = X['9 Months Cumulative Sales'].astype(int)
        X['12 Months Cumulative Sales'] = X['12 Months Cumulative Sales'].astype(int)

        return X

        