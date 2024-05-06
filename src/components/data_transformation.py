from sklearn.impute import SimpleImputer ## Handling the missing value
from sklearn.preprocessing import OrdinalEncoder ## Encoding the ordinal value
from sklearn.preprocessing import StandardScaler ## Encoding the nominal value

## pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys,os
from dataclasses import dataclass
import pandas as pd
import  numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj

@dataclass
class DataTransformationconfig:
    preprocessor_ob_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformation_object(self):
        try:
            
            logging.info('Data Transformation Initiated')
            categorical_column = ['cut', 'color', 'clarity']
            numerical_column = ['carat', 'depth', 'table', 'x', 'y', 'z']

            #Define custom ranking for each ordinal variable
            cut_category = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_category = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_category = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

            logging.info('Pipeline Initiated')
            ## Numerical Pipeline
            num_pipeline =  Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            ## Categorical Pipeline
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OrdinalEncoder(categories=[cut_category, color_category, clarity_category]))
                ]
            )

            preprocessor= ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_column),
                ('cat_pipeline', cat_pipeline, categorical_column)
            ])

            return preprocessor
        
        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e,sys)



    def initiate_data_transforamtion(self, train_data_path, test_data_path):
        try:
            #READING TRAIN AND TEST DATA
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info('Read train and test data completed')
            logging.info(f'Train DataFRame head:\n{train_df.head().to_string()}')
            logging.info(f'Test DataFrame head:\n{test_df.head().to_string()}') 
            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()
            
            target_column_name='price'
            drop_columns =[target_column_name,'id']

            ## features into independent and dependebt features
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df= train_df[target_column_name]

            input_feature_test_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df= train_df[target_column_name]

            ## apply the transformation
            input_feature_train_arr =preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info('Applying preprocessing object on trainign and testing datasets')

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_obj(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )
            logging.info("Preprocessor pickle is created and saved")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )                              




        except Exception as e:
            raise CustomException(e,sys)



