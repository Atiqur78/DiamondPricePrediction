# Basic  import
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj,evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trainer_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent Variables')
            X_train, y_train, X_test, y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                'Linear Regression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'Elasticnet':ElasticNet(),
                'DecisionTree':DecisionTreeRegressor()
               
            }
            model_report: dict= evaluate_model(models, X_train, y_train, X_test, y_test)
            print(model_report)
            print('\n===========================================================')
            logging.info(f'Model report: {model_report.values()}')

            # To get the best model from dictionary
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            best_model = models[best_model_name]

            print(f'Best Model Found, Model Name : {best_model_name}, R2 Score: {best_model_score}')
            print('\n===========================================================')
            logging.info(f'Best Model Found, Model Name : {best_model_name}, R2 Score: {best_model_score}')

            save_obj(
                file_path=self.model_trainer_config.trainer_model_file_path, 
                obj=best_model
            )
        except Exception as e:
            logging.info('Error in Model Training')
            raise CustomException(e, sys)

