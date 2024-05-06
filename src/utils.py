import pickle as pkl
import os, sys
import pandas  as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,'wb')as file_obj:
            pkl.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(models,X_train, y_train, X_test, y_test):
    try:
        report={}
        for model_name, model in models.items():

            #train model
            model.fit(X_train, y_train)

            #prediction Testing data
            y_test_pred = model.predict(X_test)

            # Get R2 scores for the train and test data
            test_model_score =r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)

