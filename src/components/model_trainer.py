import os
import sys
import numpy as np
from dataclasses import dataclass

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],   
                train_array[:,-1],  
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "XGBoost": XGBClassifier(objective='binary:logistic', random_state=42),
                "CatBoost": CatBoostClassifier(verbose=False),
             }
            
            params={
                "XGBoost":{
                    'max_depth': [6,7,8], 
                    'min_child_weight': [5],
                    'learning_rate': [0.1, 0.2, 0.3],
                    'n_estimators': [100, 500]
                },
                "CatBoost":{
                    'iterations': [500, 1000],
                    'depth': [4, 6],
                    'loss_function': ['Logloss'],
                    'l2_leaf_reg': np.logspace(-20, -19, 3),
                    'leaf_estimation_iterations': [10],
                    'random_seed': [42],
                    'early_stopping_rounds': [200]
                }
            }
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
             ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            #set threshold

            if best_model_score<0.55:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            f1 = f1_score(y_test, predicted)

            return f1
        


        except Exception as e:
            raise CustomException(e,sys)