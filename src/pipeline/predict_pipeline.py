import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        Gender: str,
        Geography: str,
        CreditScore: int,
        Age: int,
        Tenure: int,
        Balance: int,
        NumOfProducts: int,
        HasCrCard: int,
        IsActiveMember: int,
        EstimatedSalary: int):

        self.Gender = Gender

        self.Geography = Geography

        self.CreditScore = CreditScore

        self.Age = Age

        self.Tenure = Tenure

        self.Balance = Balance

        self.NumOfProducts = NumOfProducts

        self.HasCrCard = HasCrCard

        self.IsActiveMember = IsActiveMember

        self.EstimatedSalary = EstimatedSalary

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Gender": [self.Gender],
                "Geography": [self.Geography],
                "CreditScore": [self.CreditScore],
                "Age": [self.Age],
                "Tenure": [self.Tenure],
                "Balance": [self.Balance],
                "NumOfProducts": [self.NumOfProducts],
                "HasCrCard": [self.HasCrCard],
                "IsActiveMember": [self.IsActiveMember],
                "EstimatedSalary": [self.EstimatedSalary],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)