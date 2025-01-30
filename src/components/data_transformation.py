import os
import sys
import pandas as pd
import pickle
import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/vaishalikant/End-To-End-ML-Project')))

from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    transformed_data_path: str = os.path.join('artifacts', "transformed_data.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self):
        logging.info("Entered the data transformation method or component")
        try:
            df = pd.read_csv('Notebook/Ride_data.csv')
            logging.info('Read the dataset as dataframe')

            # Perform your data transformation here
            # For example, let's assume you are transforming the data and storing it in transformed_df
            transformed_df = df  # Replace this with actual transformation logic

            # Save the transformed data to a pickle file
            with open(self.transformation_config.transformed_data_path, 'wb') as f:
                pickle.dump(transformed_df, f)
            logging.info("Transformed data saved to pickle file")

        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise CustomException(e, sys)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation()