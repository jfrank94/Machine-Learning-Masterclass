from __future__ import print_function
import numpy as np
import pandas as pd
# pd.options.mode.chained_assignment = None  # default='warn'
import pickle
import sklearn
import sys

class EmployeeRetentionModel:
    
    def __init__(self, model_location):
        with open(model_location, 'rb') as f:
            self.model = pickle.load(f)
    
    def predict_proba(self, X_new, clean=True, augment=True):
        if clean:
            X_new = self.clean_data(X_new)
        
        if augment:
            X_new = self.engineer_features(X_new)
            
        return X_new, self.model.predict_proba(X_new)
    
    # Add functions here
    def clean_data(self, df):
        # Drop duplicates
        df = df.drop_duplicates()

        # Drop temporary workers
        df = df[df.department != 'temp']

        # Missing filed_complaint values should be 0
        df['filed_complaint'] = df.filed_complaint.fillna(0)

        # Missing recently_promoted values should be 0
        df['recently_promoted'] = df.recently_promoted.fillna(0)

        # 'information_technology' should be 'IT'
        df.department.replace('information_technology', 'IT', inplace=True)

        # Fill missing values in department with 'Missing'
        df['department'].fillna('Missing', inplace=True)

        # Indicator variable for missing last_evaluation
        df['last_evaluation_missing'] = df.last_evaluation.isnull().astype(int)

        # Fill missing values in last_evaluation with 0
        df.last_evaluation.fillna(0, inplace=True)

        # Return cleaned dataframe
        return df
    
    def engineer_features(self, df):
        # Create indicator features
        df['underperformer'] = ((df.last_evaluation < 0.6) & 
                                (df.last_evaluation_missing == 0)).astype(int)

        df['unhappy'] = (df.satisfaction < 0.2).astype(int)

        df['overachiever'] = ((df.last_evaluation > 0.8) & (df.satisfaction > 0.7)).astype(int)

        # Create new dataframe with dummy features
        df = pd.get_dummies(df, columns=['department', 'salary'])

        # Return augmented DataFrame
        return df


def main(data_location, output_location, model_location, clean=True, augment=True):
    # Read dataset
    df = pd.read_csv(data_location)

    # Initialize model
    retention_model = EmployeeRetentionModel(model_location)

    # Make prediction
    df, pred = retention_model.predict_proba(df)
    pred = [p[1] for p in pred]

    # Add prediction to dataset
    df['prediction'] = pred

    # Save dataset after making predictions
    df.to_csv(output_location, index=None)

if __name__ == '__main__':
    main( *sys.argv[1:] )