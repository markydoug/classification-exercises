import pandas as pd
import numpy as np
import os
import acquire

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def prep_iris(df):
    df.drop_duplicates(inplace=True)
    df.drop(columns=['species_id', 'measurement_id'], inplace=True)
    df.rename(columns={'species_name':'species'}, inplace=True)
    dummy_df = pd.get_dummies(df['species'], dummy_na=False, drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    
    return df

def prep_titanic(df):
    df.drop_duplicates(inplace=True)
    df.drop(columns=['passenger_id','embarked', 'pclass', 'age'], inplace=True)
    df['embark_town'] = df.embark_town.fillna(value='Southampton')
    dummy_df = pd.get_dummies(df[['sex','class','embark_town']], dummy_na=False, drop_first=[True, True])
    df = pd.concat([df, dummy_df], axis=1)
    
    return df

def prep_telco(df):
    df.drop_duplicates(inplace=True)
    df.drop(columns=['customer_id','payment_type_id', 'internet_service_type_id','contract_type_id'], inplace=True)
    df = df[df.total_charges!=' ']
    df.total_charges = df.total_charges.astype(float)
    
    # Creating a list of our categorical columns
    catcol = [col for col in df.columns if df[col].dtype == 'O']
    dummy_df = pd.get_dummies(df[catcol], dummy_na=False, drop_first=[True, True])

    df = pd.concat([df, dummy_df], axis=1)
    
    return df

def train_validate_test_split(df, target):
    
    train, test = train_test_split(df,
                               train_size = 0.8,
                               stratify = df[target],
                               random_state=1234)
    train, validate = train_test_split(train,
                               train_size = 0.7,
                               stratify = train[target],
                               random_state=1234)
    
    return train, validate, test