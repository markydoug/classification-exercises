import pandas as pd
import numpy as np
import os
import acquire


def prep_iris():
    df = acquire.get_iris_data()
    df.drop(columns=['species_id', 'measurement_id'], inplace=True)
    df.rename(columns={'species_name':'species'}, inplace=True)
    dummy_df = pd.get_dummies(df['species'], dummy_na=False, drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    
    return df