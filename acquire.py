import pandas as pd
import numpy as np
import os

from env import user, password, host
from pydataset import data

def get_db_url(database):
    return f'mysql+pymysql://{user}:{password}@{host}/{database}'

#function to go get passenger data from the titanic database
def get_titanic_data():
    filename = 'titanic.csv'
    #if cached file exists use it
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    #else go get the data
    else:
        df = pd.read_sql('''SELECT * FROM passengers;''', get_db_url('titanic_db'))
        #save the data locally for later use
        df.to_csv(filename, index=False)

        return df

#function to go get species data from the iris database
def get_iris_data():
    filename = 'iris.csv'
    #if cached file exists use it
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    #else go get the data
    else:
        df =  pd.read_sql('''SELECT * FROM species 
        JOIN measurements USING(species_id);
        ''', get_db_url('iris_db'))
        #save the data locally for later use
        df.to_csv(filename, index=False)
        
        return df

#function to go get customer data from the telco_churn database
def get_telco_data():
    filename = 'telco.csv'
    #if cached file exists use it
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    #else go get the data
    else:
        df = pd.read_sql('''SELECT * 
                FROM customers
                JOIN contract_types USING (contract_type_id)
                JOIN internet_service_types USING (internet_service_type_id)
                JOIN payment_types USING (payment_type_id)
        ''', get_db_url('telco_churn'))
        #save the data locally for later use
        df.to_csv(filename,index=False)
        
        return df
