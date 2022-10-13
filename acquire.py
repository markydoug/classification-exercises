import pandas as pd
import numpy as np
import os

from env import get_db_url
from pydataset import data

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
        df.to_csv(filename)

        return df

#function to go get species data from the iris database
def get_iris_data():
    filename = 'iris.csv'
    #if cached file exists use it
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    #else go get the data
    else:
        df =  pd.read_sql('''SELECT * FROM measurements 
        JOIN species USING(species_id);
        ''', get_db_url('iris_db'))
        #save the data locally for later use
        df.to_csv(filename)
        
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
        FROM customer_details cd
        JOIN customer_contracts cc USING(customer_id)
        JOIN contract_types ct USING(contract_type_id)
        JOIN customer_churn cch USING(customer_id)
        JOIN customer_payments cp USING(customer_id)
        JOIN customer_signups cs USING(customer_id)
        JOIN customer_subscriptions csub USING(customer_id)
        JOIN internet_service_types ist USING(internet_service_type_id)
        JOIN payment_types pt USING(payment_type_id);
        ''', get_db_url('telco_churn'))
        #save the data locally for later use
        df.to_csv(filename)
        
        return df
