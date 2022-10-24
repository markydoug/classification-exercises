import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from scipy import stats

###################################################################################
################################### IRIS DATA #####################################
###################################################################################

def prep_iris(df):
    
    '''Prepares acquired Iris data for exploration'''
    
    df.drop_duplicates(inplace=True)
    df.drop(columns=['species_id', 'measurement_id'], inplace=True)
    df.rename(columns={'species_name':'species'}, inplace=True)
    dummy_df = pd.get_dummies(df['species'], dummy_na=False, drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    
    return df

###################################################################################
################################# TITANIC DATA ####################################
###################################################################################


def prep_titanic(df):

    '''Prepares acquired titanic data for exploration'''

    df.drop_duplicates(inplace=True)
    df.drop(columns=['passenger_id','embarked', 'pclass', 'age','deck'], inplace=True)
    df['embark_town'] = df.embark_town.fillna(value='Southampton')
    dummy_df = pd.get_dummies(df[['sex','class','embark_town']], dummy_na=False, drop_first=[True, True])
    df = pd.concat([df, dummy_df], axis=1)
    
    return df


def impute_mean_age(train, validate, test):
    '''
    This function imputes the mean of the age column for
    observations with missing values.
    Returns transformed train, validate, and test df.
    '''
    # create the imputer object with mean strategy
    imputer = SimpleImputer(strategy = 'mean')
    
    # fit on and transform age column in train
    train['age'] = imputer.fit_transform(train[['age']])
    
    # transform age column in validate
    validate['age'] = imputer.transform(validate[['age']])
    
    # transform age column in test
    test['age'] = imputer.transform(test[['age']])
    
    return train, validate, test

###################################################################################
################################## TELCO DATA #####################################
###################################################################################


def prep_telco(df):
    '''Prepares acquired teclo data for exploration'''

    df.drop_duplicates(inplace=True)
    df.drop(columns=['customer_id','payment_type_id', 'internet_service_type_id','contract_type_id'], inplace=True)
    df = df[df.total_charges!=' ']
    df.total_charges = df.total_charges.astype(float)
    
    # Creating a list of our categorical columns
    catcol = [col for col in df.columns if df[col].dtype == 'O']
    dummy_df = pd.get_dummies(df[catcol], dummy_na=False, drop_first=[True, True])

    df = pd.concat([df, dummy_df], axis=1)
    
    return df

###################################################################################
################################## SPLIT DATA #####################################
###################################################################################


def train_validate_test_split(df, target):
    '''
    Takes in a data frame and the target variable column  and returns
    train (80%), validate (15%), and test (10%) data frames.
    '''
    train, test = train_test_split(df,test_size = 0.1, stratify = df[target], random_state=27)
    train, validate = train_test_split(train, test_size = 0.166666666666, stratify = train[target],random_state=27)
    
    return train, validate, test

def prep_for_model(train, validate, test, target):
    '''
    Takes in train, validate, and test data frames
    then splits  for X (all variables but target variable) 
    and y (only target variable) for each data frame
    '''
    drop_columns = list(train.select_dtypes(include='object').columns) + [target]

    X_train = train.drop(columns=drop_columns)
    y_train = train[target]

    X_validate = validate.drop(columns=drop_columns)
    y_validate = validate[target]

    X_test = test.drop(columns=drop_columns)
    y_test = test[target]

    return X_train, y_train, X_validate, y_validate, X_test, y_test

def prep_df_for_model(df, target):
    '''
    Takes in a data frame and the target variable column, splits the data
    into train (80%), validate (15%), and test (10%) data frames
    then splits again for X (all variables but target variable) and 
    y (only target variable) for each data frame
    '''
    
    train, validate, test = train_validate_test_split(df, target)

    X_train, y_train, X_validate, y_validate, X_test, y_test = prep_for_model(train, validate, test, target)

    return X_train, y_train, X_validate, y_validate, X_test, y_test

###################################################################################
################################## EXPLORE DATA ###################################
###################################################################################

def explore_num_uvar(df):
    '''
    Takes in a data frame and returns univarite stats for numerical data
    '''
    num_vars = list(df.select_dtypes(include=np.number).columns)
    for col in num_vars:
        print(col)
        print(df[col].describe())
        df[col].hist()
        plt.show()
        sns.boxplot(y=col, data=df)
        plt.show()

def explore_cat_uvar(df):
    '''
    Takes in a data frame and a list of categorical variables
    Returns univarite stats
    '''
    cat_vars = list(df.select_dtypes(include='object').columns)
    for col in cat_vars:
        print(col)
        print(df[col].value_counts())
        print(df[col].value_counts(normalize=True)*100)
        sns.countplot(x=col, data=df)
        plt.show()

def explore_num_bvar(df, target):
    '''
    Takes in a data frame, target variable, and a 
    list of numerical variables. Returns bivarite stats 
    '''
    num_vars = list(df.select_dtypes(include=np.number).columns)
    for col in num_vars:
        sns.barplot(x=target, y=col, data=df)
        rate = df[col].mean()
        plt.axhline(rate,  label = f'Overall Mean of {col}', linestyle='dotted', color='black')
        plt.legend()
        plt.show()

def explore_cat_bvar(df, target):
    '''
    Takes in a data frame, target variable, and a 
    list of categorical variables. Returns bivarite stats 
    '''
    cat_vars = list(df.select_dtypes(include='object').columns)
    for col in cat_vars:
        sns.barplot(x=col, y=target, data=df)
        rate = df[target].mean()
        plt.axhline(rate, label = f'Average {target} rate', linestyle='--', color='black')
        plt.legend()
        plt.show()