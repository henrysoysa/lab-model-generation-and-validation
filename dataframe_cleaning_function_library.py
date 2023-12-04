import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def dataframe_importer(url):
    """This function aims to import the specific csv dataset for the workbook by prompting for the url of the CSV from the github repo.
    This function will then specify the preset specific column names for the workbooks in order to load the csv into the dataframe into df.
     """
    #url = input("Please enter the csv url: ")
    df = pd.read_csv(url)
    return df

def unnamed_col_remover(df, colnames: list=['unnamed: 0']):
    colnames = [x.lower() for x in colnames]
    df.columns = [x.lower() for x in df.columns]
    df2 = df.copy()
    for col in colnames:
        if col in df2.columns:
            df2 = df2.drop(columns=col)
    return df2

def column_renamer(df):
    """This function will use the lamba function to first iterate through all column names to bring them to lowercase, and then replace all spaces with _underscores_
    and renames st column to state before returning the updated dataframe."""
    df.columns = [x.lower() for x in df.columns]
    df.columns = df.columns.str.replace(' ', '_')
    df.rename(columns={"st":"state"}, inplace=True)
    return df

def invalid_value_cleaner(df):
    """This function will scan through specific columns with a preset dictionary of values and replace any matching criteria with the updated element in order to have
    a cleaner dataset. """
    gender_dict = {'Femal':'F', 'female':'F', 'Male':'M'}
    df['gender'] = df['gender'].replace(gender_dict)
    state_dict = {'AZ':'Arizona', 'Cali':'California', 'WA':'Washington'}
    df['state'] = df['state'].replace(state_dict)
    df['education'] = df['education'].replace({'Bachelors':'Bachelor'})
    car_dict = {'Sports Car':'Luxury', 'Luxury SUV':'Luxury', 'Luxury Car':'Luxury'}
    df['vehicle_class'] = df['vehicle_class'].replace(car_dict)
    if df['customer_lifetime_value'].dtype == 'O':
        df['customer_lifetime_value'] = df['customer_lifetime_value'].str.replace('%','')
    return df

def datatype_formatter(df):
    """This function will first set CLV as a float datatype column, and then removes the first two characters from open complaints, and then removing the last 3
    characters, so that 1/5/00 becomes 5; representing the number of open complaints for the particular customer."""
    df['customer_lifetime_value'] = df['customer_lifetime_value'].astype(float)
    if df['number_of_open_complaints'].dtype == 'O' and len(df['number_of_open_complaints'][0]) > 2:
        df['number_of_open_complaints'] = df['number_of_open_complaints'].str.split('/').str[1]
    else:
        print("pass")
    return df

def null_value_method(df):
    """This function will first populate the gender column NaN rows with the mode value, as gender has the most null values in the dataset, and then dropping any
     additional rows containing NaN elements in order to get the most value from the dataset before removing null rows. This means the dataset has 1068 rows instead
     of 952 for further analysis."""
    df['gender'] = df['gender'].fillna(df['gender'].mode()[0])
    df = df.dropna()
    if df['number_of_open_complaints'].dtype != 'int64':
        df['number_of_open_complaints'] = df['number_of_open_complaints'].astype(int)
    return df

def na_replace_with_descriptive_stat(df: pd.DataFrame, cols: list, method: str='mode'):
    """
    Function to replace na values in a column with a preferred statistical value. This can be mode, median or mean
    Only one method can be passed per list of columns.
    inputs: df, cols to replace na values, method to use
    output: df with replaced na values in named columns
    """
    df2 = df.copy()
    for i in cols:
        if method == 'mode':
            value = df2[i].mode().values[0]
            df2[i] = df2[i].fillna(value)
        elif method == 'mean':
            df2[i] = df2[i].fillna(df2[i].mean())
        elif method == 'median':
            df2[i] = df2[i].fillna(df2[i].median())
        else:
            pass
    return df2

def column_dtype_changer(df: pd.DataFrame, col: list, type: str):
    """
    Function to change dataframe column to a desired data type.
    Would recommend using when column is already cleaned, no na values.
    """
    df2 = df.copy()
    for i in cols:
        if type == 'int':
            df2[i] = df2[i].astype(int)
        elif type == 'float':
            df2[i] = df2[i].astype(float)
    return df2


def duplicated_formatter(df):
    """This function will take a slice of the dataframe where no duplicated values are detected in the rows. """
    df = df.loc[df.duplicated() == False]
    return df

def value_replacer(df: pd.DataFrame, cols_and_vals):
    """
    Function to replace specific values (illformatted or incorrect) with desired values in specific dataframe columns
    Inputs: dataframe, dict with column name and KVPs for values.
    Output: dataframe
    example of cols_and_vals:
    # has to be a tuple for below function to accept it
    values_to_replace = (('vehicle_class', {'Sports Car':'Luxury', 'Luxury SUV':'Luxury', 'Luxury Car':'Luxury'}),)
    """
    df2 = df.copy()
    for i in cols_and_vals:
        col_name = i[0]
        values = i[1]
        df2[col_name] = df2[col_name].replace(values)
    return df2

def column_cleaner_pipeline(url):
    """This function is a pipeline of all the above functions to clean this specific dataframe from start to finish starting with the entry of the url."""
    df = dataframe_importer(url)
    df = column_renamer(df)
    df = invalid_value_cleaner(df)
    df = datatype_formatter(df)
    df = null_value_method(df)
    df = duplicated_formatter(df)
    return df

def numerical_df_sns_histplot(df: pd.DataFrame):
    """
    function:
    inputs:
    outputs: seaborn histogram plots with kde=True on subplots in one row with multiple columns
    """
    number_of_columns = len(df.columns)
    fig, axs = plt.subplots(nrows=1, ncols=number_of_columns, figsize=(20,8))

    for i in range (0,number_of_columns):
        column = df.columns[i]
        sns.histplot(x=df[column], kde=True, ax=axs[i])
    plt.show()

def multicollinearity_columns_search_and_drop(df:pd.DataFrame, y: str, corr_threshold: float=0.9):
    """
    function to search columns in a numeric df (discounting y column) that have a correlation between one another greater than the threshold.
    Default threshold is 0.9. If two columns do meet the correlation threshold, remove column less correlated with y, and compute
    correlation again until the columns meeting the threshold of correlation are all removed from the dataframe.
    Inputs: dataframe, y column to remove, threshold of correlation to measure for
    Outputs: correlation matrix, updated correlation dataframe, and names of columns dropped from dataframe if applicable.
    If not columns dropped from dataframe, relevant message displayed to console.
    """
    df2 = df.copy()
    df_corr = df2.corr()
    df_corr_no_y = df_corr.drop(columns=y)
    corr_triu = df_corr_no_y.where(~np.tril(np.ones(df_corr_no_y.shape)).astype(bool))
    multicollineary_cols = corr_triu.stack()[(corr_triu.stack() < 1.0) & (corr_triu.stack() > corr_threshold)].unstack().columns.tolist()
    if len(multicollineary_cols) == 0:
        print("No multicollinearity between X columns found with threshold of " + str(corr_threshold) + ". No features dropped")
    else:
        while len(multicollineary_cols) >= 1:
            multicollineary_corr_vals = []
            for col in multicollineary_cols:
                multicollineary_corr_vals.append(df_corr[y][col])
            min_corr_val_ind = multicollineary_corr_vals.index(min(multicollineary_corr_vals))
            multicollineary_col_to_drop = multicollineary_cols[min_corr_val_ind]
            df2.drop(columns=(multicollineary_col_to_drop), inplace=True)
            df_corr = df2.corr()
            df_corr_no_y = df_corr.drop(columns=y)
            corr_triu = df_corr_no_y.where(~np.tril(np.ones(df_corr_no_y.shape)).astype(bool))
            multicollineary_cols = corr_triu.stack()[(corr_triu.stack() < 1.0) & (corr_triu.stack() > corr_threshold)].unstack().columns.tolist()
            print(str(multicollineary_col_to_drop) + " has been dropped from dataframe")

    return df2.corr()

def compute_vif(df: pd.DataFrame, columns: list):
    """
    NOTE: There can be no NAN values in your dataframe before running this function,
    otherwise the function will not work.

    """
    X = df.loc[:, columns]
    # the calculation of variance inflation requires a constant
    X.loc[:,'intercept'] = 1

    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif.loc[vif['Variable']!='intercept'].sort_values('VIF', ascending=False).reset_index(drop=True)
    return vif