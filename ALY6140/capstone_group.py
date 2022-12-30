"""
Module 2 Assignment

Created on Mon Dec  5 01:47:51 2022

@author: pravin10717
"""
from os import chdir, getcwd
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree

def getData():
    """
    Get the DataFrame with multiple countries and attributes.
    
    This function access the data for different countries over different attributes from a csv file.
    This csv data is cleaned and melted into one DataFrame.
    
    :return final_df(DataFrame): DataFrame of countries with multiple attributes.
    """
    
    #Accessing data from csv
    chdir("/Users/pravin10717/0NEU/Aly_6140_Classes/FINAL/DataSet")
    
    p_exports = pd.read_csv('Exports_1960-2021.csv', header = 2)
    p_exports.drop('Unnamed: 66', inplace = True, axis = 1)
    
    p_fertility_rate = pd.read_csv('Fertility_Rate.csv', header = 2)
    p_fertility_rate.drop('Unnamed: 66', inplace = True, axis = 1)
    
    p_GDP = pd.read_csv('GDP.csv', header = 2)
    p_GDP.drop('Unnamed: 66', inplace = True, axis = 1)
    
    p_GDP_PC = pd.read_csv('GDP_PC.csv', header = 2)
    p_GDP_PC.drop('Unnamed: 66', inplace = True, axis = 1)
    
    p_imports = pd.read_csv('Import_1960-2021.csv', header = 2)
    p_imports.drop('Unnamed: 66', inplace = True, axis = 1)
    
    p_literacy = pd.read_csv('Literacy.csv', header = 2)
    p_literacy.drop('Unnamed: 66', inplace = True, axis = 1)
    
    p_population = pd.read_csv('Population.csv', header = 2)
    p_population.drop('Unnamed: 66', inplace = True, axis = 1)
    
    p_R_And_D = pd.read_csv('R_And_D.csv', header = 2)
    p_R_And_D.drop('Unnamed: 66', inplace = True, axis = 1)
    
    p_unemployment = pd.read_csv('Unemployment.csv', header = 2)
    p_unemployment.drop('Unnamed: 66', inplace = True, axis = 1)
    
    p_HDI = pd.read_csv('HDI.csv', header = 0)
    
    p_HDI_2 = p_HDI.iloc[:,1:37]
    p_HDI_2 = p_HDI_2.drop(['hdicode'], axis = 1)
    #p_exports_df.Year.sort_values().unique()[30:]
    p_colName = ['Country Name', 'Region', 'HDI_rank_2021', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005','2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
    p_HDI_2.columns = p_colName
    
    p_region_df = p_HDI_2.iloc[:,:2]
    
    p_HDI_2_df = pd.melt(p_HDI_2, id_vars = ['Country Name'], value_vars = p_HDI_2.columns[3:], var_name='Year', value_name='HDI').sort_values('Country Name')
    
    #Melting data into required format
    p_exports_df = pd.melt(p_exports, id_vars = ['Country Name'], value_vars = p_exports.columns[4:], var_name='Year', value_name='Exports').sort_values('Country Name')    
    p_fertility_rate_df = pd.melt(p_fertility_rate, id_vars = ['Country Name'], value_vars = p_fertility_rate.columns[4:], var_name='Year', value_name='Fertility Rate').sort_values('Country Name')    
    p_GDP_df = pd.melt(p_GDP, id_vars = ['Country Name'], value_vars = p_GDP.columns[4:], var_name='Year', value_name='GDP').sort_values('Country Name')    
    p_GDP_PC_df = pd.melt(p_GDP_PC, id_vars = ['Country Name'], value_vars = p_GDP_PC.columns[4:], var_name='Year', value_name='GDP_PC').sort_values('Country Name')    
    p_imports_df = pd.melt(p_imports, id_vars = ['Country Name'], value_vars = p_imports.columns[4:], var_name='Year', value_name='Imports').sort_values('Country Name')    
    p_literacy_df = pd.melt(p_literacy, id_vars = ['Country Name'], value_vars = p_literacy.columns[4:], var_name='Year', value_name='Literacy').sort_values('Country Name')    
    p_population_df = pd.melt(p_population, id_vars = ['Country Name'], value_vars = p_population.columns[4:], var_name='Year', value_name='Population').sort_values('Country Name')    
    p_R_And_D_df = pd.melt(p_R_And_D, id_vars = ['Country Name'], value_vars = p_R_And_D.columns[4:], var_name='Year', value_name='R_And_D').sort_values('Country Name')
    p_unemployment_df = pd.melt(p_unemployment, id_vars = ['Country Name'], value_vars = p_unemployment.columns[4:], var_name='Year', value_name='Unemployment').sort_values('Country Name')
    
    #Merging multiple dataFrames into one dataFrame
    p_final_dataframe = pd.merge(p_exports_df, p_fertility_rate_df, how = 'outer', on = ['Country Name', 'Year'])
    p_final_dataframe = pd.merge(p_final_dataframe, p_GDP_df, how = 'outer', on = ['Country Name', 'Year'])
    p_final_dataframe = pd.merge(p_final_dataframe, p_GDP_PC_df, how = 'outer', on = ['Country Name', 'Year'])    
    p_final_dataframe = pd.merge(p_final_dataframe, p_imports_df, how = 'outer', on = ['Country Name', 'Year'])    
    p_final_dataframe = pd.merge(p_final_dataframe, p_literacy_df, how = 'outer', on = ['Country Name', 'Year'])    
    p_final_dataframe = pd.merge(p_final_dataframe, p_population_df, how = 'outer', on = ['Country Name', 'Year'])    
    p_final_dataframe = pd.merge(p_final_dataframe, p_R_And_D_df, how = 'outer', on = ['Country Name', 'Year'])    
    p_final_dataframe = pd.merge(p_final_dataframe, p_unemployment_df, how = 'outer', on = ['Country Name', 'Year'])    
    p_final_dataframe = pd.merge(p_final_dataframe, p_HDI_2_df, how = 'outer', on = ['Country Name', 'Year'])    
    p_final_dataframe = pd.merge(p_final_dataframe, p_region_df, how = 'outer', on = 'Country Name')
    
    #Accessing county names to filer for from the complete dataFrame.
    p_country_name = np.array(pd.read_csv("https://pkgstore.datahub.io/core/country-list/data_csv/data/d7c9d7cfb42cb69f4422dec222dbbaa8/data_csv.csv").Name)
    p_country_name1 = np.array(list(set(p_final_dataframe['Country Name']) & set(p_country_name)))
    
    #Sorting the data based on the country name and year.
    p_final_dataframe = p_final_dataframe.sort_values(['Country Name', 'Year'], ascending=[True, True])
    p_final_dataframe  = p_final_dataframe[p_final_dataframe['Country Name'].isin(p_country_name1)]
    p_final_dataframe.reset_index(inplace = True, drop = True)
    
    #Converting the datatype of the Year attribute as integer
    p_final_dataframe['Year'] = p_final_dataframe['Year'].astype(str).astype(int)
    
    #Converting the NaN values for timeseries data using the interpolate.
    #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.interpolate.html
    len(p_final_dataframe['Country Name'].unique())
    final_df = pd.DataFrame(columns = list(p_final_dataframe.columns))
    for country in p_final_dataframe['Country Name'].unique():
        temp_df = p_final_dataframe[p_final_dataframe['Country Name'] == country].sort_values('Year')
        temp_df = temp_df.interpolate(method='linear', limit_direction='forward')
        final_df = pd.concat([final_df, temp_df])
    final_df.reset_index(inplace = True, drop = True)
    final_df['Year'] = final_df['Year'].astype(str).astype(int)
    return final_df

def getHDIModel(dataset):
    """
    Get the model for predicting the Human Development Index.
    
    This function extracts the required attributes for training and testing the model.
    We drop the data entries that does not contain the attribute for which we are training the model for.    
    
    :return HDI_X_train: Data on which the model got trained.
    :return HDI_X_test: Data on which the model can be tested.
    :return HDI_y_train: Data for which the model got trained.
    :return HDI_y_test: Data for which the model can be tested.
    :return HDI_regr: Regression model to predict the HDI.
    """
    df_HDI = dataset.dropna(subset = ['HDI'])
    HDI_train = df_HDI.drop(['Country Name', 'Year', 'Exports', 'GDP', 'Imports', 'Population', 'Unemployment', 'Region', 'HDI'], axis = 1)
    HDI_test = df_HDI['HDI']
    HDI_train.R_And_D = HDI_train.R_And_D.fillna(HDI_train.R_And_D.mean())
    HDI_train.GDP_PC = HDI_train.GDP_PC.fillna(HDI_train.GDP_PC.mean())
    HDI_train.Literacy = HDI_train.Literacy.fillna(HDI_train.Literacy.mean())
    HDI_train['Fertility Rate'] = HDI_train['Fertility Rate'].fillna(HDI_train['Fertility Rate'].mean())
    HDI_X_train, HDI_X_test, HDI_y_train, HDI_y_test = train_test_split(HDI_train, HDI_test, test_size = 0.2)
    HDI_regr = tree.DecisionTreeRegressor()
    HDI_regr.fit(HDI_X_train, HDI_y_train)
    return HDI_X_train, HDI_X_test, HDI_y_train, HDI_y_test, HDI_regr

def getFertilityModelLR(dataset):
    """
    Get a Linear Regression model for predicting the Fertility rate.
    
    This function extracts the required attributes for training and testing the model.
    We drop the data entries that does not contain the attribute for which we are training the model for.    
    
    :return X_train: Data on which the model got trained.
    :return X_test: Data on which the model can be tested.
    :return y_train: Data for which the model got trained.
    :return y_test: Data for which the model can be tested.
    :return fertility_regr: Linear Regression model to predict the Fertility Rate.
    """
    fertility_df_model = dataset.dropna(subset = ['Fertility Rate'])
    train = fertility_df_model.drop(['Country Name', 'Year', 'HDI', 'R_And_D', 'Exports', 'GDP', 'Imports', 'Population', 'Unemployment', 'Region', 'Fertility Rate'], axis = 1)
    test = fertility_df_model['Fertility Rate']
    train.GDP_PC = train.GDP_PC.fillna(train.GDP_PC.mean())
    train.Literacy = train.Literacy.fillna(train.Literacy.mean())
    X_train, X_test, y_train, y_test = train_test_split(train, test, test_size = 0.2, random_state = 2)
    fertility_regr = LinearRegression()
    fertility_regr.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test, fertility_regr

def getFertilityModelDT(dataset):
    """
    Get a Decision Tree Regressor model for predicting the Fertility rate.
    
    This function extracts the required attributes for training and testing the model.
    We drop the data entries that does not contain the attribute for which we are training the model for.    
    
    :return X_train: Data on which the model got trained.
    :return X_test: Data on which the model can be tested.
    :return y_train: Data for which the model got trained.
    :return y_test: Data for which the model can be tested.
    :return fertility_regr: Decision Tree Regressor model to predict the Fertility Rate.
    """
    fertility_df_model = dataset.dropna(subset = ['Fertility Rate'])
    train = fertility_df_model.drop(['Country Name', 'Year', 'HDI', 'R_And_D', 'Exports', 'GDP', 'Imports', 'Population', 'Unemployment', 'Region', 'Fertility Rate'], axis = 1)
    test = fertility_df_model['Fertility Rate']
    train.GDP_PC = train.GDP_PC.fillna(train.GDP_PC.mean())
    train.Literacy = train.Literacy.fillna(train.Literacy.mean())
    X_train, X_test, y_train, y_test = train_test_split(train, test, test_size = 0.2, random_state = 2)
    fertility_regr = tree.DecisionTreeRegressor()
    fertility_regr.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test, fertility_regr

def getFertilityModelRF(dataset):
    """
    Get a Random Forest Regressor model for predicting the Fertility rate.
    
    This function extracts the required attributes for training and testing the model.
    We drop the data entries that does not contain the attribute for which we are training the model for.    
    
    :return X_train: Data on which the model got trained.
    :return X_test: Data on which the model can be tested.
    :return y_train: Data for which the model got trained.
    :return y_test: Data for which the model can be tested.
    :return fertility_regr: Random Forest model to predict the Fertility Rate.
    """
    fertility_df_model = dataset.dropna(subset = ['Fertility Rate'])
    train = fertility_df_model.drop(['Country Name', 'Year', 'HDI', 'R_And_D', 'Exports', 'GDP', 'Imports', 'Population', 'Unemployment', 'Region', 'Fertility Rate'], axis = 1)
    test = fertility_df_model['Fertility Rate']
    train.GDP_PC = train.GDP_PC.fillna(train.GDP_PC.mean())
    train.Literacy = train.Literacy.fillna(train.Literacy.mean())
    X_train, X_test, y_train, y_test = train_test_split(train, test, test_size = 0.2, random_state = 2)
    fertility_regr = RandomForestRegressor()
    fertility_regr.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test, fertility_regr


def getGDPModelLR(dataset):
    """
    Get the model for predicting the GDP of a country using Linear Regression model.
    
    This function extracts the required attributes for training and testing the model.
    We drop the data entries that does not contain the attribute for which we are training the model for.    
    
    :return X_train: Data on which the model got trained.
    :return X_test: Data on which the model can be tested.
    :return y_train: Data for which the model got trained.
    :return y_test: Data for which the model can be tested.
    :return GDP_regr_LM: Linear Regression model to predict the Fertility Rate.
    """
    GDP_df_model = dataset.dropna(subset = ['GDP'])
    train = GDP_df_model.drop(['Country Name', 'Year', 'Fertility Rate', 'GDP', 'GDP_PC', 'Literacy', 'Population', 'R_And_D', 'Unemployment', 'HDI', 'Region'], axis = 1)
    test = GDP_df_model.GDP
    train.Exports = train.Exports.fillna(train.Exports.mean())
    train.Imports = train.Imports.fillna(train.Imports.mean())
    X_train, X_test, y_train, y_test = train_test_split(train, test, test_size = 0.2, random_state = 2)
    GDP_regr_LM = LinearRegression()
    GDP_regr_LM.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test, GDP_regr_LM

def getGDPModelRF(dataset):
    """
    Get the model for predicting the GDP of a country using Random Forest Regression model.
    
    This function extracts the required attributes for training and testing the model.
    We drop the data entries that does not contain the attribute for which we are training the model for.    
    
    :return X_train: Data on which the model got trained.
    :return X_test: Data on which the model can be tested.
    :return y_train: Data for which the model got trained.
    :return y_test: Data for which the model can be tested.
    :return GDP_regr_RF: Random Forest Regression model to predict the Fertility Rate.
    """
    GDP_df_model = dataset.dropna(subset = ['GDP'])
    train = GDP_df_model.drop(['Country Name', 'Year', 'Fertility Rate', 'GDP', 'GDP_PC', 'Literacy', 'Population', 'R_And_D', 'Unemployment', 'HDI', 'Region'], axis = 1)
    test = GDP_df_model.GDP
    train.Exports = train.Exports.fillna(train.Exports.mean())
    train.Imports = train.Imports.fillna(train.Imports.mean())
    X_train, X_test, y_train, y_test = train_test_split(train, test, test_size = 0.2, random_state = 2)
    GDP_regr_RF = RandomForestRegressor()
    GDP_regr_RF.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test, GDP_regr_RF

if __name__ == "__main__":
    data = getData()








