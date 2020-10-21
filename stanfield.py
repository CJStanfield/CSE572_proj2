# %%

# Import Libraries
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime, time, timedelta

# ENUM
AUTO_MODE = "AUTO MODE ACTIVE PLGM OFF"
DATE_FORMAT_INSULIN = '%m/%d/%Y'
DATE_FORMAT_GCM = '%m/%d/%y'
TIME_FORMAT = '%H:%M:%S'


def load_data(gcm_path, insulin_path):
    # load CGM and Insulin data

    columns1 = ['Index', 'Date', 'Time', 'Sensor Glucose (mg/dL)', 'ISIG Value', 'Sensor Exception']
    gcm_data = pd.read_excel(gcm_path, usecols=columns1, index_col='Index')
    gcm_data['Sensor Glucose (mg/dL)'] = gcm_data['Sensor Glucose (mg/dL)'].interpolate(method='linear', limit_direction='backward')
    gcm_data['ISIG Value'] = gcm_data['ISIG Value'].interpolate(method='linear', limit_direction='backward')
    gcm_data['DateTime'] = gcm_data.apply(lambda row: datetime.combine(row['Date'].date(), row['Time']), axis=1)
    gcm_data['isMeal'] = pd.Series([2 for x in range(len(gcm_data.index))], index=gcm_data.index)
    gcm_data = gcm_data.drop(['Date', 'Time'], axis=1).reset_index(drop=True)

    columns2 = ['Date', 'Time', 'BWZ Carb Input (grams)']
    insulin_data = pd.read_excel(insulin_path, usecols=columns2)
    insulin_data['DateTime'] = insulin_data.apply(lambda row: datetime.combine(row['Date'].date(), row['Time']), axis=1)
    insulin_data = insulin_data.drop(['Date', 'Time'], axis=1).reset_index(drop=True)

    meal_data = insulin_data[~insulin_data['BWZ Carb Input (grams)'].isnull()]
    result_data = locateMealsTimes(gcm_data, meal_data)
    return result_data

def generate_data_row(gcm_data, isig_value, isMeal):
    row = np.array(gcm_data)
    row = np.append(row, isig_value)
    row = np.append(row, isMeal)
    return row.astype(np.int)

def locateMealsTimes(gcm_data, meal_data):
    columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', 'Avg ISIG Value', 'isMeal']
    results = pd.DataFrame(columns= columns)

    for i in range(meal_data.shape[0] - 1, 1, -1):
        time_delta = (meal_data['DateTime'].iloc[i - 1] - meal_data['DateTime'].iloc[i]).total_seconds()
        percentage_diff = abs(time_delta - 7200)/time_delta

        start_time = meal_data['DateTime'].iloc[i]
        end_time = start_time

        if percentage_diff < 0.05:
            # calculate meal times first
            start_time = start_time + timedelta(hours=1.5)
            end_time = start_time + timedelta(hours=4)
        elif time_delta > 2 * 3600:
            end_time = start_time + timedelta(hours=2)

        # gcm_data.loc[(start_time <= gcm_data['DateTime']) & (gcm_data['DateTime'] < end_time), ['isMeal']] = 1
        meal_duration = gcm_data[(start_time <= gcm_data['DateTime']) & (gcm_data['DateTime'] < end_time)]
        if meal_duration.shape[0] == 24:
            results.loc[len(results)] = generate_data_row(gcm_data=meal_duration['Sensor Glucose (mg/dL)'].values,
                                                          isig_value=meal_duration['ISIG Value'].mean(),
                                                          isMeal=1)
        # non meal data
        # basic idea: Set the new start time to the end of the meal data. Look ahead in two hour intervals to check
        # to see if a meal lies within the interval. If no meal exists within 2 hours, set that two hour window as a non
        # meal data. Repeat this until a meal time lies within a time interval
        start_time = end_time
        end_time = end_time + timedelta(hours=2)
        while (meal_data['DateTime'].iloc[i - 1] - end_time).total_seconds() > 0:
            gcm_data.loc[(start_time <= gcm_data['DateTime']) & (gcm_data['DateTime'] < end_time), ['isMeal']] = 2
            start_time = end_time
            end_time = end_time + timedelta(hours=2)

            meal_duration = gcm_data[(start_time <= gcm_data['DateTime']) & (gcm_data['DateTime'] < end_time)]
            if meal_duration.shape[0] == 24:
                results.loc[len(results)] = generate_data_row(gcm_data=meal_duration['Sensor Glucose (mg/dL)'].values,
                                                              isig_value=meal_duration['ISIG Value'].mean(),
                                                               isMeal=0)
    results = results.reset_index(drop=True)
    results.to_csv('training_data.csv')
    return results


def extract_feature_data(gcm_path, insulin_path):
    print("Begin Data Mining")
    data = load_data(gcm_path, insulin_path)
    print("Extracting Features")
    # Normalize the data
    data = (data - data.min())/(data.max()-data.min())

    # Extract features on the data
    # First Feature: Average of time series data
    data['mean'] = data.mean(axis=1)
    # Second Feature: Variance
    data['variance'] = data.var(axis=1)
    # Third Feature: Standard Deviation
    data['std'] = data.std(axis=1)




def train_model(training_data):
    data = pd.read_csv(training_data, index_col=0)
    X = data[data.columns[:-1]]
    y = data[data.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=101)

    return


if __name__ == '__main__':
    extract_feature_data(gcm_path="CGMData670GPatient2.xlsx", insulin_path="InsulinAndMealIntake670GPatient2.xlsx")
    # train_model("training_data.csv")
