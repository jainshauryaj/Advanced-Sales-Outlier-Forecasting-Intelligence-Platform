# import pencilbox as pb
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import json
import sys
import subprocess
import psutil
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import boto3
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing

# from untitled import order_query
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 300)
import warnings

warnings.filterwarnings('ignore')

# Install required packages
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pencilbox', 'pymysql', 'pandasql', 'matplotlib'])

# Set pandas options
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 300)

# Set up Redshift and Trino connections
# CON_REDSHIFT = pb.get_connection("[Warehouse] Redshift")
# CON_TRINO = pb.get_connection("[Warehouse] Trino")


# Utility function to read SQL query
def read_sql_query(sql, con):
    max_tries = 3
    for attempt in range(max_tries):
        print(f"Read attempt: {attempt}...")
        try:
            start = time.time()
            df = pd.read_sql_query(sql, con)
            end = time.time()
            duration = end - start
            print(f"Time: {duration / 60} min" if duration > 60 else f"Time: {duration} s")
            return df
        except BaseException as e:
            print(e)
            time.sleep(5)


def read_order_data(city_name):
    name = 'citywise_data_csv/' + city_name + '.csv'
    # print(name)
    df = pd.read_csv(name)
    df = df.groupby(['date_','item_id','item_name']).agg({'sales_quantity':'sum','sales_value':'sum','sales_value_mrp':'sum'}).reset_index()
    # df = df[df['item_id'] == 10000953]
    return df

def get_error_data(order_data):
    error_data = order_data[order_data['remove_flag'] == 1].copy()
    error_data['min'] = 1.00
    error_data['max'] = 1.00
    error_data['error_reason'] = 'Less Than 2 carts per month'
    error_data = error_data[['item_id', 'min', 'max', 'error_reason']]
    return error_data


def preprocess_order_data(order_data):
    # date_format = "%Y-%m-%d"
    # Convert the "date_" column to pandas datetime using the determined format
    try:
        order_data['date_'] = pd.to_datetime(order_data['date_'], format='mixed')
    except:
        pass

    # Rest of the function remains unchanged
    # order_data = order_data[order_data['remove_flag'] != 1].drop(columns=['remove_flag']).dropna()
    return order_data

def get_unique_store_item_pairs(order_data):
    return order_data.groupby(['item_id']).groups.keys()


def apply_boxcox(order_data, item_id):
    # print(order_data)
    item_data = order_data[order_data['item_id'] == item_id]
    # print('step-1')
    item_data['Sales'] = item_data['Sales'].astype(float)
    # print(item_data)
    transformed_sales = stats.boxcox(item_data['Sales'])
    non_positive_mask = item_data['Sales'] <= 0
    if non_positive_mask.any():
        # Handle non-positive values, such as replacing with a small positive value or removing them
        # For example, replacing with a small positive value:
        item_data.loc[non_positive_mask, 'Sales'] = 0.001  # Replace with 0.001 or any small positive value

    # Perform the boxcox transformation on the 'Sales' column
    transformed_sales, lambda_value = stats.boxcox(item_data['Sales'])
    # print(transformed_sales)
    # print('step-2')
    # Update the 'Sales' column with the transformed values
    item_data['Sales'] = transformed_sales

    # item_data['Sales'] = transformed_sales
    # print(item_data)
    # print('step-3')
    # print(item_data)
    return item_data


def impute_outliers(item_data):
    q1 = item_data['Sales'].quantile(0.25)
    q3 = item_data['Sales'].quantile(0.75)
    iqr = q3 - q1
    lower_tail = q1 - 1.5 * iqr
    upper_tail = q3 + 1.5 * iqr
    median_sales = item_data['Sales'].median()
    item_data.loc[(item_data['Sales'] > upper_tail) | (item_data['Sales'] < lower_tail), 'Sales'] = median_sales
    return item_data


def perform_clustering(item_data):
    item_data['hours'] = item_data['date_'].dt.hour
    # print(1)
    # item_data['week'] = item_data['date_'].dt.week
    # print(2)
    item_data['day'] = item_data['date_'].dt.day
    # print(3)
    item_data['month'] = item_data['date_'].dt.month
    # print(4)
    item_data['DayOfTheWeek'] = item_data['date_'].dt.dayofweek
    # print(5)
    item_data['WeekDay'] = (item_data['DayOfTheWeek'] < 5).astype(int)
    data = item_data[['Sales', 'hours', 'WeekDay']]
    # scaler = preprocessing.StandardScaler()
    # scaled_data = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=3, init='k-means++', n_init=20, random_state=42, algorithm='auto', max_iter=100)
    cluster_labels = kmeans.fit_predict(data)
    return cluster_labels


def get_cluster_means(item_data, cluster_labels):
    item_data['cluster'] = cluster_labels

    cluster_means = item_data.groupby('cluster')['Sales'].mean()
    return cluster_means


def get_min_max_clusters(cluster_means):
    min_cluster = cluster_means.idxmin()
    max_cluster = cluster_means.idxmax()
    return min_cluster, max_cluster


def get_no_outlier_cluster(cluster_means, min_cluster, max_cluster):
    return cluster_means.drop([min_cluster, max_cluster]).idxmin()


def calculate_min_max_values(item_data, cluster_labels, min_cluster, max_cluster, no_outlier_cluster):
    min_val = item_data[cluster_labels == min_cluster]['sales_quantity'].mean()
    max_val = item_data[cluster_labels == max_cluster]['sales_quantity'].mean()
    # fval = item_data[cluster_labels == no_outlier_cluster]['sales_quantity'].mean()
    # min_fval = (min_val + fval) / 2
    # max_fval = (max_val + fval) / 2
    return min_val, max_val


def handle_error_transformation(city_error_df, item_id, order_data):
    df = order_data[order_data['item_id'] == item_id]
    min_val = df['sales_quantity'].min()
    max_val = df['sales_quantity'].max()
    new_row = [item_id,round(min_val,2),round(max_val,2),'Error occurred during transformation']
    temp_df = pd.DataFrame([new_row],columns=['item_id', 'min', 'max', 'error_reason'])
    city_error_df = pd.concat([city_error_df, temp_df], ignore_index=True)

    return city_error_df


def handle_error_modeling(city_error_df, item_id, order_data):
    df = order_data[order_data['item_id'] == item_id]
    min_val = df['sales_quantity'].min()
    max_val = df['sales_quantity'].max()
    new_row = [item_id, round(min_val, 2), round(max_val, 2), 'Error occurred during modeling']
    temp_df = pd.DataFrame([new_row], columns=['item_id', 'min', 'max', 'error_reason'])
    city_error_df = pd.concat([city_error_df, temp_df], ignore_index=True)

    return city_error_df


def final_columns_to_write(final_df):
    final_df = final_df[
        [
            "item_id",
            "min",
            "max",
            "updated_at",
        ]
    ]
    return final_df


def error_columns_to_write(error_df):
    error_df = error_df[
        [
            "item_id",
            "min",
            "max",
            "error_reason",
            "updated_at",
        ]
    ]
    return error_df


def final_columns_dtypes():
    column_dtypes = [
        {
            "name": "item_id",
            "type": "integer",
            "description": "unique identifier for item"
        },
        {
            "name": "min",
            "type": "float",
            "description": "sales outlier range minimum sales",
        },
        {
            "name": "max",
            "type": "float",
            "description": "sales outlier range maxiumum sales",
        },
        {
            "name": "updated_at",
            "type": "datetime",
            "description": "updated time stamp in IST",
        },
    ]
    return column_dtypes


def error_columns_dtypes():
    column_dtypes = [
        {
            "name": "item_id",
            "type": "integer",
            "description": "unique identifier for item"
        },
        {
            "name": "min",
            "type": "float",
            "description": "sales outlier range minimum sales",
        },
        {
            "name": "max",
            "type": "float",
            "description": "sales outlier range maxiumum sales",
        },
        {
            "name": "error_reason",
            "type": "varchar",
            "description": "reason of error",
        },
        {
            "name": "updated_at",
            "type": "datetime",
            "description": "updated time stamp in IST",
        },
    ]
    return column_dtypes

def process_city(pan_india_df):
    count = 0
    order_data = pan_india_df.copy()
    print('Total Combinations:', (len(set(order_data['item_id']))))
    order_data['Sales'] = order_data['sales_quantity']
    # print(order_data.head())
    # error_data = get_error_data(order_data)
    # print(error_data.head())
    error_data = pd.DataFrame(columns=['item_id', 'min', 'max', 'error_reason'])
    order_data = preprocess_order_data(order_data)
    # print(order_data.head())

    city_final_df = pd.DataFrame(columns=['item_id', 'min', 'max'])
    city_error_df = pd.DataFrame(columns=['item_id', 'min', 'max', 'error_reason'])

    outlier_df = pd.DataFrame(columns=['date_', 'item_id', 'item_name', 'sales_quantity', 'Sales', 'outlier_pan_item'])

    for item_id in get_unique_store_item_pairs(order_data):
        # print(item_id)
        try:
            transformed_sales = apply_boxcox(order_data, item_id)
            # print('Failed-1')
            # imputed_sales = impute_outliers(transformed_sales)
        except Exception:
            error_data = handle_error_transformation(city_error_df, item_id, order_data)
            continue
        try:
            cluster_labels = perform_clustering(transformed_sales)
            # print(transformed_sales)
            # print('Failed-2')
            cluster_means = get_cluster_means(transformed_sales, cluster_labels)
            # print(transformed_sales)
            # print('Failed-3')
            min_cluster, max_cluster = get_min_max_clusters(cluster_means)
            # print(transformed_sales)
            # print('Failed-4')
            no_outlier_cluster = get_no_outlier_cluster(cluster_means, min_cluster, max_cluster)
            # print(transformed_sales)
            # print('Failed-5')
            min_val, max_val = calculate_min_max_values(transformed_sales, cluster_labels, min_cluster, max_cluster,no_outlier_cluster)
            # print(transformed_sales)
            transformed_sales['outlier_pan_item'] = np.where(transformed_sales['cluster'] == no_outlier_cluster,0, 1)
            transformed_sales = transformed_sales[['date_', 'item_id', 'item_name', 'sales_quantity', 'Sales', 'outlier_pan_item']]
            # print(transformed_sales)
            outlier_df = pd.concat([outlier_df, transformed_sales], ignore_index=True)
            # print('Failed-6')
            new_row = [item_id,round(min_val, 2),round(max_val, 2)]
            temp_df = pd.DataFrame([new_row], columns=['item_id', 'min', 'max'])
            # print(temp_df)
            city_final_df = pd.concat([city_final_df, temp_df], ignore_index=True)
            city_final_df.to_csv('city_final_df_pan_india.csv', index=False)
            # print('Failed-8')
        except Exception:
            error_data = handle_error_modeling(error_data, item_id, order_data)

        count += 1
        print(' Processed:', count)

    city_final_df = city_final_df.drop_duplicates()
    error_data = error_data.drop_duplicates()
    outlier_df = outlier_df.drop_duplicates()

    return city_final_df, error_data, outlier_df

def process_city_parallel(pan_india_df):
    # print(city_name)
    city_final_df, city_error_df, outlier_df = process_city(pan_india_df)
    return city_final_df, city_error_df, outlier_df


def main():
    city_df = pd.read_csv('cities.csv')
    # city_df = city_df[~city_df['city_name'].isin(['Delhi'])]
    # city_df = city_df[city_df['city_name'].isin(['Bengaluru'])]

    pan_india_df = pd.DataFrame(columns=['date_','city_name','facility_id','item_id','item_name','sales_quantity','sales_value','sales_value_mrp','ipc','remove_flag'])
    final_df = pd.DataFrame(columns=['item_id', 'min', 'max'])
    error_df = pd.DataFrame(columns=['item_id', 'min', 'max', 'error_reason'])

    for city_name in list(city_df['city_name'].unique()):
        print(city_name)
        df = read_order_data(city_name)
        pan_india_df = pd.concat([pan_india_df, df], ignore_index=True)

    pan_india_df = pan_india_df.groupby(['date_','item_id','item_name']).agg({'sales_quantity':'sum','sales_value':'sum','sales_value_mrp':'sum'}).reset_index()
    final_df, error_df, outlier_df = process_city_parallel(pan_india_df)

    # # Create a multiprocessing pool
    # pool = multiprocessing.Pool()
    #
    # # Apply parallel processing to process_city_parallel function for each city name
    # results = pool.map(process_city_parallel, city_df['city_name'])
    #
    # # Close the pool and wait for all processes to finish
    # pool.close()
    # pool.join()

    # Retrieve the results
    # for city_final_df, city_error_df in results:
    #     final_df = pd.concat([final_df, city_final_df], ignore_index=True)
    #     error_df = pd.concat([error_df, city_error_df], ignore_index=True)

    final_df['item_id'] = final_df['item_id'].astype('int')
    final_df["updated_at"] = pd.to_datetime(datetime.today() + timedelta(hours=5.5))

    error_df['item_id'] = error_df['item_id'].astype('int')
    error_df["updated_at"] = pd.to_datetime(datetime.today() + timedelta(hours=5.5))

    final_df.to_csv('output_pan_item/final_df_b.csv', index=False)
    error_df.to_csv('output_pan_item/error_df_b.csv', index=False)
    outlier_df.to_csv('outlier_output_pan_item/final_df.csv', index=False)


if __name__ == '__main__':
    main()