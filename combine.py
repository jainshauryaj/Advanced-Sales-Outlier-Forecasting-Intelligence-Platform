import pencilbox as pb
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

# ---------------------------------------- CODE START --------------------------------------------------------------

store_item_df = pd.read_csv('outlier_output_store_item/final_df.csv')
store_item_df = store_item_df.drop(columns=['outlier_city_item'])
# store_item_df = store_item_df[['city_name','facility_id','item_id','min','max']].rename(columns = {'min':'store_item_min','max':'store_item_max'})
# print(store_item_df.head(1))
city_item_df = pd.read_csv('outlier_output_city_item/final_df.csv')
city_item_df = city_item_df.rename(columns={'sales_quantity':'sales_quantity_city','Sales':'Sales_city'})
# city_item_df = city_item_df[['city_name','item_id','min','max']].rename(columns = {'min':'city_item_min','max':'city_item_max'})
# print(city_item_df.head(1))
pan_india_item_df = pd.read_csv('outlier_output_pan_item/final_df.csv')
pan_india_item_df = pan_india_item_df.rename(columns={'sales_quantity':'sales_quantity_pan','Sales':'Sales_pan'})
# pan_item_df = pan_item_df[['item_id','min','max']].rename(columns = {'min':'pan_item_min','max':'pan_item_max'})
# print(pan_india_item_df.head(1))

final_df = store_item_df.merge(city_item_df, on=['date_','city_name','item_id','item_name'], how = 'left')
# print(final_df.head())
#
final_df = final_df.merge(pan_india_item_df, on=['date_','item_id','item_name'], how = 'left')
# print(final_df.head())
#
final_df['outlier_flag'] = np.where((final_df['outlier_store_item'] == 1) & (final_df['outlier_city_item'] == 0) & (final_df['outlier_pan_item'] == 0), 1, 0)
# print(final_df.head())

# Event Logic

query = f"""
SELECT event_date as date_, event_name FROM dwh.bl_calendar_events_holidays
"""
event_df = read_sql_query(query, CON_REDSHIFT)
event_df['date_'] = pd.to_datetime(event_df['date_'])
print(event_df.head())

item_list = tuple(final_df['item_id'].unique())
print(item_list)
#
# query = f"""
# SELECT
#     item_id, pt.name as p_type
# FROM
#     lake_rpc.item_product_mapping ipm
# LEFT JOIN
#     (select id, type_id from lake_cms.gr_product) gp on gp.id = ipm.product_id
# LEFT JOIN
#     (select id, name from lake_cms.gr_product_type) pt on pt.id = gp.type_id
# WHERE
#     ipm.item_id IN {item_list}
# """
# item_ptype_mapping = read_sql_query(query, CON_REDSHIFT)
# print(item_ptype_mapping.head())
#
# final_df = final_df.merge(item_ptype_mapping, on = ['item_id'], how = 'left')
# print(final_df.head())
#
# temp_df = final_df.groupby(['date_','city_name','facility_id','p_type']).agg({'sales_quantity':'sum'}).reset_index()
# print(temp_df.head())
#
# temp_df_overall = temp_df.groupby(['city_name','facility_id','p_type']).agg({'sales_quantity':'mean'}).reset_index().rename(columns={'sales_quantity':'3m_cpd'})
# print(temp_df_overall.head())
#
# temp_df = temp_df.merge(event_df, on =['date_'], how = 'left')
# temp_df = temp_df.merge(temp_df_overall, on =['city_name','facility_id','p_type'], how = 'left')
# temp_df['3m_cpd'] = 1.5 * event_ptype_df['3m_cpd']
# print(temp_df.head())
#
# final_df = final_df.merge(temp_df[['date_','city_name','facility_id','p_type','event_name','3m_cpd']], on = ['date_','city_name','facility_id','p_type'], how = 'left')
# print(final_df.head())
#
# final_df['event_flag'] = np.where((final_df['sales_quantity'] >= final_df['3m_cpd']) & (final_df['outlier_flag'] == 1), 1, 0)
# print(final_df.head())
#
# final_df = final_df.drop(columns={'event_name','3m_cpd'})
# print(final_df.head())

# # Seasonality Logic

# query = f"""
# SELECT item_id, l2_id, l2 FROM lake_rpc.item_category_details WHERE item_id IN {item_list}
# """
# item_l2_df = read_sql_query(query, CON_REDSHIFT)
# item_l2_df = item_l2_df[['item_id','l2_id']].rename(columns={'l2_id':'l2_category_id'})
# print(item_l2_df.head())

# query = f"""
# DROP TABLE IF EXISTS keyword_pid;
#     CREATE temporary table keyword_pid as (
#     select
#         trim(lower(properties__search_actual_keyword))::varchar as keyword,
#         properties__product_id as product_id,
#         dmm.facility_id,
#         l2_category_id,
#         at_date_ist as date_,
#         count (device_uuid) as key_prod_atc
#     from
#         spectrum.mobile_event_data k
#         inner join dwh.dim_product p
#             on k.properties__product_id = p.product_id
#             and is_current = true
#             and is_product_enabled = true
#         join dwh.dim_merchant_outlet_facility_mapping dmm
#             on dmm.frontend_merchant_id = traits__merchant_id
#             and is_express_store = True
#             and dmm.is_current = True
#     where
#         at_date_ist between '2023-07-09' and '2023-07-13'
#         and name = 'Product Added'
#         and properties__search_actual_keyword is not null
#         and properties__search_actual_keyword not in ('-NA-', '#-NA', '')
#         and properties__search_keyword_parent in ('type-to-search', 'type_to_search')
#         and traits__user_id is not NULL
#         and traits__user_id not in (-1,0)
#         and traits__user_id not in ('14647274', '9961423','9709403','13957980','13605597','3927621','14740725','4144617','10045662')
#         and traits__city_id is not NULL
#         and traits__merchant_id is not NULL
#         and traits__merchant_name is not NULL
#         and keyword not in ('',' ','#-na')
#         and len(keyword)>2
#     group by 1,2,3,4,5
#     );
#
#     DROP TABLE IF EXISTS search;
#     CREATE TEMPORARY TABLE search as(
#     select
#         at_date_ist as date_,
#         dmm.facility_id,
#         traits__merchant_id,
#         lower(device_uuid) device_uuid,
#         trim(lower(properties__search_actual_keyword))::varchar as keyword
#     from
#         spectrum.mobile_event_data
#     join dwh.dim_merchant_outlet_facility_mapping dmm
#         on dmm.frontend_merchant_id = traits__merchant_id
#         and is_express_store = True
#         and dmm.is_current = True
#     where
#         at_date_ist between '2023-07-09' and '2023-07-13'
#         and name = 'Product Searched'
#         and properties__search_actual_keyword is not null
#         and properties__search_actual_keyword not in ('-NA-', '#-NA', '')
#         and properties__search_keyword_parent in ('type-to-search', 'type_to_search')
#         and traits__user_id is not NULL
#         and traits__user_id not in (-1,0)
#         and traits__user_id not in ('14647274', '9961423','9709403','13957980','13605597','3927621','14740725','4144617','10045662')
#         and traits__city_id is not NULL
#         and traits__merchant_id is not NULL
#         and traits__merchant_name is not NULL
#         and keyword not in ('',' ','#-na')
#         and len(keyword)>2
#
#     group by 1,2,3,4,5
#
#     );
#
#     with keyword_atc as (
#     select
#         keyword keyword1,
#         sum(key_prod_atc) as key_tot_atc
#     from keyword_pid
#     group by 1
#     having key_tot_atc >= 100
#     ),
#
#     l0_atc as (
#     select
#         l2_category_id,
#         sum(key_prod_atc) as l0_tot_atc
#     from
#         keyword_pid k
#     group by 1
#     ),
#
#     keyword_l0 as(
#     select
#         k.keyword,
#         key_tot_atc,
#         k.l2_category_id,
#         l0_tot_atc,
#         sum(key_prod_atc) as key_l0_atc,
#         (key_l0_atc*100/key_tot_atc) as l0_in_key,
#         (key_l0_atc*100.00/l0_tot_atc) as key_in_l0,
#         row_number () over (partition by k.l2_category_id order by key_l0_atc desc) as keyword_rank_in_l0
#     from
#         keyword_pid k
#         inner join keyword_atc a on k.keyword = a.keyword1
#         inner join l0_atc l on l.l2_category_id=k.l2_category_id
#     group by 1,2,3,4
#     having key_l0_atc > 10
#     )
#     ,
#
#     pre_final as (
#     select
#         l2_category_id,
#         keyword
#     from
#         keyword_l0
#     where
#         key_in_l0>=0.1
#         and l0_in_key >=10
#     group by 1,2
#     order by l2_category_id
#     )
#
#     SELECT
#         s.date_,
#         cl.name as city_name,
#         s.facility_id,
#         l2_category_id,
#         s.traits__merchant_id as merchant_id,
#         count (s.device_uuid) as searches
#     FROM
#         search s
#     JOIN
#         pre_final u on u.keyword = s.keyword
#     INNER JOIN
#         lake_retail.console_outlet co ON co.facility_id = s.facility_id
#     INNER JOIN
#         lake_retail.console_location cl ON co.tax_location_id = cl.id
#     group by 1,2,3,4,5
# """
# search_l2_df = read_sql_query(query, CON_REDSHIFT)
# print(search_l2_df.head())
#
# search_l2_df = search_l2_df.merge(item_l2_df, on=['l2_category_id'], how = 'left')
# print(search_l2_df.head())
#
# search_store_l2_df = search_l2_df.groupby(['facility_id','item_id']).agg({'searches':'mean'}).rename(columns={'searches':'store_searches'}).reset_index()
# print(search_store_l2_df.head())
#
# search_city_l2_df = search_l2_df.groupby(['city_name','item_id']).agg({'searches':'mean'}).rename(columns={'searches':'city_searches'}).reset_index()
# print(search_city_l2_df.head())
#
# final_df = final_df.merge(search_store_l2_df, on=['facility_id','item_id'], how = 'left')
# print(final_df.head())
#
# final_df = final_df.merge(search_city_l2_df, on=['city_name','item_id'], how = 'left')
# print(final_df.head())
#
# final_df['store_seasonality_calc'] = ((0.3 * final_df['sales_quantity']) + (0.7 * final_df['store_searches']))
# print(final_df.head())
#
# final_df['city_seasonality_calc'] = ((0.3 * final_df['sales_quantity']) + (0.7 * final_df['city_searches']))
# print(final_df.head())
#
# final_df['store_seasonality_flag'] = np.where(final_df['store_searches'] > final_df['store_seasonality_calc'], 1, 0)
# print(final_df.head())
#
# final_df['city_seasonality_flag'] = np.where(final_df['city_searches'] > final_df['city_seasonality_calc'], 1, 0)
# print(final_df.head())
#
# final_df['seasonality_flag'] = np.where((final_df['store_seasonality_flag'] == 1) & (final_df['city_seasonality_flag'] == 1) & (final_df['outlier_flag'] == 1), 1, 0)
# print(final_df.head())

# # HFS Logic
#
# non_hfs_df = final_df[~final_df['day'].isin([1,2,3,4,5,6,7])]
# hfs_df = final_df[final_df['day'].isin([1,2,3,4,5,6,7])]
#
# non_hfs_mean = non_hfs_df['sales_quantity'].mean()
# hfs_mean = hfs_df['sales_quantity'].mean()
# hfs_lift = hfs_mean/non_hfs_mean
# print(hfs_lift)
#
# hfs_bump_value = hfs_lift * (final_df[~final_df['day'].isin([1,2,3,4,5,6,7])]['sales_quantity'].mean())
# print(hfs_bump_value)
#
# final_df['hfs_flag'] = np.where((final_df['day'].isin([1,2,3,4,5,6,7])) & (final_df['sales_quantity'] >= hfs_bump_value) & (final_df['outlier_flag'] == 1), 1, 0)
# print(final_df[final_df['hfs_flag'] == 1].head())

# # Bulk Buying Logic
#
# ipc_non_outlier_mean = final_df[final_df['outlier_flag'] == 0]['sales_quantity'].mean()
# ipc_max_outlier_mean = final_df[final_df['outlier_flag'] == 1]['sales_quantity'].mean()
# ipc_lift = (ipc_max_outlier_mean/ ipc_non_outlier_mean)
# print(ipc_lift)
#
# bulk_buying_value = ipc_lift * (final_df[final_df['outlier_flag'] == 0]['ipc'].mean())
# final_df['bulk_buying_flag'] = np.where((final_df['ipc'] >= bulk_buying_value) & (final_df['outlier_flag'] == 1), 1, 0)
# print(final_df[final_df['bulk_buying_flag'] == 1].head())

# # Surge Charges Logic
#
# query = f"""
# SELECT
#     date_,
#     facility_id,
#     SUM(surge_charges)* 1.00/COUNT(DISTINCT cart_id) as surge_cart_percentage
# FROM
#     (SELECT
#         DISTINCT oid.cart_id,
#         DATE(oid.cart_checkout_ts_ist) AS date_,
#         rco.facility_id,
#         CASE WHEN oid.slot_charges > 0 THEN 1 ELSE 0 END AS surge_charges
#     FROM
#         dwh.fact_sales_order_details oid
#     JOIN
#         dwh.fact_sales_invoice_item_details fsiid ON fsiid.cart_id = oid.cart_id
#     JOIN
#         lake_retail.console_outlet rco ON rco.id = oid.outlet_id AND business_type_id IN (7) AND active = 1
#     WHERE
#         (oid.cart_checkout_ts_ist BETWEEN (CURRENT_DATE - interval '7' day) AND (CURRENT_DATE))
#         AND oid.is_internal_order = FALSE
#         AND (oid.order_type NOT LIKE '%%internal%%' OR oid.order_type IS NULL)
#         AND oid.total_procured_quantity > 0
#         AND oid.order_current_status = 'DELIVERED')
# GROUP BY
#     1,2
# """
# surge_df = read_sql_query(query, CON_REDSHIFT)
# surge_df['date_'] = pd.to_datetime(surge_df['date_'])
# overall_surge_df = surge_df.groupby(['facility_id']).agg('surge_cart_percentage':'mean').rename(columns={'surge_cart_percentage':'overall_surge_cart_percentage'}).reset_index()
# surge_df = surge_df.merge(overall_surge_df, on = ['facility_id'], how = 'left')
# surge_df['surge_flag'] = np.where(surge_df['surge_cart_percentage'] > surge_df['overall_surge_cart_percentage'], 1, 0)
# surge_df = surge_df[['date_','facility_id','surge_flag']]
# surge_df['surge_flag'] = surge_df['surge_flag'].fillna(0)
#
# final_df = final_df.merge(surge_df, on = ['date_','facility_id'], how = 'left')
# print(final_df.head())



