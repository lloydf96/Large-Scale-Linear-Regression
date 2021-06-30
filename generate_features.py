import os
import dask.bag as db
import dask
import dask.dataframe as ddf
import pandas as pd
import osmnx as ox
import random
import matplotlib.pyplot as plt
from dask.distributed import Client, progress
import features as f
from dask_ml.preprocessing import StandardScaler
import dask.dataframe as ddf
from dask.distributed import Client, progress

global X_template,weather_template,time_template,area,weather,time,col_list
#dask client initialisation

def partition_dataset(path,file_name,blocksize = 10e6):
#partition dataset
    df =ddf.read_csv(file_name,blocksize = blocksize)
    df = df.drop(['Unnamed: 0', 'Unnamed: 0.1','key'], axis=1)
    df.to_csv(path + "\\Data\\train_*.csv")

def partition_feature_dataset(path,city,norm_limit,pickup_area_dict,dropoff_area_dict,pickup_tree,dropoff_tree):
    task_lists = []
    for i in range(0,272):
        lazy_task_1 = dask.delayed(pd.read_csv)(path + "\\Data\\train_"+str(i).zfill(3)+".csv")
        lazy_task_2 = dask.delayed(f.add_features)(lazy_task_1,city,norm_limit,pickup_area_dict,dropoff_area_dict,pickup_tree,dropoff_tree)
        lazy_task_3 = dask.delayed(lazy_task_2.to_csv)(path + "\\Data\\feature_"+str(i).zfill(3)+".csv")
        task_lists.append(lazy_task_3)

    client.persist(task_lists)

def add_cat_features(i,path,X_template,weather_template,time_template,col_list):
    lazy_task_1 = pd.read_csv(path + "\\Data\\feature_"+str(i).zfill(3)+".csv")
    lazy_task_X = f.lr_features(lazy_task_1,X_template,weather_template,time_template,col_list=col_list,area=True,weather=True,time = True,merge_xy=True,delay = True)
    lazy_task_3 =lazy_task_X.to_csv(path+ "\\Data\\X_"+str(i).zfill(3)+".csv")
