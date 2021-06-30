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
from generate_features import *
from sgd import *

global X_template,weather_template,time_template,area,weather,time,col_list
#dask client initialisation

def generate_features():
    client = Client(n_workers=1, threads_per_worker=10, memory_limit='10GB')

    path = os.getcwd()
    partition_dataset(path,file_name,blocksize = 10e6)
    city = ox.geocode_to_gdf('New York City,New York')
    norm_limit = 0.05

    df = pd.read_csv("Data/train_000.csv").dropna().sample(40000)
    pickup_area_dict, dropoff_area_dict ,pickup_tree, dropoff_tree = f.get_area_tree(df,city,norm_limit)

    #creates empty dataframe with requisite columns
    df_struct = pd.read_csv("Data/train_000.csv").sample(40000)
    df_struct = f.add_features(df_struct,city,norm_limit,pickup_area_dict,dropoff_area_dict,pickup_tree,dropoff_tree)

    task_lists = []
    #divide and store area information based on subset of training sample

    X_template,weather_template,time_template = f.create_templates(df_struct)
    df_struct_sample = df_struct.sample(1000)
    _,_,col_list = f.lr_features(df_struct_sample,X_template,weather_template,time_template,area=True,weather=True,time = True,col_list = None,delay=False)
    
    #get features, split and save it to smaller files
    partition_feature_dataset(path,city,norm_limit,pickup_area_dict,dropoff_area_dict,pickup_tree,dropoff_tree)
    
    client.close()

    client = Client(n_workers=2, threads_per_worker=3, memory_limit='6GB')
    client

    #add categorical data

    for i in range(0,272):
        lazy_task = dask.delayed(add_cat_features)(i,path,X_template,weather_template,time_template,col_list)
        #lazy_task_4 = dask.delayed(lazy_task_Y.to_csv)(path + "\\Data\\Y_"+str(i).zfill(3)+".csv")
        task_lists.append(lazy_task)

    client.compute(task_lists)
    
def fit_sgd():
    client.close()
    client = Client(n_workers=5, threads_per_worker=1, memory_limit='3GB')

    #test train set
    mini_batch = [path+"\\Data\\databag\\X_"+str(i).zfill(3)+".csv" for i in range(0,200)]
    random.shuffle(mini_batch)
    split_ratio = 0.9
    no_of_batches = len(mini_batch)
    training_batch_cutoff = int(0.9*no_of_batches)
    train_batches = mini_batch[0:training_batch_cutoff]
    test_batches = mini_batch[training_batch_cutoff:]
    task_lists = []

    train_set =ddf.read_csv(train_batches,blocksize = 5e6)
    X_train, y_train = train_set.iloc[:,3:],train_set.iloc[:,2]
    scaler = StandardScaler()
    scaler.fit(X_train)

    lr = 0.09
    eta = 1
    eta1 = 0.95
    epochs = 100
    tol = 0.1
    _,no_of_features = X_train.shape
    W = np.zeros(no_of_features+1)
    loss_list = [(0,0)]
    try_count = 0
    avg_loss = 0
    read_no = 20
    core = 5
    train_batch_per_core = int(len(train_batches)/core) +1

    for i in range(epochs): 

        random.shuffle(train_batches)

        old_W = W
        old_avg_loss = avg_loss
        task_list = []

        for i in range(core):
            train_batch_for_core = train_batches[i*train_batch_per_core:(i+1)*train_batch_per_core]
            task_1 = dask.delayed(SGD_Batch)(train_batch_for_core,W,scaler,lr,eta)
            task_list.append(task_1)

        W_futures = dask.persist(*task_list)
        W_from_core = dask.compute(*W_futures)
        #W_from_core = [(task.compute()) for task in task_list]
        W,avg_loss = zip(*W_from_core)
        W = np.mean(np.array(W),0)
        avg_loss = np.mean(np.array(avg_loss))
        loss_list.append((i,avg_loss))
        lr = lr*eta1
        print(avg_loss)
        if np.all(np.abs(avg_loss - old_avg_loss) <= tol):
            try_count +=1

        if try_count > 3:
            break
  
    
if __name__ == "__main__":
    
    #generate features
    generate_features()
    
    #fit a lr model using mini-batch sgd
    fit_sgd()
    
   