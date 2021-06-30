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

def SGD_Grad(X,Y,W):
    rows,col = X.shape
    X["bias"] = np.ones(rows)
    return X.T @ (X @ W - Y) / rows

def SGD_Update(X,Y,W,lr):
    grad = SGD_Grad(X,Y,W)
    step = lr*grad
    W -= step
    return W

def SGD_Batch(train_batches,W,scaler,lr,eta):
    loss = 0
    train_size = len(train_batches)
    i=0
    #df = ddf.read_csv(train_batches)
    #npartitions = df.npartitions
    #print("npartitions = ",npartitions)
    #while(i<npartitions):
    for train_file in train_batches:
        df_part = pd.read_csv(train_file)#df.get_partition(i).compute()
        X = df_part.iloc[:,3:]

        Y = df_part.iloc[:,2]
        X = scaler.transform(X)
        rows,col = X.shape

        W= SGD_Update(X,Y,W,lr)
        lr = lr*eta
        loss += np.sum(np.square(X @ W - Y))/(2*rows)
        i+=1
 
    return (np.array(W),loss/len(train_batches))

def read_csv_to_array(train_list):
    df = ddf.read_csv(train_list)
   
    return df
    
def test_score(test_batches,W,scaler):
    nrows = 0
    loss = 0
    for test_name in test_batches:
        df = pd.read_csv(test_name)
        X = df.iloc[:,3:]

        Y = df.iloc[:,2]
        X = scaler.transform(X)
        rows,columns = X.shape
        X["bias"] = np.ones(rows)
        loss +=  np.sum(np.square(X @ W - Y))/2
        nrows += rows
    
    net_error = (loss/nrows )**0.5
    return net_error