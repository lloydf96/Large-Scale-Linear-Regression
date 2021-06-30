from numpy import deg2rad, cos, sin, arcsin, sqrt
import numpy as np
import pandas as pd
import osmnx as ox
import dask

#get distance between two points using different distance metrics
def distance_vector(pointA,pointB,norm):
    latA,lonA = deg2rad(pointA).T.values
    latB,lonB = deg2rad(pointB).T.values
    R = 6371 # radius of the earth
    xA = R * cos(latA) * cos(lonA)
    yA = R * cos(latA) * sin(lonA)
    xB = R * cos(latB) * cos(lonB)
    yB = R * cos(latB) * sin(lonB)
    x = xB - xA
    y = yB - yA
    if norm == "norm1 cartesian":
        dist = abs(x) + abs(y)
    elif norm == "euclidean":
        dist = sqrt(x**2 + y**2)
    elif norm == "haversine" :
        dlon = lonB - lonA 
        dlat = latB - latA 
        a = sin(dlat/2)**2 + cos(latB) * cos(latA) * sin(dlon/2)**2
        c = 2 * arcsin(sqrt(a)) 
        r = 6371 
        dist = c*r
    elif norm == "norm1":
        dlon = lonB - lonA 
        dlat = latB - latA 
        dist = abs(dlon) + abs(dlat)
        
    return dist
   
#remove points outside of newyork bounding box(imported from OSM)
def remove_geo_outliers(df,city):
    
    pickup_mask = (df['pickup_latitude']>= city.bbox_south[0]) & (df['pickup_latitude']<= city.bbox_north[0]) \
    & (df['pickup_longitude']<= city.bbox_east[0]) & (df['pickup_longitude']>= city.bbox_west[0])
    dropoff_mask = (df['dropoff_latitude']>= city.bbox_south[0]) & (df['dropoff_latitude']<= city.bbox_north[0]) \
    & (df['dropoff_longitude']<= city.bbox_east[0]) & (df['dropoff_longitude']>= city.bbox_west[0])
    
    df = df[pickup_mask & dropoff_mask]
    return df

#get pickup time features
def add_time_features(df):
    df["pickup_hour"],df["pickup_day"],df["pickup_week"],df["pickup_month"],df["pickup_Year"] = df.pickup_datetime.dt.hour,df.pickup_datetime.dt.day \
    ,df.pickup_datetime.dt.isocalendar().week,df.pickup_datetime.dt.month,df.pickup_datetime.dt.year
    return df
 
#divide newyork bounding box into smaller squares with square dimensions = norm1_limit, allocate pickup and dropoff area to their respective small squares
def add_area_data(df,city,norm1_limit):
    min_lat = city.bbox_south[0]
    min_lon = city.bbox_west[0]
    
    df["pickup_area"] = (((df["pickup_latitude"] - min_lat)/norm1_limit).astype(int))*10000 + ((df["pickup_longitude"] - min_lon)/norm1_limit).astype(int)
    df["dropoff_area"] = (((df["dropoff_latitude"] - min_lat)/norm1_limit).astype(int))*10000 + ((df["dropoff_longitude"] - min_lon)/norm1_limit).astype(int)
    
    return df

#generate dict with the area dvisions along with lat and lon of its center
def generate_area_dict(df,city,norm1_limit):
    min_lat = city.bbox_south[0]
    min_lon = city.bbox_west[0]
    dict_area = [(area,(int(area/10000)*norm1_limit + min_lat + 0.5*norm1_limit , norm1_limit*(area - int(area/10000)*10000) + min_lon + 0.5*norm1_limit)) for     i,area in enumerate(df)]
    return dict(dict_area)

#store segregated area in kd-tree
def get_area_tree(df,city,norm1_limit):
    df = add_area_data(df,city,norm1_limit)
    #store area data as kd-tree
    area_sample_size = 200000 if df.shape[0]> 200000 else df.shape[0] 
    df_sample = df.sample(n = area_sample_size)
    pickup_area_dict = pd.DataFrame.from_dict(generate_area_dict(df_sample.pickup_area,city,norm1_limit),columns = ['lat','lon'],orient = 'index')
    dropoff_area_dict = pd.DataFrame.from_dict(generate_area_dict(df_sample.dropoff_area,city,norm1_limit),columns = ['lat','lon'],orient = 'index')
    from scipy import spatial
    dropoff_tree = spatial.KDTree(dropoff_area_dict[['lat','lon']])
    pickup_tree = spatial.KDTree(pickup_area_dict[['lat','lon']])
    
    return pickup_area_dict,dropoff_area_dict,pickup_tree,dropoff_tree

#find area division for other training and testing data using data stored in kd-tree
def find_area(df,pickup_area_dict,dropoff_area_dict,city,pickup_tree,dropoff_tree,norm1_limit):
    df = add_area_data(df,city,norm1_limit)
    
    df=pd.merge(df,pickup_area_dict.reset_index().loc[:,"index"], left_on=["pickup_area"],suffixes= (None,'_pickup'), right_on = ['index'] , how = 'left')
    df=pd.merge(df, dropoff_area_dict.reset_index().loc[:,"index"], left_on=["dropoff_area"], suffixes= (None,'_dropoff'),right_on = ['index'] , how = 'left')

    df.drop(['pickup_area','dropoff_area'],axis = 1,inplace = True)
    df.rename(columns={'index': 'pickup_area' , 'index_dropoff': 'dropoff_area'},inplace = True)
    
    df_pickup_area_cat = df.loc[df.pickup_area.isna(),["pickup_latitude","pickup_longitude"]]
    df_dropoff_area_cat = df.loc[df.dropoff_area.isna(),["dropoff_latitude","dropoff_longitude"]]
    
    if df_pickup_area_cat.shape[0] ==0 or df_dropoff_area_cat.shape[0] == 0:
        return df
    
    _,df_pickup_area_cat["pickup_area_index"] = pickup_tree.query(list(zip(df_pickup_area_cat["pickup_latitude"],df_pickup_area_cat["pickup_longitude"])),1)
    _,df_dropoff_area_cat["dropoff_area_index"] = dropoff_tree.query(list(zip(df_dropoff_area_cat["dropoff_latitude"],df_dropoff_area_cat["dropoff_longitude"])),1)
    
    df_pickup_area_cat["pickup_area"] = pickup_area_dict.index.values[[df_pickup_area_cat["pickup_area_index"]]]
    df_dropoff_area_cat["dropoff_area"] = dropoff_area_dict.index.values[[df_dropoff_area_cat["dropoff_area_index"]]]
    df.update(df_pickup_area_cat["pickup_area"].astype(int))
    df.update(df_dropoff_area_cat["dropoff_area"].astype(int))
    df["pickup_area"] = df["pickup_area"].astype(int)
    df["dropoff_area"] = df["dropoff_area"].astype(int)
    return df

#add area weather and time feature columns
def add_features(df,city,norm1_limit,pickup_area_dict,dropoff_area_dict,pickup_tree,dropoff_tree):

    df = df.dropna()
    
    #remove points outside geography
    df = remove_geo_outliers(df,city)
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"],utc = True)
    #add time features
    df = add_time_features(df)
    
    #read and add weather data
    weather_data = pd.read_pickle("weather_data.pkl")
    weather_data.index = pd.to_datetime(weather_data.index,utc = True)
    df["pickup_datetime_round"] = df.pickup_datetime.round('H')
    df=pd.merge(df, weather_data, left_on=["pickup_datetime_round"], right_index =True , how = 'left')
    df.drop(['Unnamed: 0'],axis = 1,inplace = True)
    df = df.drop(['humid'],axis = 1)
    
    #add distance metric
    pointA= df.loc[:,["pickup_longitude","pickup_latitude"]]
    pointB= df.loc[:,["dropoff_longitude","dropoff_latitude"]]
    df["h_distance"] = distance_vector(pointA,pointB,"haversine")
    df["norm1_cartesian"] = distance_vector(pointA,pointB,"norm1 cartesian")
    df = df.dropna()

    df.drop( df.index[df["norm1"] == 0],inplace = True,axis = 0)
    df.reset_index(inplace = True,drop = True)
    df = find_area(df,pickup_area_dict,dropoff_area_dict,city,pickup_tree,dropoff_tree,norm1_limit) 

    return df

def create_templates(train):

    X_pickup_area_hc =pd.get_dummies(train.pickup_area,prefix="pickup",dtype = 'uint8').fillna(0)
    X_dropoff_area_hc = pd.get_dummies(train.dropoff_area,prefix = "dropoff",dtype = 'uint8').fillna(0)
    X_template = pd.concat([X_pickup_area_hc,X_dropoff_area_hc],axis = 1)
    
    X_weather = pd.get_dummies(list(range(1,37)),prefix = 'weather_cat',dtype = 'uint8')[0:0]
    
    X_hour = pd.get_dummies(list(range(0,24)),prefix = 'hour',dtype = 'uint8')[0:0]
    X_day = pd.get_dummies(list(range(1,32)),prefix = 'day',dtype = 'uint8')[0:0]
    X_month = pd.get_dummies(list(range(1,13)),prefix = 'month',dtype = 'uint8')[0:0]
    X_year = pd.get_dummies(list(range(2012,2018)),prefix = 'year',dtype = 'uint8')[0:0]
    X_time = pd.concat([X_hour,X_day,X_month,X_year],axis = 1)
    return X_template[0:0],X_weather,X_time
        
def area_cat_features(train,X_area_template):
    
    X_pickup_area_hc = pd.get_dummies(train.pickup_area,prefix="pickup",dtype = 'uint8').fillna(0)

    X_dropoff_area_hc = pd.get_dummies(train.dropoff_area,prefix = "dropoff",dtype = 'uint8').fillna(0)

    X_area_hc = X_area_template[0:0]
    X_area_hc = X_area_hc.add(pd.concat([X_pickup_area_hc,X_dropoff_area_hc],axis = 1),fill_value = 0).fillna(0).astype('uint8')
    
    return X_area_hc #,test_area_hc
 
def weather_cat_features(train,weather_template):
    
    X_weather = weather_template.add(pd.get_dummies(train.weather_cat,prefix = 'weather_cat',dtype = 'uint8'),fill_value = 0).fillna(0).astype('uint8')

    return X_weather
      
def time_cat_features(train,time_template):

    time_template = time_template.add(pd.get_dummies(train.pickup_hour,prefix = 'hour',dtype = 'uint8'),fill_value = 0).fillna(0).astype('uint8')
    time_template = time_template.add(pd.get_dummies(train.pickup_day,prefix = 'day',dtype = 'uint8'),fill_value = 0).fillna(0).astype('uint8')
    time_template = time_template.add(pd.get_dummies(train.pickup_month,prefix = 'month',dtype = 'uint8'),fill_value = 0).fillna(0).astype('uint8')
    time_template = time_template.add(pd.get_dummies(train.pickup_Year,prefix = 'year',dtype = 'uint8'),fill_value = 0).fillna(0).astype('uint8')

    return time_template #,test_time

#categorical features as one hot encoded vectors generated for area, weather category and time features.
def lr_features(train,X_template,weather_template,time_template,area=False,weather=False,time=False ,col_list = None,merge_xy = False,delay = True):
    X = train
    if area:
        X_area_hc = area_cat_features(train,X_template)
        X = pd.concat([X,X_area_hc],axis = 1)
        print(1)
        
    if weather:
        X_weather= weather_cat_features(train,weather_template)
        X = pd.concat([X,X_weather],axis = 1) 
        
    if time:
        X_time= time_cat_features(train,time_template)
        X = pd.concat([X,X_time],axis = 1)
    
    X = X.drop(['wind_drec','wind_speed','weather_cat','pickup_datetime','pickup_day','pickup_week','pickup_month','pickup_Year','pickup_hour','weather_desc','pickup_datetime_round','pickup_area','dropoff_area'],axis = 1)
    
    X = X.dropna()
    
    Y = X.loc[:,'fare_amount']
    X = X.drop(['fare_amount'], axis = 1)
    

    if col_list == None:
        
        col_list = X.columns.tolist()
        
    X = X[col_list]
    if merge_xy:
        X=pd.concat([Y,X],axis = 1)
    if delay:
        return X
    else :
        return X,Y,col_list
