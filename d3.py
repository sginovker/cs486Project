import pandas as pd 
import numpy as np
from surprise.model_selection import cross_validate
from math import sqrt

#data combining+formatting

df1 = pd.read_csv('combined_data1.txt',header = None, names = ['CustomerID', 'Rating'], usecols = [0,1])
df1['Rating'] = df1['Rating'].astype(float)
df2 = pd.read_csv('combined_data2.txt',header = None, names = ['CustomerID', 'Rating'], usecols = [0,1])
df2['Rating'] = df2['Rating'].astype(float)
df3 = pd.read_csv('combined_data3.txt',header = None, names = ['CustomerID', 'Rating'], usecols = [0,1])
df3['Rating'] = df3['Rating'].astype(float)
df4 = pd.read_csv('combined_data4.txt',header = None, names = ['CustomerID', 'Rating'], usecols = [0,1])
df4['Rating'] = df4['Rating'].astype(float)

frames = [df1,df2,df3,df4]
#concat all data
df = pd.concat(frames)

# add movieid col - courtesy of https://www.kaggle.com/laowingkin/netflix-movie-recommendation#Data-manipulation
df_nan = pd.DataFrame(pd.isnull(df.Rating))
df_nan = df_nan[df_nan['Rating'] == True]
df_nan = df_nan.reset_index()

movie_np = []
movie_id = 1

for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]):
    # numpy approach
    temp = np.full((1,i-j-1), movie_id)
    movie_np = np.append(movie_np, temp)
    movie_id += 1

# Account for last record and corresponding length
# numpy approach
last_record = np.full((1,len(df) - df_nan.iloc[-1, 0] - 1),movie_id)
movie_np = np.append(movie_np, last_record)

# remove those Movie ID rows
df = df[pd.notnull(df['Rating'])]

df['MovieID'] = movie_np.astype(int)
df['CustomerID'] = df['CustomerID'].astype(int)

# courtesy of https://www.kaggle.com/laowingkin/netflix-movie-recommendation#Data-manipulation
# get rid of data that is too sparse so it actually loads rip
f=['count','mean']

df_movie_summary = df.groupby('MovieID')['Rating'].agg(f)
df_movie_summary.index = df_movie_summary.index.map(int)
movie_benchmark = round(df_movie_summary['count'].quantile(0.7),0)
drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

df_cust_summary = df.groupby('CustomerID')['Rating'].agg(f)
df_cust_summary.index = df_cust_summary.index.map(int)
cust_benchmark = round(df_cust_summary['count'].quantile(0.7),0)
drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

df = df[~df['MovieID'].isin(drop_movie_list)]
df = df[~df['CustomerID'].isin(drop_cust_list)]

# nxm matrix (customer x movie)
matrix = pd.pivot_table(df,values='Rating',index='CustomerID',columns='MovieID')

# movieid to movie title data
movieNames = pd.read_csv('movie_titles.csv',encoding = "ISO-8859-1", header = None, names = ['MovieID', 'Name'],usecols = [0,2])
movieNames.set_index('MovieID', inplace=True)

print (movieNames.head(10))

#TODO:add col to signify if user has rated given movie 
#TODO:substitute any nulls with mean 

#TODO:do 5-fold cross validation


#copy pasta starts HERE
# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Locate the most DISsimilar neighbors
#I have modified this by adding "reverse=true"
def get_neighbors(train, test_row, k):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1], reverse=True)
	neighbors = list()
	for i in range(k):
		neighbors.append(distances[i][0])
	return neighbors

