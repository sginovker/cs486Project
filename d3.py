import pandas as pd 
import numpy as np
from surprise.model_selection import cross_validate
from math import sqrt
#data combining+formatting
BUCKET_PATH = "gs://cs486-unrecommendation-engine/"
df = pd.DataFrame()
for i in map(str, range(1, 2)):
    print(i)
    df = df.append(pd.read_csv(BUCKET_PATH + 'combined_data_{}.txt'.format(i),header = None, 
names = ['CustomerID', 'Rating'], usecols = [0,1]))
#concat all data
df['Rating'] = df['Rating'].astype(float)
print("Done with files")
# add movieid col - courtesy of https://www.kaggle.com/laowingkin/netflix-movie-recommendatio
#Data-manipulation
df_nan = pd.DataFrame(pd.isnull(df.Rating))
df_nan = df_nan[df_nan['Rating'] == True]
df_nan = df_nan.reset_index()
movie_np = []
movie_id = 1
print("before movie_np")
for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]):
    # numpy approach
    temp = np.full((1,i-j-1), movie_id)
    movie_np = np.append(movie_np, temp)
    movie_id += 1
print("movie_np")
# Account for last record and corresponding length
# numpy approach
last_record = np.full((1,len(df) - df_nan.iloc[-1, 0] - 1),movie_id)
movie_np = np.append(movie_np, last_record)
# remove those Movie ID rows
df = df[pd.notnull(df['Rating'])]
df['MovieID'] = movie_np.astype(int)
df['CustomerID'] = df['CustomerID'].astype(int)
# courtesy of https://www.kaggle.com/laowingkin/netflix-movie-recommendation#Data-manipulatio
# get rid of data that is too sparse so it actually loads rip
f=['count','mean']
quantile = 0.9
print("getting rid of sparse data")
df_movie_summary = df.groupby('MovieID')['Rating'].agg(f)
df_movie_summary.index = df_movie_summary.index.map(int)
movie_benchmark = round(df_movie_summary['count'].quantile(quantile),0)
drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index
df_cust_summary = df.groupby('CustomerID')['Rating'].agg(f)
df_cust_summary.index = df_cust_summary.index.map(int)
cust_benchmark = round(df_cust_summary['count'].quantile(quantile),0)
drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index
df = df[~df['MovieID'].isin(drop_movie_list)]
df = df[~df['CustomerID'].isin(drop_cust_list)]
print("Making nxm matrix")
# nxm matrix (customer x movie)
matrix = pd.pivot_table(df,values='Rating',index='CustomerID',columns='MovieID')
# movieid to movie title data
movieNames = pd.read_csv(BUCKET_PATH + 'movie_titles.csv',encoding = "ISO-8859-1", header = None, names = ['MovieID', 'Name'],usecols = [0,2])
movieNames.set_index('MovieID', inplace=True)
print (movieNames.head(10))
print(matrix.head(10))
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
def get_neighbours(train, test_row, k):
        distances = [(train_row, euclidean_distance(test_row, train_row)) for i,train_row in 
train.iterrows()]
        distances.sort(key=lambda tup: tup[1], reverse=True)
        neighbours = [distances[i][0] for i in range(k)]
        return neighbours
def predict(row_index=0):
    k = 10
    neighbours = get_neighbours(matrix, matrix.iloc[[row_index]], k)
    for neighbour in neighbours:
        neighbour.fillna(avgs, inplace=True)
    neighbours_combined = pd.concat(map(lambda x: x.to_frame(), neighbours))
    print(neighbours_combined)
    scores = neighbours_combined.sum(axis=1)
    print(scores)
    top_10_movies = scores.nlargest(10)
    return top_10_movies
avgs = matrix.mean(axis=0, skipna=True)
#matrix.fillna(avgs, inplace=True)
top_10_movies = predict()
print(top_10_movies)
