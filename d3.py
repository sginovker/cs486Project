import pandas as pd
import numpy as np
from surprise.model_selection import cross_validate
from math import sqrt
from collections import defaultdict

BUCKET_PATH = "gs://cs486-unrecommendation-engine/"
df = pd.DataFrame()
for i in map(str, range(1, 2)):
    print(i)
    df = df.append(pd.read_csv(BUCKET_PATH + 'combined_data_{}.txt'.format(i),header = None, 
names = ['CustomerID', 'Rating'], usecols = [0,1]))
df['Rating'] = df['Rating'].astype(float)
print("Done with files")

# add movieid col - courtesy of https://www.kaggle.com/laowingkin/netflix-movie-recommendation
df_nan = pd.DataFrame(pd.isnull(df.Rating))
df_nan = df_nan[df_nan['Rating'] == True]
df_nan = df_nan.reset_index()
movie_np = []
movie_id = 1
print("before movie_np")

for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]):
    temp = np.full((1,i-j-1), movie_id)
    movie_np = np.append(movie_np, temp)
    movie_id += 1

print("movie_np")
last_record = np.full((1,len(df) - df_nan.iloc[-1, 0] - 1),movie_id)
movie_np = np.append(movie_np, last_record)

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
print(len(df))
user_ratings = defaultdict(dict)
for i, row in df.iterrows():
    user_ratings[row.CustomerID][row.MovieID] = row.Rating

print("Making nxm matrix")

matrix = pd.pivot_table(df,values='Rating',index='CustomerID',columns='MovieID')

movieNames = pd.read_csv(BUCKET_PATH + 'movie_titles.csv',encoding = "ISO-8859-1", header = None, names = ['MovieID', 'Name'],usecols = [0,2])

print (movieNames.head(10))
print(matrix)

def euclidean_distance(id1, id2):
    distance = 0.0
    if len(user_ratings[id1]) >= len(user_ratings[id2]):
        id1, id2 = id2, id1
    movies_rated = 0
    for movie in user_ratings[id1]:
        if movie in user_ratings[id2]:
            movies_rated += 1
            distance += (user_ratings[id1][movie] - user_ratings[id2][movie])**2
    return [distance, movies_rated]

# Locate the most DISsimilar neighbors
#I have modified this by adding "reverse=true"
def get_neighbours(train, test_index, k):
    min_movies = 3
    distances = [(train.loc[i], euclidean_distance(train.iloc[[test_index]].index[0], i)) for i in train.index]
    distances_meeting_min_movies = list(filter(lambda x: x[1][1] >= min_movies, distances))
    distances.sort(key=lambda tup: tup[1][0], reverse=True)
    distances_meeting_min_movies.sort(key=lambda tup: tup[1][0])
    print([distance[1] for distance in distances[:5]])
    furthest_neighbours = [distances[i][0] for i in range(k)]
    nearest_neighbours = [distance[0] for distance in distances_meeting_min_movies[:k]]
    return furthest_neighbours, nearest_neighbours

def movies_not_seen(row_index):
    customer_row = matrix.iloc[[row_index]].transpose()
    customer_row = customer_row[customer_row[customer_row.columns[0]].isnull()]
    return customer_row

def predict(row_index=0):
    k = 20
    percent_of_neighbours_rated = 0.25
    unseen_movies = movies_not_seen(row_index)
    furthest_neighbours, nearest_neighbours = get_neighbours(matrix, row_index, k)
    furthest_neighbours_combined = pd.concat(map(lambda x: x.to_frame().transpose(), furthest_neighbours))
    nearest_neighbours_combined = pd.concat(map(lambda x: x.to_frame().transpose(), nearest_neighbours))
    not_enough_ratings = []
    for column in furthest_neighbours_combined.columns:
        if len(furthest_neighbours_combined[column].dropna()) < percent_of_neighbours_rated*k:
            not_enough_ratings.append(column)
    furthest_neighbours_combined.drop(columns=not_enough_ratings, inplace=True)
    not_enough_ratings = []
    for column in nearest_neighbours_combined.columns:
        if len(nearest_neighbours_combined[column].dropna()) < percent_of_neighbours_rated*k:
            not_enough_ratings.append(column)
    nearest_neighbours_combined.drop(columns=not_enough_ratings, inplace=True)
    furthest_scores = furthest_neighbours_combined.mean(axis=0, skipna=True)
    nearest_scores = nearest_neighbours_combined.mean(axis=0, skipna=True)
    furthest_scores = furthest_scores.filter(items=unseen_movies.index)
    nearest_scores = nearest_scores.filter(items=unseen_movies.index)
    top_10_furthest_movies = furthest_scores.nlargest(10)
    top_10_nearest_movies = nearest_scores.nlargest(10)
    return top_10_furthest_movies, top_10_nearest_movies

avgs = matrix.mean(axis=0, skipna=True)

fs_ns = []

for i in range(5):
    print("CUSTOMER", i)
    f, n = predict(i)
    fs_ns.append((f, n))
    print("FARTHEST:\n", f)
    print("NEAREST:\n", n)
    print(movieNames[movieNames['MovieID'].isin(f.index)])
    print(movieNames[movieNames['MovieID'].isin(n.index)])

for i,e in enumerate(fs_ns):
    print("OVERLAP", i)
    f, n = e
    print(len(set(f.index).intersection(set(n.index))))
    print("FARTHEST QUALITY:", [avgs.loc[k] for k in f.index])
    print("NEAREST QUALTY:", [avgs.loc[k] for k in n.index])
