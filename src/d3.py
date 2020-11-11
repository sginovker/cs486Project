from math import sqrt

import pandas as pd
from sklearn.model_selection import cross_validate


def euclidean_distance(row1, row2):
    """Calculate the Euclidean distance between two vectors"""
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


def get_neighbors(train, test_row, k):
    """
    Locate the most DISsimilar neighbors
    I have modified this by adding "reverse=true"
    """
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1], reverse=True)
    neighbors = list()
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors


if __name__ == '__main__':
    # TODO: need to add col labels, formatting to csv (see d2 methodology) BEFORE running this (make separate script)

    # data combining+formatting

    df1 = pd.read_csv('combined_data1.txt', index_col=0)
    df2 = pd.read_csv('combined_data2.txt', index_col=0)
    df3 = pd.read_csv('combined_data3.txt', index_col=0)
    df4 = pd.read_csv('combined_data4.txt', index_col=0)

    frames = [df1, df2, df3, df4]
    # concat all data
    df = pd.concat(frames)

    # drop date col
    df.drop(columns='date')

# TODO:add col to signify if user has rated given movie

# TODO:substitute any nulls with mean

# TODO:do 5-fold cross validation
