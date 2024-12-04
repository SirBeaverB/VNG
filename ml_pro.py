import pandas as pd
import numpy as np
from scipy.sparse import hstack, vstack, csr_matrix
from collections import defaultdict

# 加载数据
links = pd.read_csv(r'VNG/ml-latest-small/links.csv')
movies = pd.read_csv(r'VNG/ml-latest-small/movies.csv')
ratings = pd.read_csv(r'VNG/ml-latest-small/ratings.csv')
tags = pd.read_csv(r'VNG/ml-latest-small/tags.csv')

print("loading data done")

# 提取主类型作为电影的分类标签
movies['main_genre'] = movies['genres'].str.split('|').str[0]  # 获取第一个类型
unique_genres = movies['main_genre'].unique()
genre_to_label = {genre: i for i, genre in enumerate(unique_genres)}  # 类型映射为标签
movie_labels = movies['main_genre'].map(genre_to_label).values  # 获取电影的标签
print("movie_labels done")

# 统计用户对每种类型的评分总和
ratings_with_genres = ratings.merge(movies[['movieId', 'genres']], on='movieId')
ratings_with_genres['main_genre'] = ratings_with_genres['genres'].str.split('|').str[0]

# 计算用户对每种类型的评分总和
user_genre_preferences = ratings_with_genres.groupby(['userId', 'main_genre'])['rating'].sum().unstack(fill_value=0)

# 用户偏好标签为评分最高的类型
user_labels = user_genre_preferences.idxmax(axis=1).map(genre_to_label).values

# 合并用户和电影标签
num_users = len(user_labels)
num_movies = len(movie_labels)

# 创建标签向量
labels = np.zeros(num_users + num_movies, dtype=int)
labels[:num_users] = user_labels  # 用户标签
labels[num_users:] = movie_labels  # 电影标签


print("Labels combined for all nodes")

# 获取唯一的用户ID和电影ID
unique_user_ids = ratings['userId'].unique()
unique_movie_ids = ratings['movieId'].unique()

# 映射用户和电影的唯一 ID
user_map = {uid: i for i, uid in enumerate(unique_user_ids)}
movie_map = {mid: i + len(unique_user_ids) for i, mid in enumerate(unique_movie_ids)}

# 构建邻接矩阵
rows = ratings['userId'].map(user_map).values
cols = ratings['movieId'].map(movie_map).values
data = ratings['rating'].values  # 边权重为评分值

# 处理缺失值，将缺失的用户和电影 ID 映射到一个特殊值（例如 -1）
rows = ratings['userId'].map(user_map).fillna(0).astype(int).values
cols = ratings['movieId'].map(movie_map).fillna(0).astype(int).values

# 构建 genre 到电影 ID 的映射
genre_to_movies = defaultdict(list)
for movie_id, genres in zip(movies['movieId'], movies['genres']):
    for genre in genres.split('|'):  # 使用所有类型
        genre_to_movies[genre].append(movie_id)

for genre, movie_list in genre_to_movies.items():
    print(f"Genre: {genre}, Number of movies: {len(movie_list)}")

# 为每个类型生成附加边
additional_edges = []
for genre, movie_list in genre_to_movies.items():
    print(genre)
    for i in range(len(movie_list)):
        for j in range(i + 1, len(movie_list)):
            movie_i = movie_map.get(movie_list[i], None)
            movie_j = movie_map.get(movie_list[j], None)
            if movie_i is not None and movie_j is not None:
                additional_edges.append((movie_i, movie_j))
                additional_edges.append((movie_j, movie_i))  # 双向边

expected_edges = sum(len(movie_list) * (len(movie_list) - 1) for movie_list in genre_to_movies.values())
print(f"Expected number of edges (theoretical): {expected_edges}")
print(f"Actual number of edges: {len(additional_edges)}")

# 构建电影-电影的边矩阵
if additional_edges:
    movie_rows, movie_cols = zip(*additional_edges)
    movie_data = np.ones(len(movie_rows))  # 默认边权重为 1
else:
    movie_rows, movie_cols, movie_data = [], [], []

# 合并所有边
all_rows = np.concatenate([rows, movie_rows])
all_cols = np.concatenate([cols, movie_cols])
all_data = np.concatenate([data, movie_data])

# 构建最终邻接矩阵
adj_matrix = csr_matrix((all_data, (all_rows, all_cols)),
                        shape=(num_users + num_movies, num_users + num_movies))

print("Adjacency matrix created")

# 用户对每种类型的偏好转为 one-hot 编码
user_attr_matrix = np.zeros((num_users, len(genre_to_label)), dtype=int)
for user_id, genre in enumerate(user_genre_preferences.idxmax(axis=1)):
    user_attr_matrix[user_id, genre_to_label[genre]] = 1

# 转为稀疏矩阵
user_attr_matrix = csr_matrix(user_attr_matrix)

print("User attribute matrix created")

# 电影属性矩阵（基于 genres）
movie_genres = movies['genres'].str.get_dummies('|')  # Genres one-hot 编码
movie_attr_matrix = csr_matrix(movie_genres.values)

# 对用户属性矩阵进行扩展
if user_attr_matrix.shape[1] < movie_attr_matrix.shape[1]:
    diff = movie_attr_matrix.shape[1] - user_attr_matrix.shape[1]
    user_attr_matrix = csr_matrix(hstack([user_attr_matrix, csr_matrix((user_attr_matrix.shape[0], diff))]))

# 对电影属性矩阵进行扩展
if movie_attr_matrix.shape[1] < user_attr_matrix.shape[1]:
    diff = user_attr_matrix.shape[1] - movie_attr_matrix.shape[1]
    movie_attr_matrix = csr_matrix(hstack([movie_attr_matrix, csr_matrix((movie_attr_matrix.shape[0], diff))]))


print(f"user_attr_matrix shape: {user_attr_matrix.shape}")
print(f"movie_attr_matrix shape: {movie_attr_matrix.shape}")

# 合并属性矩阵
attr_matrix = vstack([user_attr_matrix, movie_attr_matrix])

print("Attribute matrix created")

# 节点名称
node_names = np.concatenate([ratings['userId'].unique(), movies['title'].values])

# 属性名称（如果有电影特征）
attr_names = np.concatenate([[], movie_genres.columns])

# 边属性（如评分）
edge_attr_names = ['rating']  # 假设只有评分作为边属性

# 类别名称（电影类型）
class_names = np.array(list(genre_to_label.keys()))

print("Metadata created")

np.savez('movielens.npz',
         adj_matrix_data=adj_matrix.data,
         adj_matrix_indices=adj_matrix.indices,
         adj_matrix_indptr=adj_matrix.indptr,
         adj_matrix_shape=adj_matrix.shape,
         attr_matrix_data=attr_matrix.data,
         attr_matrix_indices=attr_matrix.indices,
         attr_matrix_indptr=attr_matrix.indptr,
         attr_matrix_shape=attr_matrix.shape,
         labels=labels,
         node_names=node_names,
         attr_names=attr_names,
         #edge_attr_names=edge_attr_names,
         class_names=class_names)
         
print("Graph data saved as movielens.npz")