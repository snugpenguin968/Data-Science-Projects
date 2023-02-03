import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

st.title('Movie Recommendation System')
st.write('Simple Item-Based Collaborative Filtering on user reviews with the K-Nearest-Neighbors Algorithm🎬')
title=st.text_input('Enter Movie Title (Case-Sensitive)','Star Wars')
number=st.number_input('Enter Number of Recommendations',min_value=1,max_value=20)

movies=pd.read_csv('dataset/movies.csv')
ratings=pd.read_csv('dataset/ratings.csv')

final_dataset = ratings.pivot(index='movieId',columns='userId',values='rating')
final_dataset.fillna(0,inplace=True)
number_user_voted = ratings.groupby('movieId')['rating'].agg('count')
number_movies_voted = ratings.groupby('userId')['rating'].agg('count')
final_dataset = final_dataset.loc[number_user_voted[number_user_voted > 10].index,:]
final_dataset=final_dataset.loc[:,number_movies_voted[number_movies_voted > 50].index]
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

def get_movie_recommendation(movie_name,n_recommendations):
    movie_list = movies[movies['title'].str.contains(movie_name)]  
    if len(movie_list):
        movie_idx=movie_list.iloc[0]['movieId']
        movie_idx=final_dataset[final_dataset['movieId']==movie_idx].index[0]
        distances,indices=knn.kneighbors(csr_data[movie_idx],n_neighbors=n_recommendations+1)
        rec_movie_indices=sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x:x[1])[:0:-1]
        recommend_frame=[]
        for val in rec_movie_indices:
            movie_idx=final_dataset.iloc[val[0]]['movieId']
            idx=movies[movies['movieId']==movie_idx].index
            recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Similarity':val[1],'Keywords':movies.iloc[idx]['genres'].values[0].split('|')})
        df=pd.DataFrame(recommend_frame,index=range(1,n_recommendations+1))
        return df
    else:
        return "No movies found. Please check your input"
if st.button('Submit'):
    st.write(get_movie_recommendation(title,number))
