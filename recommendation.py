# libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import  matplotlib.colors as m
import seaborn as sns
import sklearn
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from ast import literal_eval

# mount data
df1=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')
df2=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')

# combining both datasetss

data = pd.merge(df1,df2)

# preprocess

print(f'null vals')
print(data.isna().sum())
print()
print(f'dups')
print(data.duplicated().sum())
print()
print(f'information')
data.info()
print()
print(f'describe')
data.describe()

# distribution of ratings

bins = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
labels = ['1–2', '2–3', '3–4', '4–5', '5–6', '6–7', '7–8', '8–9', '9–10']

data['rating_category'] = pd.cut(data['vote_average'], bins=bins, labels=labels, include_lowest=True)
category_counts = data['rating_category'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette='coolwarm')

plt.title('Distribution of Movies by Vote Average Category')
plt.xlabel('Vote Average Category')
plt.ylabel('Number of Movies')
plt.tight_layout()
plt.show()

# highest rated

high_count = data.sort_values('vote_count', ascending=False)
high_count

# production companies
print(f'no.countries =', data['production_countries'].nunique())
print()
#languafe
print(f'no.languages =',data['original_language'].nunique())
print()
#status
print(f'no.status =',data['status'].nunique())
print()
#budget
print(f'Budget description')
data['budget'].describe()

# languaes distribution

label = data['original_language'].nunique()
lang_counts = data['original_language'].value_counts()

lang_df = lang_counts.reset_index()
lang_df.columns = ['language', 'count']

plt.figure(figsize=(10, 6))
sns.barplot(x='language', y='count', data=lang_df, palette='pastel')

plt.title('Distribution By Original Language')
plt.xlabel('Language')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# distribution of status

labels = data['status'].nunique()
status_count = data['status'].value_counts()
status_df = status_count.reset_index()
status_df.columns =['status', 'count']


plt.figure(figsize=(10, 5))
ax = sns.barplot(data=status_df, x='count', y='status', palette='coolwarm')
for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', padding=3)
plt.title('Movie Status Distribution')
plt.xlabel('Number of Movies')
plt.ylabel('Status')
plt.tight_layout()
plt.show()

null_release = data[data['release_date'].isna()]
null_release['movie_id']

null_overview = data[data['overview'].isna()]
null_overview['movie_id']

null_overview = data[data['runtime'].isna()]
print(null_overview['runtime'])

df = data.copy()
df = df.drop(['id'], axis=1)
df['tagline'] = df['tagline'].fillna('No tagline')
df['homepage'] = df['homepage'].fillna('No homepage')

df=df.dropna()
df.info()

# get all num cats
num_cat = df.select_dtypes(include=['int64', 'float64']).columns
print(num_cat)

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df[feature] = df[feature].apply(literal_eval)
df

# director
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return 'no director'

# return top 3 lisstss
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
      
        if len(names) > 3:
            names = names[:3]
        return names
    return ''

df['director'] = df['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    df[feature] = df[feature].apply(get_list)
df


df[['title', 'cast', 'director', 'keywords', 'genres']]

# cleaning it

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
        
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    df[feature] = df[feature].apply(clean_data)
    

# connect it into 1 place

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
df['final'] = df.apply(create_soup, axis=1)

print(len(df['final']))


# TF - IDF

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['final'])

print(tfidf_matrix.shape)

# cosine sim
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['title'])

# function to run cosine sim

def cosine_sim(title,cosine_sim=cosine_sim_matrix):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]

    return df['title'].iloc[movie_indices]

# testing with Avatar
cosine_sim('Avatar')

cosine_sim('JFK')

# import groundtruth
df_gt = pd.read_csv('/kaggle/input/sadsaa/recommendation_groundtruth_clustered.csv')
df_gt

id_to_title = pd.Series(df.title.values, index=df.movie_id).to_dict()
title_to_id = pd.Series(df.movie_id.values, index=df.title).to_dict()

def recommend_movie_ids(title, k=10, cosine_sim=cosine_sim_matrix):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:k+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['movie_id'].iloc[movie_indices].tolist()

def precision_at_k(groundtruth_df, k=10):
    user_groups = groundtruth_df.groupby('user_id')['movie_id'].apply(set).to_dict()
    precisions = []

    for user_id, relevant_movies in user_groups.items():
        seed_movie_id = next(iter(relevant_movies))
        seed_title = id_to_title.get(seed_movie_id, None)
        
        if seed_title is None or seed_title not in indices:
            continue
        recommended_ids = recommend_movie_ids(seed_title, k)
        hits = len(set(recommended_ids) & relevant_movies)
        precisions.append(hits / k)

    return sum(precisions) / len(precisions)

p_5 = precision_at_k(df_gt, k=5)
print(f"Precision@5: {p_5:.4f}")