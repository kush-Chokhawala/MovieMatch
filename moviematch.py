import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

movies = pd.read_csv('/content/drive/MyDrive/Kush_Modi Personal/Personal Projects/Unsupervised ML/Movie Recommender System (Content Based)/tmdb_5000_movies.csv')

credits = pd.read_csv('/content/drive/MyDrive/Kush_Modi Personal/Personal Projects/Unsupervised ML/Movie Recommender System (Content Based)/tmdb_5000_credits.csv')

movies.head(2)

movies.shape

credits.head()

movies = movies.merge(credits,on='title')

movies.head()
# budget
# homepage
# id
# original_language
# original_title
# popularity
# production_comapny
# production_countries
# release-date(not sure)

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

movies.head()

import ast

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies.dropna(inplace=True)

movies['genres'] = movies['genres'].apply(convert)
movies.head()

movies['keywords'] = movies['keywords'].apply(convert)
movies.head()

import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')

def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L

movies['cast'] = movies['cast'].apply(convert)
movies.head()

movies['cast'] = movies['cast'].apply(lambda x:x[0:3])

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

#movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.sample(5)

def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

movies.head()

movies['overview'] = movies['overview'].apply(lambda x:x.split())

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
#new.head()

new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')

vector = cv.fit_transform(new['tags']).toarray()

vector.shape

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vector)

similarity

new[new['title'] == 'The Lego Movie'].index[0]

def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)

recommend('Gandhi')

recommend('Avatar')

recommend('Dolphin Tale 2')

import matplotlib.pyplot as plt

# Example: Visualization of movie genres
genre_counts = movies['genres'].explode().value_counts()
genre_counts.plot(kind='bar', figsize=(12, 6))
plt.title('Distribution of Movie Genres')
plt.xlabel('Genres')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

# Example: Comparing the number of recommendations for different movies
def get_recommendations_count(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommendation_count = [new.iloc[i[0]].title for i in distances[1:6]]
    return recommendation_count

movie_titles = ['Gandhi', 'Avatar', 'Dolphin Tale 2']
recommendation_counts = [len(get_recommendations_count(movie)) for movie in movie_titles]

plt.bar(movie_titles, recommendation_counts)
plt.title('Number of Recommendations for Different Movies')
plt.xlabel('Movies')
plt.ylabel('Recommendation Count')
plt.show()

a = get_recommendations_count('Avatar')
a

# Example: Visualization of tag frequency
from collections import Counter

tags = movies['tags'].explode().tolist()
tag_counts = Counter(tags)
most_common_tags = tag_counts.most_common(10)

tags, counts = zip(*most_common_tags)

plt.figure(figsize=(12, 6))
plt.bar(tags, counts)
plt.title('Top 10 Most Common Tags')
plt.xlabel('Tags')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

"""Word Cloud for Movie Overview"""

from wordcloud import WordCloud

# Example: Word cloud for movie overviews
overview_text = ' '.join(movies['overview'].explode().astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(overview_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Movie Overviews')
plt.axis('off')
plt.show()

"""Analysis of Movie Keywords"""

# Example: Visualization of movie keywords
keyword_counts = movies['keywords'].explode().value_counts()
keyword_counts[:20].plot(kind='bar', figsize=(12, 6))
plt.title('Top 20 Movie Keywords')
plt.xlabel('Keywords')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

"""Analysis of Movie Cast"""

# Example: Visualization of movie cast
cast_counts = movies['cast'].explode().value_counts()
cast_counts[:20].plot(kind='bar', figsize=(12, 6))
plt.title('Top 20 Movie Cast Members')
plt.xlabel('Actors')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

"""Analysis of Movie Crew"""

# Example: Visualization of movie crew (directors)
crew_counts = movies['crew'].explode().value_counts()
crew_counts[:20].plot(kind='bar', figsize=(12, 6))
plt.title('Top 20 Movie Crew Members')
plt.xlabel('Crew Members')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

""" Movie Title Length Analysis"""

# Example: Histogram of movie title lengths
movies['title_length'] = movies['title'].apply(len)
plt.hist(movies['title_length'], bins=20, alpha=0.7, color='blue', edgecolor='black', linewidth=1.2)
plt.title('Distribution of Movie Title Lengths')
plt.xlabel('Title Length')
plt.ylabel('Count')
plt.show()
