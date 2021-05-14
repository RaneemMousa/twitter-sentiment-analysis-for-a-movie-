import nltk
import pandas as pd
import numpy as np
import seaborn as sns
import tweepy
import matplotlib.pyplot as plt
import os
from textblob import TextBlob


consumer_key = 'SHc52peDQfk8XkzCNRHv2GNwf'
consumer_secret = 'kx2YPmus02eEWe1hcvvUAjqqYth1yRL7tFlczIUTU5DduEl7nX'
access_token = '632285589-mjcZvyLN2XgcaqYI4lX8G9hWaB8OLKCE0NuhGiEI'
access_token_secret = 'xjGkz1Qb2oTHAx1uEQbJhBIWLWzzGMSNso91czXeOPBbF'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


def fetch_tweets(hashtag):
    tweet_user = []
    tweet_time = []
    tweet_string = []

    for tweet in tweepy.Cursor(api.search, q=hashtag, count=2000).items(2000):
        if (not tweet.retweeted) and ("RT @" not in tweet.text):
            if tweet.lang == "en":
                tweet_user.append(tweet.user.name)
                tweet_time.append(tweet.created_at)
                tweet_string.append(tweet.text)

    df = pd.DataFrame({"username": tweet_user, "time": tweet_time, "tweet": tweet_string})

    return df

movie= fetch_tweets("moana movie")


# ### Checking the data:
pd.set_option('display.max_colwidth', None)
print(movie.head(50))
print(movie.info())


# ### Data cleaning
import nltk

from nltk.corpus import stopwords
sw=stopwords.words('english')
sw=[word for word in sw if word !='didn' and word !="didn't" and word !='doesn'
    and word !="doesn't" and word != 'don' and word !="don't" and word !='not' ]
print(len(sw))
print(len(stopwords.words('english')))
print(sw)

import string
import re
import demoji

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
from wordsegment import load,segment
load()
def text_process(tweet):
    #removing mentions
    tweet = re.sub(r'@[A-Za-z0-9]+','' ,tweet, flags=re.MULTILINE)
    #removing url links
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    #removing numbers
    tweet = ''.join([i for i in tweet if not i.isdigit()])
    #converting some words to not
    tweet=re.sub(r"\bdidn't\b","not",tweet.lower())
    tweet=re.sub(r"\bdoesn't\b","not",tweet.lower())
    tweet=re.sub(r"\bdon't\b","not",tweet.lower())

    #converting emojis to their meaning
    #demoji.download_codes()
    l=demoji.findall(tweet)
    for key, value in l.items():
        tweet = tweet.replace(key, value)
    #removing puctuations
    nopunc = [char for char in tweet if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    #seperating words
    nopunc=' '.join(segment(nopunc))
    #returning the tweet without the stopwords
    tokens = [word for word in nopunc.split() if word.lower() not in sw]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


# ### adding the processed tweets to the dataframe
movie['ptweet']=movie['tweet'].apply(text_process)

#just a test
text_process('why is Trolls: World Tour on Hulu but not Trolls going 😔')


# ### Is there any direct rating in the tweets that could help?
def stars(tweet):
    X=0
    for word in tweet.split():
        if word.upper() == 'STARS' or word.upper()== 'STAR':
            X=X+1
    return (X)

movie['star']=movie['tweet'].apply(stars)

movie['star'].sum()

star_coloumn = movie.loc[movie['star'] == 1]
print (star_coloumn)
#no benefits


#vectorizing:
from sklearn.feature_extraction.text import TfidfVectorizer

tfidfconvert = TfidfVectorizer(analyzer=text_process).fit(movie['tweet'])

X_transformed=tfidfconvert.transform(movie['tweet'])


# Clustering the training sentences with K-means technique

from sklearn.cluster import KMeans
modelkmeans = KMeans(n_clusters=2, init='k-means++', n_init=100)
modelkmeans.fit(X_transformed)

r=modelkmeans.labels_
newr = ['Positive' if x ==0 else 'Negative' for x in r]
movie['Clustering labels']=newr


#using polarity
def getPolarity(tw):
   return TextBlob(tw).sentiment.polarity

movie['polarity']=movie['tweet'].apply(getPolarity)
print(movie.head())


def getresult(score):
    if(score == 0):
         return "Neutral"
    elif(score > 0 and score <= 0.3):
        return"Weakly Positive"
    elif(score > 0.3 and score <= 0.6):
        return "Positive"
    elif(score > 0.6 and score <= 1):
        return"Strongly Positive"
    elif(score > -0.3 and score <= 0):
        return"Weakly Negative"
    elif(score > -0.6 and score <= -0.3):
        return "Negative"
    elif(score > -1 and score <= -0.6):
        return"Strongly Negative"

movie['result']=movie['polarity'].apply(getresult)
print(movie.head())

# Neutral
percent1 = movie[movie.result == "Neutral"]
percent1 = percent1['tweet']

f = round((percent1.shape[0] / movie.shape[0]) * 100, 1)
print(" Neutral percentage= " + str(f))

# Weakly Positive
percent1 = movie[movie.result == "Weakly Positive"]
percent1 = percent1['tweet']

f = round((percent1.shape[0] / movie.shape[0]) * 100, 1)
print(" Weakly Positive= " + str(f))

# Positive
percent1 = movie[movie.result == "Positive"]
percent1 = percent1['tweet']

f = round((percent1.shape[0] / movie.shape[0]) * 100, 1)
print(" Positive= " + str(f))

# Strongly Positive
percent1 = movie[movie.result == "Strongly Positive"]
percent1 = percent1['tweet']

f = round((percent1.shape[0] / movie.shape[0]) * 100, 1)
print(" Strongly Positive= " + str(f))

#Weakly Negative
percent1 = movie[movie.result == "Weakly Negative"]
percent1 = percent1['tweet']

f = round((percent1.shape[0] / movie.shape[0]) * 100, 1)
print(" Weakly Negative= " + str(f))
#Negative
percent1 = movie[movie.result == "Negative"]
percent1 = percent1['tweet']

f = round((percent1.shape[0] / movie.shape[0]) * 100, 1)
print("Negative= " + str(f))

#Strongly Negative
percent1 = movie[movie.result == "Strongly Negative"]
percent1 = percent1['tweet']

f = round((percent1.shape[0] / movie.shape[0]) * 100, 1)
print("Strongly Negative= " + str(f))













'''

from sklearn.feature_extraction.text import CountVectorizer
# Might take awhile...
bow_transformer = CountVectorizer(analyzer=text_process).fit(df['message'])

# Print total number of vocab words
print(len(bow_transformer.vocabulary_))

messages_bow = bow_transformer.transform(messages['message'])
print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)

sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, df['label']) ''