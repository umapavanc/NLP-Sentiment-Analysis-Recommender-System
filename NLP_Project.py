#!/usr/bin/env python
# coding: utf-8

# ### Data Exploration

# In[65]:


import pandas as pd
import gzip
import json

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF("../Office_Products_5.json.gz")


# In[66]:


df


# This Dataset is an updated version of the Amazon review dataset released in 2014. As in the previous version, this dataset includes reviews (ratings, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features), and links (also viewed/also bought graphs). In addition, this version provides the following features:
# 
# More reviews:
# The total number of reviews is 233.1 million (142.8 million in 2014).
# Newer reviews:
# Current data includes reviews in the range May 1996 - Oct 2018.
# Metadata:
# We have added transaction metadata for each review shown on the review page. Such information includes:
# Product information, e.g. color (white or black), size (large or small), package type (hardcover or electronics), etc.
# Product images that are taken after the user received the product.
# Added more detailed metadata of the product landing page. Such detailed information includes:
# Bullet-point descriptions under product title.
# Technical details table (attribute-value pairs).
# Similar products table.
# More categories:
# Includes 5 new product categories.
# 
# Source: https://nijianmo.github.io/amazon/index.html#code

# In[67]:


df.shape


# In[68]:


df.columns


# In[69]:


counts_of_reviews_per_product = df.groupby('asin').size()
for product, count_of_reviews_per_product in counts_of_reviews_per_product.iteritems():
  print(f'{product} has {count_of_reviews_per_product} reviews')
# counts_of_reviews_per_product


# In[70]:


len(counts_of_reviews_per_product)


# In[71]:


import matplotlib.pyplot as plt
import numpy as np

# Plot the distribution using matplotlib.pyplot.hist() function.
plt.hist(counts_of_reviews_per_product[:50], bins=np.arange(0, 100, 5))
plt.xlabel('Number of Reviews')
plt.ylabel('Frequency')
plt.title('Distribution of the Number of Reviews Across Products')
plt.show()


# In[72]:


counts_of_reviews_per_product[:10]


# In[73]:


plt.figure(figsize=(25,8))
plt.plot(counts_of_reviews_per_product[:10])
plt.xlabel('Product')
plt.ylabel('Number of reviews')
plt.title('Distribution of the number of reviews per product')
plt.show()


# In[74]:


counts_of_reviews_across_products = df.groupby(['asin', 'overall']).size()
# for product, count_of_reviews_per_product in counts_of_reviews_per_product.iteritems():
#   print(f'{product} has {count_of_reviews_per_product} reviews')
counts_of_reviews_across_products[:10]


# In[ ]:


# Unstack the data to create a pivot table with product ids as rows and review ratings as columns
reviews_by_product_and_rating = counts_of_reviews_across_products[:20].unstack()

# Plot the distribution of the number of reviews per product per star rating as a histogram
reviews_by_product_and_rating.plot(kind='bar', stacked=True)
plt.xlabel('Product ID')
plt.ylabel('Number of reviews')
plt.title('Distribution of the number of reviews per product per star rating')
plt.show()


# In[ ]:


counts_of_reviews_per_user = df.groupby('reviewerID').size()
for user, count_of_review in counts_of_reviews_per_user.iteritems():
  print(f'{user} has {count_of_review} reviews')


# In[13]:


plt.figure(figsize=(25,8))
plt.plot(counts_of_reviews_per_user[:10])
plt.xlabel('User')
plt.ylabel('Number of reviews')
plt.title('Distribution of the number of reviews per user')
plt.show()


# In[14]:


positive = df[df['overall'] > 3]
negative = df[df['overall'] < 3]
positive = positive.dropna()
negative = negative.dropna()


# In[15]:


# common words in positive review comments
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud

stopwords = set(stopwords.words('english'))
stopwords.update(["br", "stuff", "href","taste", "product", "flavour","like", "coffee", "dog","flavor","buy"]) # need detemination

pos = " ".join(review for review in positive.reviewText)
wordcloud = WordCloud(stopwords=stopwords).generate(pos)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[16]:


# common words in negative review comments
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
stopwords = set(stopwords.words('english'))
negreviews = " ".join(review for review in negative.reviewText)
wordcloud = WordCloud(stopwords=stopwords).generate(negreviews)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[17]:


df['reviewText']


# ### Pre-processing

# In[18]:


import random
n_samples = random.randint(500, 1000)
df_random = df.sample(n=n_samples)


# In[19]:


df_random.shape


# In[20]:


df_random


# In[21]:


def condition(overall):
  # print(df)
  rating = overall
  if rating in (4.0, 5.0):
    return 'Positive'
  elif rating == 3.0:
    return 'Neutral'
  elif rating in (1.0, 2.0):
    return 'Negative'

df_random['label'] = df_random['overall'].apply(condition)


# In[22]:


df_random


# In[23]:


final_df = pd.DataFrame(df_random['reviewText']) 


# In[24]:


type(final_df)


# In[25]:


final_df


# In[26]:


import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import string

nltk.download('stopwords')
nltk.download('punkt')

for _, review in final_df.iterrows():
  sentence = review['reviewText'].lower()
  sentence = sentence.translate(str.maketrans("", "", string.punctuation))

  # Tokenize the sentence into words and remove stop words
  stop_words = set(stopwords.words('english'))
  # Tokenize the sentence into words
  words = [word for word in nltk.word_tokenize(sentence) if word.lower() not in stop_words]

  if words:
    # Create the TF-IDF vectorizer object
    tfidf = TfidfVectorizer()

    # Fit and transform the words using the vectorizer object
    tfidf_matrix = tfidf.fit_transform(words)

    # Print the TF-IDF matrix
    print(tfidf_matrix.toarray())


# ### Modeling

# #### VADR

# In[27]:


get_ipython().system('pip install vaderSentiment')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[28]:


# Valence Aware Dictionary and Sentiment Reasoner (VADR)
VADR_analyzer = SentimentIntensityAnalyzer()

predicted_sentiments = []
# Pass the analyzer for head 5 rows
for index, row in df_random.head().iterrows():
    vs = VADR_analyzer.polarity_scores(row["reviewText"])
    print(
        f"Index: {index}\n"+
        f"Sentimental Analysis Result: {vs}\n"+
        f"Overall rating category: {row.label}\n"+
        f"Full Text:\n{row.reviewText}\n"+
        "-"*50)
    if vs['neg'] > vs['pos']:
      sentiment = 'Negative'
    elif vs['pos'] > vs['neg']:
      sentiment= 'Positive'
    else:
      sentiment = 'Neutral'
    predicted_sentiments.append(sentiment)


# #### TextBlob

# In[29]:


get_ipython().system('pip install textblob')
from textblob import TextBlob


# In[30]:


list(final_df['reviewText'])


# In[ ]:


# Plot first 10


# #### SENTIWORDNET

# In[82]:


from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize
from nltk.wsd import lesk

# Download necessary resources
nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')

# function to calculate the sentiment score for each word using SentiWordNet
def sw_sentiment_score(word, tag):
    synsets = list(swn.senti_synsets(word, tag))
    if synsets:
        pos_score = synsets[0].pos_score()
        neg_score = synsets[0].neg_score()
        return pos_score - neg_score
    else:
        return 0

# function to calculate the overall sentiment score for each review
def sw_review_sentiment_score(review):
    tokens = word_tokenize(review)
    sentiment_score = 0
    for token in tokens:
        synset = lesk(tokens, token)
        if synset:
            sentiment_score += sw_sentiment_score(synset.lemmas()[0].name(), synset.pos())
    return sentiment_score / len(tokens)


# ### Validation

# In[42]:


# VADR
predicted_sentiments_vadr = []
for index, row in df_random.iterrows():
    # Pass analyzer
    vs = VADR_analyzer.polarity_scores(row["reviewText"])
    if vs['neg'] > vs['pos']:
      sentiment = 'Negative'
    elif vs['pos'] > vs['neg']:
      sentiment= 'Positive'
    else:
      sentiment = 'Neutral'
    predicted_sentiments_vadr.append(sentiment)

(predicted_sentiments_vadr == df_random['label']).value_counts()


# In[62]:


count = (predicted_sentiments_vadr == df_random['label']).value_counts()
accuracy = count[True]/(count[True]+count[False]) * 100
accuracy


# In[44]:


# TextBlob
predicted_sentiments = []
for text in list(final_df['reviewText']):
  if isinstance(text, str):
    wiki = TextBlob(text)
    predicted_sentiments.append(wiki.sentiment)


# In[45]:


predicted_ratings_txt = []
for predicted_sentiment in predicted_sentiments:
  if predicted_sentiment.polarity == 0:
    predicted_ratings_txt.append('Neutral')
  elif predicted_sentiment.polarity < 0 :
    predicted_ratings_txt.append('Negative')
  elif predicted_sentiment.polarity > 0 :
    predicted_ratings_txt.append('Positive')


# In[46]:


(predicted_ratings_txt == df_random['label']).value_counts()


# In[64]:


count = (predicted_ratings_txt == df_random['label']).value_counts()
accuracy = count[True]/(count[True]+count[False]) * 100
accuracy


# In[84]:


# SENTIWORDNET

predicted_sentiments_senti = []
# Classifing each review into positive, negative, or neutral sentiment.
for review in final_df['reviewText']:
    sentiment_score = sw_review_sentiment_score(review)
    if sentiment_score > 0:
# #         print('\nPositive Review:', review)
        predicted_sentiments_senti.append('Positive')
    elif sentiment_score < 0:
#         print('\nNegative Review:', review)
        predicted_sentiments_senti.append('Negative')
    else:
#         print('\nNeutral Review:', review)
        predicted_sentiments_senti.append('Neutral')


# In[85]:


count = (predicted_sentiments_senti == df_random['label']).value_counts()
accuracy = count[True]/(count[True]+count[False]) * 100
accuracy

