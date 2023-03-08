#!/usr/bin/env python
# coding: utf-8

# ### Data Exploration

# In[4]:


import pandas as pd
import gzip
import json
import seaborn as sns
import re
import matplotlib.pyplot as plt

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

df = getDF("/content/drive/MyDrive/Centennial/NLP/Office_Products_5.json.gz")


# In[6]:


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

# In[7]:


df.info()


# In[8]:


df.describe()

# Observation1: Average of review rating is around 4.47 which might imply that dataset contains more positive reviews.


# In[9]:


df.shape


# In[10]:


df.columns


# In[11]:


print("Count of null values: ", df['reviewText'].isnull().sum())

# Observation2: Fetch the indexes of null values
nullIndexes = df[df['reviewText'].isnull()].index.tolist()
for index in (nullIndexes):
    print("Index:", index)


# In[12]:


df["overall"].value_counts()

# Observation3: Dataset need to be balanced for ML approach


# In[13]:


df[df["reviewText"].isnull()].overall.value_counts()

# # Observation4: Most of the null values in review column are for high ratings. 
# Options to handle null values: Filling with a constant value, delete those rows, or text imputation techniques


# In[14]:


# Distribution of the number of reviews across products
# Group the reviews by product ID and count the number of reviews per product
reviews_per_product = df['asin'].value_counts()

plt.figure(figsize=(15,5))
# Visualize the distribution of the number of reviews across products using a histogram
plt.hist(reviews_per_product, bins=50)
plt.xlabel('Number of reviews')
plt.ylabel('Number of products')
plt.show()


# In[15]:


# Group the reviews by product ID and count the number of reviews per product
reviews_per_product_df = df.groupby('asin').size().reset_index(name='review_count')

plt.figure(figsize=(15,5))
# Visualize the distribution of the number of reviews per product using a box plot
sns.boxplot(x=reviews_per_product_df['review_count'])
plt.xlabel('Number of reviews per product')
plt.show()


# In[16]:


# Calculate the average number of reviews per product
reviews_per_product_df_ = df.groupby('asin').size()
avg_reviews_per_product = reviews_per_product_df_.mean()

plt.figure(figsize=(15,5))
# Plot the distribution of reviews per product
sns.histplot(reviews_per_product_df_/avg_reviews_per_product, bins=20)


# In[17]:


# Group the reviews by user ID and count the number of reviews per user
reviews_per_user = df.groupby('reviewerID').size().reset_index(name='review_count')

plt.figure(figsize=(15,5))
# Visualize the distribution of the number of reviews per user using a histogram
sns.histplot(x=reviews_per_user['review_count'])
plt.xlabel('Number of reviews per user')
plt.show()


# In[18]:


sns.kdeplot(reviews_per_user['review_count'])


# In[19]:


print("Average review per product:", avg_reviews_per_product)


# In[20]:


counts_of_reviews_per_product = df.groupby('asin').size()
for product, count_of_reviews_per_product in counts_of_reviews_per_product.iteritems():
  print(f'{product} has {count_of_reviews_per_product} reviews')
# counts_of_reviews_per_product


# In[21]:


len(counts_of_reviews_per_product)


# In[22]:


import matplotlib.pyplot as plt
import numpy as np

# Plot the distribution using matplotlib.pyplot.hist() function.
plt.hist(counts_of_reviews_per_product[:50], bins=np.arange(0, 100, 5))
plt.xlabel('Number of Reviews')
plt.ylabel('Frequency')
plt.title('Distribution of the Number of Reviews Across Products')
plt.show()


# In[23]:


counts_of_reviews_per_product[:10]


# In[24]:


plt.figure(figsize=(25,8))
plt.plot(counts_of_reviews_per_product[:10])
plt.xlabel('Product')
plt.ylabel('Number of reviews')
plt.title('Distribution of the number of reviews per product')
plt.show()


# In[25]:


counts_of_reviews_across_products = df.groupby(['asin', 'overall']).size()
# for product, count_of_reviews_per_product in counts_of_reviews_per_product.iteritems():
#   print(f'{product} has {count_of_reviews_per_product} reviews')
counts_of_reviews_across_products[:10]


# In[26]:


# Unstack the data to create a pivot table with product ids as rows and review ratings as columns
reviews_by_product_and_rating = counts_of_reviews_across_products[:10].unstack()

# Plot the distribution of the number of reviews per product per star rating as a histogram
reviews_by_product_and_rating.plot(kind='bar', stacked=True)
plt.xlabel('Product ID')
plt.ylabel('Number of reviews')
plt.title('Distribution of the number of reviews per product per star rating')
plt.show()


# In[27]:


counts_of_reviews_per_user = df.groupby('reviewerID').size()
for user, count_of_review in counts_of_reviews_per_user.iteritems():
  print(f'{user} has {count_of_review} reviews')


# In[28]:


plt.figure(figsize=(25,8))
plt.plot(counts_of_reviews_per_user[:10])
plt.xlabel('User')
plt.ylabel('Number of reviews')
plt.title('Distribution of the number of reviews per user')
plt.show()


# In[29]:


positive = df[df['overall'] > 3]
negative = df[df['overall'] < 3]
positive = positive.dropna()
negative = negative.dropna()


# In[31]:


import nltk
nltk.download('stopwords')


# In[32]:


# common words in positive review comments
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud

stopwords = set(stopwords.words('english'))

pos = " ".join(review for review in positive.reviewText)
wordcloud = WordCloud(stopwords=stopwords).generate(pos)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[33]:


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


# In[34]:


df['reviewText']


# ### Pre-processing

# In[35]:


# Initial trail deleting the rows with null values
'''
Verify the summary column is null if reviewText is null
'''

df.dropna(subset=['reviewText'], inplace=True)


# In[36]:


print("Count of null values after the action: ",df["reviewText"].isnull().sum(), "\n")
df.info()


# In[37]:


import random
n_samples = random.randint(500, 1000)
df_random = df.sample(n=n_samples)


# In[38]:


df_random.shape


# In[39]:


df_random


# In[40]:


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


# In[41]:


df_random


# In[42]:


# Chose the appropriate columns for your sentiment analyzer.
final_df = pd.DataFrame(df_random[['reviewText', 'label']]) 


# In[43]:


type(final_df)


# In[44]:


final_df


#     Text cleaning (removing punctuation, special characters, digits.)
#     Lowercasing the text
#     Tokenization (splitting the text into individual words or tokens)
#     Stopword removal (removing common words like "the" or "and")
#     Lemmatization (reducing words to their base form)

# In[45]:


import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import string

nltk.download('stopwords')
nltk.download('punkt')

words = []
for _, review in final_df.iterrows():
    # lowercase
    sentence = review['reviewText'].lower()
    # Remove Punctuations
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))
    # Remove Digits
    sentence = re.sub(r'\d+','', sentence)
    # Remove special characters
    sentence = re.sub("[^A-Z]", "", sentence, re.IGNORECASE)

    # Remove stop words
    stopwords = nltk.corpus.stopwords.words("english")
    sentence = " ".join([token for token in sentence.split() if (token not in stopwords)])
    
    # Tokenize the sentence into words
    words.append([word for word in nltk.word_tokenize(sentence)])


# In[46]:


words


# In[50]:


nltk.download('wordnet')
nltk.download('omw-1.4')


# In[51]:


# Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
#lemmatized_samples = final_df.copy()
lemmatized_samples = words

for i, text in enumerate(lemmatized_samples):
    text = [lemmatizer.lemmatize(word) for word in text]
    # Save to DataFrame
    lemmatized_samples[i] = text
    print(lemmatized_samples[i])


# In[52]:


# Store back into lemmatized_df
lemmatized_df = final_df.copy()
for i, row in enumerate(lemmatized_df["reviewText"]):
    lemmatized_df["reviewText"].iloc[i] = lemmatized_samples[i]


# In[53]:


lemmatized_df


# In[54]:


lemmatized_df["reviewText"].isnull().any()


# In[55]:


X_tfidf = []
for sent in lemmatized_samples:
    # Create the TF-IDF vectorizer object
    tfidf = TfidfVectorizer()

    # Fit and transform the words using the vectorizer object
    try:
        tfidf_matrix = tfidf.fit_transform(sent)
        X_tfidf.append(tfidf_matrix)
    except ValueError as e:
        print("ValueError:",e)
        X_tfidf.append([])

    # Print the TF-IDF matrix
    print(tfidf_matrix.toarray())


# ### Modeling

# #### VADR

# In[56]:


get_ipython().system('pip install vaderSentiment')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[57]:


# Valence Aware Dictionary and Sentiment Reasoner (VADR)
VADR_analyzer = SentimentIntensityAnalyzer()
# Pass the analyzer for head 5 rows
for index, row in final_df.head().iterrows():
    vs = VADR_analyzer.polarity_scores(row["reviewText"])
    full_text = row.reviewText
    if vs['neg'] > vs['pos']:
        sentiment = 'Negative'
    elif vs['pos'] > vs['neg']:
        sentiment= 'Positive'
    else:
        sentiment = 'Neutral'
    print(
        f"Index: {index}\n"+
        f"Sentimental Analysis Result: {vs}\n"+
        f"Predicted category: {sentiment}\n"+
        f"Actual category: {row.label}\n"+
        f"Full Text:\n{full_text}\n"+
        "-"*50)


# #### TextBlob

# In[58]:


get_ipython().system('pip install textblob')
from textblob import TextBlob


# In[59]:


# Print for head 5 rows
for index, row in final_df.head().iterrows():
    text = row["reviewText"]
    full_text = row.reviewText
    sentiment = ""
    
    if isinstance(text, str):
        wiki = TextBlob(text)
        if wiki.sentiment.polarity == 0:
            sentiment = 'Neutral'
        elif wiki.sentiment.polarity < 0 :
            sentiment = ('Negative')
        elif wiki.sentiment.polarity > 0 :
            sentiment = ('Positive')
        
        # Display results
        print(
        f"Index: {index}\n"+
        f"Sentimental Analysis Result: {wiki.sentiment}\n"+
        f"Predicted category: {sentiment}\n"+
        f"Actual category: {row.label}\n"+
        f"Full Text:\n{full_text}\n"+
        "-"*50)


# #### SENTIWORDNET

# In[60]:


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


# In[61]:


# Print for head 5 rows
for index, row in final_df.head().iterrows():
    review = row["reviewText"]
    full_text = row.reviewText
    sentiment = ""
    sentiment_score = sw_review_sentiment_score(review)
    
    if sentiment_score > 0:
        sentiment = ('Positive')
    elif sentiment_score < 0:
        sentiment = ('Negative')
    else:
        sentiment = ('Neutral')
    print(
            f"Index: {index}\n"+
            f"Sentimental Analysis Result: {sentiment_score}\n"+
            f"Predicted category: {sentiment}\n"+
            f"Actual category: {row.label}\n"+
            f"Full Text:\n{full_text}\n"+
            "-"*50)


# ### Validation

# In[103]:


# VADR
predicted_sentiments_vadr = []
for index, row in final_df.iterrows():
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


# In[104]:


count = (predicted_sentiments_vadr == df_random['label']).value_counts()
accuracy = count[True]/(count[True]+count[False]) * 100
accuracy


# In[105]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

confusion_matrix(df_random['label'], predicted_sentiments_vadr)


# In[106]:


print(classification_report(df_random['label'], predicted_sentiments_vadr))


# In[107]:


# TextBlob
predicted_sentiments = []
for text in list(final_df['reviewText']):
  if isinstance(text, str):
    wiki = TextBlob(text)
    predicted_sentiments.append(wiki.sentiment)


# In[108]:


predicted_ratings_txt = []
for predicted_sentiment in predicted_sentiments:
  if predicted_sentiment.polarity == 0:
    predicted_ratings_txt.append('Neutral')
  elif predicted_sentiment.polarity < 0 :
    predicted_ratings_txt.append('Negative')
  elif predicted_sentiment.polarity > 0 :
    predicted_ratings_txt.append('Positive')


# In[109]:


(predicted_ratings_txt == df_random['label']).value_counts()


# In[110]:


count = (predicted_ratings_txt == df_random['label']).value_counts()
accuracy = count[True]/(count[True]+count[False]) * 100
accuracy


# In[111]:


confusion_matrix(df_random['label'], predicted_ratings_txt)


# In[112]:


print(classification_report(df_random['label'], predicted_ratings_txt))


# In[113]:


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


# In[114]:


count = (predicted_sentiments_senti == df_random['label']).value_counts()
accuracy = count[True]/(count[True]+count[False]) * 100
accuracy


# In[115]:


confusion_matrix(df_random['label'], predicted_sentiments_senti)


# In[116]:


print(classification_report(df_random['label'], predicted_sentiments_senti))


# ### Validation with Preprocessed data

# In[117]:


# VADR
predicted_sentiments_vadr = []
for index, row in lemmatized_df.iterrows():
    # Pass analyzer
    vs = VADR_analyzer.polarity_scores(row["reviewText"])
    if vs['neg'] > vs['pos']:
      sentiment = 'Negative'
    elif vs['pos'] > vs['neg']:
      sentiment= 'Positive'
    else:
      sentiment = 'Neutral'
    predicted_sentiments_vadr.append(sentiment)

(predicted_sentiments_vadr == final_df['label']).value_counts()


# In[118]:


count = (predicted_sentiments_vadr == df_random['label']).value_counts()
accuracy = count[True]/(count[True]+count[False]) * 100
accuracy


# In[119]:


confusion_matrix(df_random['label'], predicted_sentiments_vadr)


# In[120]:


print(classification_report(df_random['label'], predicted_sentiments_senti))


# In[121]:


# TextBlob
predicted_sentiments = []
for text in list(lemmatized_df['reviewText']):
    wiki = TextBlob(" ".join(text))
    predicted_sentiments.append(wiki.sentiment)


# In[122]:


predicted_ratings_txt = []
for predicted_sentiment in predicted_sentiments:
  if predicted_sentiment.polarity == 0:
    predicted_ratings_txt.append('Neutral')
  elif predicted_sentiment.polarity < 0 :
    predicted_ratings_txt.append('Negative')
  elif predicted_sentiment.polarity > 0 :
    predicted_ratings_txt.append('Positive')


# In[123]:


(predicted_ratings_txt == final_df['label']).value_counts()


# In[124]:


count = (predicted_ratings_txt == df_random['label']).value_counts()
accuracy = count[True]/(count[True]+count[False]) * 100
accuracy


# In[125]:


confusion_matrix(df_random['label'], predicted_ratings_txt)


# In[126]:


print(classification_report(df_random['label'], predicted_ratings_txt))


# In[127]:


# SENTIWORDNET

predicted_sentiments_senti = []
# Classifing each review into positive, negative, or neutral sentiment.
for review in final_df['reviewText']:
    text = " ".join(review)
    sentiment_score = sw_review_sentiment_score(text)
    if sentiment_score > 0:
# #         print('\nPositive Review:', review)
        predicted_sentiments_senti.append('Positive')
    elif sentiment_score < 0:
#         print('\nNegative Review:', review)
        predicted_sentiments_senti.append('Negative')
    else:
#         print('\nNeutral Review:', review)
        predicted_sentiments_senti.append('Neutral')


# In[128]:


count = (predicted_sentiments_senti == final_df['label']).value_counts()
accuracy = count[True]/(count[True]+count[False]) * 100
print(count)
print(accuracy)


# In[129]:


confusion_matrix(df_random['label'], predicted_sentiments_senti)


# In[130]:


print(classification_report(df_random['label'], predicted_sentiments_senti))

