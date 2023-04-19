# %% [markdown]
# ### Data Exploration

# %%
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

df = getDF("../../Office_Products_5.json.gz")

# %%
df

# %% [markdown]
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

# %%
df.info()

# %%
df.describe()

# Observation1: Average of review rating is around 4.47 which might imply that dataset contains more positive reviews.

# %%
df.shape

# %%
df.columns

# %%
print("Count of null values: ", df['reviewText'].isnull().sum())

# Observation2: Fetch the indexes of null values
nullIndexes = df[df['reviewText'].isnull()].index.tolist()
for index in (nullIndexes):
    print("Index:", index)

# %%
df["overall"].value_counts()

# Observation3: Dataset need to be balanced for ML approach

# %%
df[df["reviewText"].isnull()].overall.value_counts()

# # Observation4: Most of the null values in review column are for high ratings. 
# Options to handle null values: Filling with a constant value, delete those rows, or text imputation techniques

# %%
# Distribution of the number of reviews across products
# Group the reviews by product ID and count the number of reviews per product
reviews_per_product = df['asin'].value_counts()

plt.figure(figsize=(15,5))
# Visualize the distribution of the number of reviews across products using a histogram
plt.hist(reviews_per_product, bins=50)
plt.xlabel('Number of reviews')
plt.ylabel('Number of products')
plt.show()

# %%
# Group the reviews by product ID and count the number of reviews per product
reviews_per_product_df = df.groupby('asin').size().reset_index(name='review_count')

plt.figure(figsize=(15,5))
# Visualize the distribution of the number of reviews per product using a box plot
sns.boxplot(x=reviews_per_product_df['review_count'])
plt.xlabel('Number of reviews per product')
plt.show()

# %%
# Calculate the average number of reviews per product
reviews_per_product_df_ = df.groupby('asin').size()
avg_reviews_per_product = reviews_per_product_df_.mean()

plt.figure(figsize=(15,5))
# Plot the distribution of reviews per product
sns.histplot(reviews_per_product_df_/avg_reviews_per_product, bins=20)

# %%
# Group the reviews by user ID and count the number of reviews per user
reviews_per_user = df.groupby('reviewerID').size().reset_index(name='review_count')

plt.figure(figsize=(15,5))
# Visualize the distribution of the number of reviews per user using a histogram
sns.histplot(x=reviews_per_user['review_count'])
plt.xlabel('Number of reviews per user')
plt.show()

# %%
sns.kdeplot(reviews_per_user['review_count'])

# %%
print("Average review per product:", avg_reviews_per_product)

# %%
counts_of_reviews_per_product = df.groupby('asin').size()
for product, count_of_reviews_per_product in counts_of_reviews_per_product.iteritems():
  print(f'{product} has {count_of_reviews_per_product} reviews')
# counts_of_reviews_per_product

# %%
len(counts_of_reviews_per_product)

# %%
import matplotlib.pyplot as plt
import numpy as np

# Plot the distribution using matplotlib.pyplot.hist() function.
plt.hist(counts_of_reviews_per_product[:50], bins=np.arange(0, 100, 5))
plt.xlabel('Number of Reviews')
plt.ylabel('Frequency')
plt.title('Distribution of the Number of Reviews Across Products')
plt.show()

# %%
counts_of_reviews_per_product[:10]

# %%
plt.figure(figsize=(25,8))
plt.plot(counts_of_reviews_per_product[:10])
plt.xlabel('Product')
plt.ylabel('Number of reviews')
plt.title('Distribution of the number of reviews per product')
plt.show()

# %%
counts_of_reviews_across_products = df.groupby(['asin', 'overall']).size()
# for product, count_of_reviews_per_product in counts_of_reviews_per_product.iteritems():
#   print(f'{product} has {count_of_reviews_per_product} reviews')
counts_of_reviews_across_products[:10]

# %%
# Unstack the data to create a pivot table with product ids as rows and review ratings as columns
reviews_by_product_and_rating = counts_of_reviews_across_products[:10].unstack()

# Plot the distribution of the number of reviews per product per star rating as a histogram
reviews_by_product_and_rating.plot(kind='bar', stacked=True)
plt.xlabel('Product ID')
plt.ylabel('Number of reviews')
plt.title('Distribution of the number of reviews per product per star rating')
plt.show()

# %%
counts_of_reviews_per_user = df.groupby('reviewerID').size()
for user, count_of_review in counts_of_reviews_per_user.iteritems():
  print(f'{user} has {count_of_review} reviews')

# %%
plt.figure(figsize=(25,8))
plt.plot(counts_of_reviews_per_user[:10])
plt.xlabel('User')
plt.ylabel('Number of reviews')
plt.title('Distribution of the number of reviews per user')
plt.show()

# %%
positive = df[df['overall'] > 3]
negative = df[df['overall'] < 3]
positive = positive.dropna()
negative = negative.dropna()

# %%
# common words in positive review comments
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
#!pip install wordcloud
from wordcloud import WordCloud

stopwords = set(stopwords.words('english'))

pos = " ".join(review for review in positive.reviewText)
wordcloud = WordCloud(stopwords=stopwords).generate(pos)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# %%
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

# %%
df['reviewText']

# %% [markdown]
# ### Pre-processing

# %%
# Initial trail deleting the rows with null values
'''
Verify the summary column is null if reviewText is null
'''

df.dropna(subset=['reviewText'], inplace=True)

# %%
print("Count of null values after the action: ", df["reviewText"].isnull().sum(), "\n")
df.info()

# %%
# Count the ratings for each rating class
df["overall"].value_counts()

# %%
# Balance the dataset
balanced_df = df.groupby("overall")
balanced_df = balanced_df.apply(lambda x: x
                                .sample(balanced_df
                                        .size()
                                        .min())
                                .reset_index(drop=True))
# Count the balanced dataset ratings for each rating class
balanced_df.overall.value_counts()

# %%
# Drop the group
balanced_df = balanced_df.reset_index(drop=True)

# %%
# Chose the appropriate columns for your sentiment analyzer.
df_project = balanced_df.copy()[["reviewText", "overall"]]
df_project

# %%
# Label Engineering
def condition(overall):
  # print(df)
  rating = overall
  if rating in (4.0, 5.0):
    return 'Positive'
  elif rating == 3.0:
    return 'Neutral'
  elif rating in (1.0, 2.0):
    return 'Negative'

df_project['label'] = df_project['overall'].apply(condition)

# %%
df_project

# %% [markdown]
# Text cleaning
#     
#     Lowercasing the text
#     Removing punctuation, special characters, digits
#     Tokenization (splitting the text into individual words or tokens
#     Stopword removal (removing common words like "the" or "and")
#     Lemmatization (reducing words to their base form)

# %%
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')
nltk.download('punkt')

words = []
for _, review in df_project.iterrows():
    # lowercase
    sentence = review['reviewText'].lower()
    # Remove Punctuations
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))
    # Remove Digits
    sentence = re.sub(r'\d+','', sentence)
    # Remove special characters
    sentence = re.sub("[^A-Za-z0-9]", " ", sentence, re.IGNORECASE)

    # Remove stop words
    stopwords = nltk.corpus.stopwords.words("english")
    sentence = " ".join([token for token in sentence.split() if (token not in stopwords)])
    
    # Tokenize the sentence into words
    words.append([word for word in nltk.word_tokenize(sentence)])

# %%
words

# %%
# Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatized_samples = words

for i, text in enumerate(lemmatized_samples):
    text = [lemmatizer.lemmatize(word) for word in text]
    # Save to DataFrame
    lemmatized_samples[i] = text
    #print(lemmatized_samples[i])

# %%
# Store into lemmatized_df
lemmatized_df = df_project.copy()
lemmatized_df["reviewText"] = lemmatized_samples

# %%
lemmatized_df

# %%
lemmatized_df["reviewText"].isnull().any()

# %%
# Stratify Sample 500-1000 products
import random

n_samples = random.randint(500, 1000)
n_classes = len(lemmatized_df.label.unique())

df_random = lemmatized_df.groupby("label").apply(
    lambda x: x.sample(n=int(n_samples/n_classes))).reset_index(drop=True)
df_random

# %%
df_random.overall.value_counts()

# %%
df_random.label.value_counts()

# %%
# Convert tokenized text to string
str_review = []
for i, r in df_random.iterrows():
    str_corpus = " ".join(r.reviewText)
    str_review.append(str_corpus)
str_review

# %%
""" Vectorization would be done within the pipeline
# Verctorize with TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(str_review)
"""

# %%
"""tfidf.get_feature_names_out().shape"""

# %%
"""tfidf_matrix.shape"""

# %% [markdown]
# ### Modeling (Lexicon)

# %% [markdown]
# #### VADR

# %%
# !pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# %%
# Valence Aware Dictionary and Sentiment Reasoner (VADR)
VADR_analyzer = SentimentIntensityAnalyzer()
# Pass the analyzer for head 5 rows
for index, row in df_project.head().iterrows():
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

# %% [markdown]
# #### TextBlob

# %%
# !pip install textblob
from textblob import TextBlob

# %%
# Print for head 5 rows
for index, row in df_project.head().iterrows():
    text = row["reviewText"]
    sentiment = ""
    
    if isinstance(text, str):
        wiki = TextBlob(text)
        if wiki.sentiment.polarity == 0:
            sentiment = 'Neutral'
        elif wiki.sentiment.polarity < 0 :
            sentiment = ('Negative')
        elif wiki.sentiment.polarity > 0 :
            sentiment = ('Positive')
        else:
            print("nothing")
        
        # Display results
        print(
            f"Index: {index}\n"+
            f"Sentimental Analysis Result: {wiki.sentiment}\n"+
            f"Predicted category: {sentiment}\n"+
            f"Actual category: {row.label}\n"+
            f"Full Text:\n{full_text}\n"+
            "-"*50)

# %% [markdown]
# #### SENTIWORDNET

# %%
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

# %%
# Print for head 5 rows
for index, row in df_project.head().iterrows():
    review = row["reviewText"]
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

# %% [markdown]
# #### Validation with Un-processed data

# %%
# Stratified sample raw dataset
n_classes = len(df_project.label.unique())

df_random_r = df_project.groupby("label").apply(
    lambda x: x.sample(n=int(n_samples/n_classes))).reset_index(drop=True)
df_random_r

# %%
df_random_r.label.value_counts()

# %%
accuracy_list = []

# %%
# VADR
predicted_sentiments_vadr = []
for index, row in df_random_r.iterrows():
    # Pass analyzer
    vs = VADR_analyzer.polarity_scores(row["reviewText"])
    if vs['neg'] > vs['pos']:
      sentiment = 'Negative'
    elif vs['pos'] > vs['neg']:
      sentiment= 'Positive'
    else:
      sentiment = 'Neutral'
    predicted_sentiments_vadr.append(sentiment)

(predicted_sentiments_vadr == df_random_r['label']).value_counts()

# %%
count = (predicted_sentiments_vadr == df_random_r['label']).value_counts()
accuracy = count[True]/(count[True]+count[False]) * 100
accuracy

# %%
accuracy_list.append(('VADR', accuracy))

# %%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

confusion_matrix(df_random_r['label'], predicted_sentiments_vadr)

# %%
print(classification_report(df_random_r['label'], predicted_sentiments_vadr))

# %%
# TextBlob
predicted_sentiments = []
for text in list(df_random_r['reviewText']):
  text_str = " ".join(text)
  if isinstance(text_str, str):
    wiki = TextBlob(text_str)
    predicted_sentiments.append(wiki.sentiment)

# %%
predicted_ratings_txt = []
for predicted_sentiment in predicted_sentiments:
  if predicted_sentiment.polarity == 0:
    predicted_ratings_txt.append('Neutral')
  elif predicted_sentiment.polarity < 0 :
    predicted_ratings_txt.append('Negative')
  elif predicted_sentiment.polarity > 0 :
    predicted_ratings_txt.append('Positive')

# %%
(predicted_ratings_txt == df_random_r['label']).value_counts()

# %%
count = (predicted_ratings_txt == df_random_r['label']).value_counts()
accuracy = count[True]/(count[True]+count[False]) * 100
accuracy

# %%
accuracy_list.append(('TextBlob', accuracy))

# %%
confusion_matrix(df_random_r['label'], predicted_ratings_txt)

# %%
print(classification_report(df_random_r['label'], predicted_ratings_txt))

# %%
# SENTIWORDNET

predicted_sentiments_senti = []
# Classifing each review into positive, negative, or neutral sentiment.
for review in df_random_r['reviewText']:
    try:
        sentiment_score = sw_review_sentiment_score(review)
    except ZeroDivisionError:
        print("some corpus is empty")
        predicted_sentiments_senti.append('Neutral')
        continue
    
    if sentiment_score > 0:
# #         print('\nPositive Review:', review)
        predicted_sentiments_senti.append('Positive')
    elif sentiment_score < 0:
#         print('\nNegative Review:', review)
        predicted_sentiments_senti.append('Negative')
    else:
#         print('\nNeutral Review:', review)
        predicted_sentiments_senti.append('Neutral')

# %%
count = (predicted_sentiments_senti == df_random_r['label']).value_counts()
accuracy = count[True]/(count[True]+count[False]) * 100
accuracy

# %%
accuracy_list.append(('Sentiwordnet', accuracy))

# %%
confusion_matrix(df_random_r['label'], predicted_sentiments_senti)

# %%
print(classification_report(df_random_r['label'], predicted_sentiments_senti))

# %% [markdown]
# #### Validation with pre-processed data

# %%
preprocessed_accuracy_list = []

# %%
# VADR
predicted_sentiments_vadr = []
polarity_scores_vadr = []
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
    polarity_scores_vadr.append(vs)

(predicted_sentiments_vadr == df_random['label']).value_counts()

# %%
count = (predicted_sentiments_vadr == df_random['label']).value_counts()
accuracy = count[True]/(count[True]+count[False]) * 100
accuracy

# %%
preprocessed_accuracy_list.append(('VADR', accuracy))

# %%
confusion_matrix(df_random['label'], predicted_sentiments_vadr)

# %%
print(classification_report(df_random['label'], predicted_sentiments_vadr))

# %%
# TextBlob
predicted_sentiments = []
for text in list(df_random['reviewText']):
    wiki = TextBlob(" ".join(text))
    predicted_sentiments.append(wiki.sentiment)

# %%
predicted_ratings_txt = []
for predicted_sentiment in predicted_sentiments:
  if predicted_sentiment.polarity == 0:
    predicted_ratings_txt.append('Neutral')
  elif predicted_sentiment.polarity < 0 :
    predicted_ratings_txt.append('Negative')
  elif predicted_sentiment.polarity > 0 :
    predicted_ratings_txt.append('Positive')

# %%
(predicted_ratings_txt == df_random['label']).value_counts()

# %%
count = (predicted_ratings_txt == df_random['label']).value_counts()
accuracy = count[True]/(count[True]+count[False]) * 100
accuracy

# %%
preprocessed_accuracy_list.append(('TextBlob', accuracy))

# %%
confusion_matrix(df_random['label'], predicted_ratings_txt)

# %%
print(classification_report(df_random['label'], predicted_ratings_txt))

# %%
# SENTIWORDNET

predicted_sentiments_senti = []
# Classifing each review into positive, negative, or neutral sentiment.
for review in df_random['reviewText']:
    text = " ".join(review)
    try:
        sentiment_score = sw_review_sentiment_score(text)
    except ZeroDivisionError:
        print("some corpus is empty")
        predicted_sentiments_senti.append('Neutral')
        continue
    if sentiment_score > 0:
# #         print('\nPositive Review:', review)
        predicted_sentiments_senti.append('Positive')
    elif sentiment_score < 0:
#         print('\nNegative Review:', review)
        predicted_sentiments_senti.append('Negative')
    else:
#         print('\nNeutral Review:', review)
        predicted_sentiments_senti.append('Neutral')

# %%
count = (predicted_sentiments_senti == df_random['label']).value_counts()
accuracy = count[True]/(count[True]+count[False]) * 100
print(count)
print(accuracy)

# %%
preprocessed_accuracy_list.append(('Sentiwordnet', accuracy))

# %%
confusion_matrix(df_random['label'], predicted_sentiments_senti)

# %%
print(classification_report(df_random['label'], predicted_sentiments_senti))

# %% [markdown]
# ### Modeling (Machine Learning)

# %%
# Declare dataset
X = np.asarray(str_review)
y = df_random.label
X_raw = np.asarray(df_random_r.reviewText)

# %%
# Split into train, test dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    stratify=y)
xr_train, xr_test, yr_train, yr_test = train_test_split(X_raw, y,
                                                        test_size=0.3,
                                                        stratify=y)

# %%
print(
    "Shape of training dataset:", x_train.shape, y_train.shape,
    "\nShape of testing dataset:", x_test.shape, y_test.shape
)

# %%
print(
    "Shape of raw text training dataset:", xr_train.shape, yr_train.shape,
    "\nShape of raw text testing dataset:", xr_test.shape, yr_test.shape
)

# %% [markdown]
# #### Voting Classifier (With features pre-processed by our team)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression # REF (LogisticRegression Solvers): https://stackoverflow.com/questions/38640109/logistic-regression-python-solvers-definitions
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm, tree, metrics
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

'''
lr_C = LogisticRegression(max_iter = 1400)
rf_C = RandomForestClassifier()
# LinearSVC(random_state = 42)
svm_C = svm.SVC(probability=True, random_state = 42)
dt_C = tree.DecisionTreeClassifier(criterion="entropy", max_depth =42)
et_C = ExtraTreesClassifier()
'''

lr_C = LogisticRegression(class_weight='balanced',random_state = 0, max_iter = 1400)
rf_C = RandomForestClassifier(class_weight='balanced',
        criterion='entropy',
        max_depth=22,
        min_samples_split=4,
        n_estimators=250,
        random_state=56)
gb_C = GradientBoostingClassifier(n_estimators=250, learning_rate=1.0,
     max_depth=22, random_state=56)
# LinearSVC(random_state = 42)
svm_C = svm.SVC(probability=True, random_state = 42)
dt_C = tree.DecisionTreeClassifier(criterion="entropy", max_depth =22, class_weight='balanced', random_state = 42)
et_C = ExtraTreesClassifier(criterion="entropy", max_depth =22, class_weight='balanced')
mnb_C = MultinomialNB()

voting_clf = VotingClassifier(
    estimators=[('lr', lr_C), 
                ('rf', rf_C), 
                ('svm', svm_C), 
                ('dt',dt_C),
                ('et',et_C), 
                ('mnb', mnb_C), 
                ('gbc', gb_C)],
    voting='soft')

# %%
final_pipeline = Pipeline([("NLP", TfidfVectorizer()),
                           ('classifier', voting_clf)])

# %%
final_pipeline.fit(x_train, y_train)

# %%
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

y_pred = final_pipeline.predict(x_test)
print(f'Model score {final_pipeline.score(x_train, y_train)}')
print(f'Test score {final_pipeline.score(x_test, y_test)}')
	

"""
# convert categorical labels into binary labels
y_test_binary = lb.transform(y_test)
y_pred_binary = lb.transform(y_pred)

# from sklearn.metrics import plot_confusion_matrix, roc_auc_score, plot_roc_curve
print('Classifier ROC curve')
print(metrics.roc_curve(y_test_binary, y_pred_binary))
"""

cf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cf_matrix)

cf_report = metrics.classification_report(y_test, y_pred)
print(cf_report)


# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
cf_matrix

# %%
from seaborn import heatmap
plt.figure(figsize=(15,8))
ax = heatmap(cf_matrix, 
             annot=True, fmt=".0f", 
             xticklabels=np.unique(y_pred), yticklabels=np.unique(y_test))
ax.set_xlabel("Predicted")
ax.set_ylabel("True")

# %%
# Observe the performance of each models
# Vectorizing step
tfidf = final_pipeline.named_steps["NLP"]
tfidf.fit(x_train)
x_train_tr = tfidf.transform(x_train).toarray()
x_test_tr = tfidf.transform(x_test).toarray()

for name, clf in final_pipeline.named_steps["classifier"].named_estimators_.items():
    print( "-----------------", clf.__class__.__name__, "------------------" )

    # Estimator step
    print("training going on...")
    clf.fit(x_train_tr, y_train)

    y_pred_tr = clf.predict(x_test_tr)
    print(f'Model score {clf.score(x_train_tr, y_train)}')
    print(f'Test score {clf.score(x_test_tr, y_test)}')

    cf_matrix = metrics.confusion_matrix(y_test, y_pred_tr)
    print(cf_matrix)

    cf_report = metrics.classification_report(y_test, y_pred_tr)
    print(cf_report)



# %% [markdown]
# ### Voting Classifier (With TF-IDF built-in pre-proceccor)

# %%
lr_Cr = LogisticRegression(class_weight='balanced',random_state = 0, max_iter = 1400)
rf_Cr = RandomForestClassifier(class_weight='balanced',
        criterion='entropy',
        max_depth=22,
        min_samples_split=4,
        n_estimators=250,
        random_state=56)
gb_Cr = GradientBoostingClassifier(n_estimators=250, learning_rate=1.0,
     max_depth=22, random_state=56)
# LinearSVC(random_state = 42)
svm_Cr = svm.SVC(probability=True, random_state = 42)
dt_Cr = tree.DecisionTreeClassifier(criterion="entropy", max_depth =22, class_weight='balanced', random_state = 42)
et_Cr = ExtraTreesClassifier(criterion="entropy", max_depth =22, class_weight='balanced')
mnb_Cr = MultinomialNB()

voting_clf_r = VotingClassifier(
    estimators=[('lr', lr_Cr), 
                ('rf', rf_Cr), 
                ('svm', svm_Cr), 
                ('dt',dt_Cr),
                ('et',et_Cr), 
                ('mnb', mnb_Cr), 
                ('gbc', gb_Cr)],
    voting='soft')

# %%
rawtxt_pipeline = Pipeline([("NLP", TfidfVectorizer(stop_words='english',analyzer='word')),
                           ('classifier', voting_clf_r)])

# %%
rawtxt_pipeline.fit(xr_train, yr_train)

# %%
yr_pred = rawtxt_pipeline.predict(xr_test)
print(f'Model score {rawtxt_pipeline.score(xr_train, yr_train)}')
print(f'Test score {rawtxt_pipeline.score(xr_test, yr_test)}')

cf_matrix_r = metrics.confusion_matrix(yr_test, yr_pred)
print(cf_matrix_r)

cf_report_r = metrics.classification_report(yr_test, yr_pred)
print(cf_report_r)

# %%
from seaborn import heatmap
plt.figure(figsize=(15,8))
ax_r = heatmap(cf_matrix_r, 
        annot=True, fmt=".0f",
        xticklabels=np.unique(y_pred), yticklabels=np.unique(y_test))
ax_r.set(xlabel="Predicted",
        ylabel="True")
plt.show()

# %%
# Observe the performance of each models
# Vectorizing step
tfidf = rawtxt_pipeline.named_steps["NLP"]
tfidf.fit(xr_train)
x_train_tr = tfidf.transform(xr_train).toarray()
x_test_tr = tfidf.transform(xr_test).toarray()

for name, clf in rawtxt_pipeline.named_steps["classifier"].named_estimators_.items():
    print( "-----------------", clf.__class__.__name__, "------------------" )

    # Estimator step
    print("training going on...")
    clf.fit(x_train_tr, yr_train)

    y_pred_tr = clf.predict(x_test_tr)
    print(f'Model score {clf.score(x_train_tr, yr_train)}')
    print(f'Test score {clf.score(x_test_tr, yr_test)}')

    cf_matrix = metrics.confusion_matrix(yr_test, y_pred_tr)
    print(cf_matrix)

    cf_report = metrics.classification_report(yr_test, y_pred_tr)
    print(cf_report)



# %% [markdown]
# ### Rating Profile approach

# %%
# Load Libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# %%
# Copy the dataframe we use
df_rp = lemmatized_df.copy()
df_rp

# %%
df_rp.overall.value_counts()

# %%
# Extract adjectives and verbs as opinion words
opinion_words = []
for review in df_rp.reviewText:
    pos_tags = nltk.pos_tag(review)
    for word, pos in pos_tags:
        if pos.startswith('JJ') or pos.startswith('VB'):
            opinion_words.append(word)


# %%
# Determine the sentiment orientation of each opinion word
positive_reviews = df_rp[df_rp['label'] == 'Positive']['reviewText'].tolist()
negative_reviews = df_rp[df_rp['label'] == 'Negative']['reviewText'].tolist()
positive_opinion_words = []
negative_opinion_words = []

# %%
for word in opinion_words:
    freq_pos = sum([1 for review in positive_reviews if word in review])
    freq_neg = sum([1 for review in negative_reviews if word in review])
    if freq_pos > 0 or freq_neg > 0:
        ratio = freq_pos / freq_neg if freq_neg > 0 else freq_pos
        if ratio > 1:
            positive_opinion_words.append(word)
        elif ratio < 1:
            negative_opinion_words.append(word)

# %%
# Calculate the overall sentiment strength of each review
sentiment_strengths = []
for review in df_rp['reviewText']:
    sentiment_strength = 0
    num_opinion_words = 0
    for token in word_tokenize(review.lower()):
        if token in positive_opinion_words:
            sentiment_strength += 1
            num_opinion_words += 1
        elif token in negative_opinion_words:
            sentiment_strength -= 1
            num_opinion_words += 1
    if num_opinion_words > 0:
        overall_sentiment_strength = sentiment_strength / num_opinion_words
    else:
        overall_sentiment_strength = 0
    sentiment_strengths.append(overall_sentiment_strength)

# %%
# Map the overall sentiment strength to a corresponding rating on a 5-point scale
max_possible_sentiment_strength = len(positive_opinion_words) + len(negative_opinion_words)
ratings = []
for sentiment_strength in sentiment_strengths:
    if sentiment_strength > 0:
        rating = (sentiment_strength / max_possible_sentiment_strength) * 3 + 4
    elif sentiment_strength < 0:
        rating = (sentiment_strength / max_possible_sentiment_strength) * 3 + 2
    else:
        rating = 3
    ratings.append(rating)

# %%
# Add the ratings to the dataframe
df_rp['rating'] = ratings

# # Save the results to a CSV file
# reviews_df.to_csv('reviews_with_ratings.csv', index=False)

# %%
df_rp['enhanced_rating'] = df_rp[['overall', 'rating']].mean(axis=1)

# %%
sentiment_values = []
for i in df_rp['enhanced_rating'].values:
    if (i >= 4.):
        sentiment_values.append('positive')
    elif (i < 4. or i > 2.):
        sentiment_values.append('neutral')
    elif (i <= 2.):
        sentiment_values.append('negative')

df_rp['enhanced_sentiment'] = sentiment_values
df_rp.head()

# %%
xrp_train, xrp_test, yrp_train, yrp_test = train_test_split(df_rp['reviewText'], df_rp['enhanced_sentiment'], test_size=0.3, stratify=df_rp['label'])

print("Shape of training data: ", xrp_train.shape)
print("Shape of testing data: ", xrp_test.shape)
print("Shape of training labels: ", yrp_train.shape)
print("Shape of testing labels: ", yrp_test.shape)

# %%
lr_Crp = LogisticRegression(class_weight='balanced',random_state = 0, max_iter = 1400)
rf_Crp = RandomForestClassifier(class_weight='balanced',
        criterion='entropy',
        max_depth=22,
        min_samples_split=4,
        n_estimators=250,
        random_state=56)
gb_Crp = GradientBoostingClassifier(n_estimators=250, learning_rate=1.0,
     max_depth=22, random_state=56)
# LinearSVC(random_state = 42)
svm_Crp = svm.SVC(probability=True, random_state = 42)
dt_Crp = tree.DecisionTreeClassifier(criterion="entropy", max_depth =22, class_weight='balanced', random_state = 42)
et_Crp = ExtraTreesClassifier(criterion="entropy", max_depth =22, class_weight='balanced')
mnb_Crp = MultinomialNB()

voting_clfrp = VotingClassifier(
    estimators=[('lr', lr_Crp), 
                ('rf', rf_Crp), 
                ('svm', svm_Crp), 
                ('dt',dt_Crp),
                ('et',et_Crp), 
                ('mnb', mnb_Crp), 
                ('gbc', gb_Crp)],
    voting='soft')

# %%
rp_pipeline = Pipeline([("NLP", TfidfVectorizer()),
                           ('classifier', voting_clfrp)])

rp_pipeline.fit(xrp_train, yrp_train)

# %%
yrp_pred = rawtxt_pipeline.predict(xrp_test)
print(f'Model score {rawtxt_pipeline.score(xrp_train, yrp_train)}')
print(f'Test score {rawtxt_pipeline.score(xrp_test, yrp_test)}')

cf_matrix_rp = metrics.confusion_matrix(yrp_test, yrp_pred)
print(cf_matrix_r)

cf_report_rp = metrics.classification_report(yrp_test, yrp_pred)
print(cf_report_rp)

# %%
from seaborn import heatmap
plt.figure(figsize=(15,8))
ax_rp = heatmap(cf_matrix_rp, 
        annot=True, fmt=".0f",
        xticklabels=np.unique(yrp_pred), yticklabels=np.unique(yrp_test))
ax_rp.set(xlabel="Predicted",
        ylabel="True")
plt.show()

# %%



