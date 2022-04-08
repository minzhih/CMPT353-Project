#!/usr/bin/env python
# coding: utf-8

# In[91]:


import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_colwidth', None)
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import re
import nltk
import spacy
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud, STOPWORDS


# In[92]:


test = pd.read_csv("covid19_tweets.csv")


# In[93]:


##tweet test
tweet= test['text']
def convert(lst):
    return ([i for item in lst for i in item.split()])
panda_df = pd.DataFrame(data =test, columns = ["text","hashtags"])
panda_df=panda_df[panda_df['hashtags'].notnull()]
df = pd.DataFrame(convert(panda_df['hashtags']))
df=df.replace(',','', regex=True)
df=df.replace("]",'', regex=True)
df=df.replace('\'','', regex=True)
df=df.replace('\[''', '', regex=True)
df.columns=['Words']
s=df.value_counts()
Words=pd.DataFrame(s.nlargest(155))
modified = Words.reset_index()
modified.columns=['Words',"Counts"]
#print(modified) ## Mention count above 100
selection = modified['Words'].tolist()
Nna=test[test['hashtags'].notnull()]
## Hashtags which contains these top 100 words
mask = Nna.hashtags.apply(lambda x: any(item for item in selection if item in x))
Nna=Nna[mask]


# # Data Cleaning 

# In[94]:


##remove website
tweet= tweet.str.replace(r'http://[^\s<>"]+|www\.[^\s<>"]+', "") ##Ref https://stackoverflow.com/questions/10475027/extracting-url-link-using-regular-expression-re-string-matching-python
##remove mentions and the word followed 
tweet = tweet.str.replace(r'\s*@\s\w+', '', regex=True)
#remove hashtags
tweet = tweet.str.replace(r'\W*#\W+', '', regex=True)
##remove punctuation
tweet = tweet.str.replace('[^\w\s]','')
##remove numbers
tweet = tweet.str.replace('\S*\d\S*', '')
##remove underscore
tweet = tweet.str.replace(r"\W+_\W+", '', regex=True)
tweet = tweet.str.replace(r"_", '', regex=True)
##remove brackets
tweet = tweet.str.replace('\[.*?\]', '')
##convert to lower case
tweet= tweet.str.lower()
##drop empty rows
tweet.dropna(inplace=True)
##remove leading and ending white space
tweet = tweet.str.strip()


# # Word Cloud

# In[100]:


get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud, STOPWORDS
word_cloud = WordCloud(background_color='black',colormap='Set2',stopwords=set(STOPWORDS),max_words=50,width = 3000, height = 2000,random_state=1,collocations=False).generate(str(Nna))
fig = plt.figure(1, figsize=(10,10))
plt.axis('off')
fig.suptitle('Word Cloud for top 50 words', fontsize=20)
fig.subplots_adjust(top=2.3)
plt.imshow(word_cloud)
plt.show()


# # Perform sentiment analyze using NLTK(nlp) 

# In[80]:


# ref NLTK sentiment analyse https://www.nltk.org/howto/sentiment.html
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiment_analyzer = SentimentIntensityAnalyzer()
scores=[]
for i in test['text']:
    sentiment_score = sentiment_analyzer.polarity_scores(i)
    score = sentiment_score['compound']
    scores.append(score)
test['sentiment_score'] = scores
test
# todo: maybe we can label the scaores(positive, negative, neutral)


# In[79]:


data = test[['user_name','date','text','sentiment_score']]
data


# In[ ]:





# In[47]:


#convert time and do linear fitting 
import time
import datetime
for i in data['date']:
    time=data['date'][i]
    time= datetime.datetime.strptime(time,"%d/%m/%Y")
    data['date'][i] = time
    
def to_timestamp(t):
    return datetime.timestamp(t)

data['timestamp'] = data['date'].apply(to_timestamp)
fit = stats.linregress(data['timestamp'] , data['sentiment_score'])
data['prediction'] = data['timestamp']*fit.slope + fit.intercept
data_tweet

