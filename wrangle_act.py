
# coding: utf-8

# # Set-up

# In[48]:


import tweepy
from tweepy import OAuthHandler
import json
from timeit import default_timer as timer
import pandas as pd
import os
import requests
import re
import numpy as np
import plotly
import plotly.offline as pyo
import plotly.graph_objs
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import *


from functools import reduce
from tweepy import OAuthHandler
from tweepy import API


# In[49]:


consumer_key = 'HIDDEN'
consumer_secret = 'HIDDEN'
access_token = 'HIDDEN'
access_secret = 'HIDDEN'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)


# # Retrieval from Udacity Servers

# In[50]:


folder_name = 'image_predictions'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv'
response = requests.get(url)

with open(os.path.join(folder_name,
                      url.split('/')[-1]), mode='wb') as file:
    file.write(response.content)

with open(os.path.join(folder_name,
                      url.split('/')[-1]), mode='wb') as file:
    file.write(response.content)

os.listdir(folder_name)


# # Enhanced Twitter Archive

# In[51]:


twitter_archive = pd.read_csv(r'twitter_archive_enhanced.csv')

tweets = list(twitter_archive['tweet_id'])


# # Twitter API Retrieval
# NOTE TO STUDENT WITH MOBILE VERIFICATION ISSUES:
# df_1 is a DataFrame with the twitter_archive_enhanced.csv file. You may have to
# change line 17 to match the name of your DataFrame with twitter_archive_enhanced.csv
# NOTE TO REVIEWER: this student had mobile verification issues so the following
# Twitter API code was sent to this student from a Udacity instructor
# Tweet IDs for which to gather additional data via Twitter's API
tweet_ids = twitter_archive.tweet_id.values
len(tweet_ids)

# Query Twitter's API for JSON data for each tweet ID in the Twitter archive
count = 0
fails_dict = {}
start = timer()
# Save each tweet's returned JSON as a new line in a .txt file
with open('tweet_json.txt', 'r') as outfile:
    # This loop will likely take 20-30 minutes to run because of Twitter's rate limit
    for tweet_id in tweet_ids:
        count += 1
        print(str(count) + ": " + str(tweet_id))
        try:
            #tweet = api.get_status(tweet_id, tweet_mode='extended')
            print("Success")
            json.dump('tweets._json.txt', outfile)
            outfile.read('\n')
        except tweepy.TweepError as e:
            print("Fail")
            fails_dict[tweet_id] = e
            pass
end = timer()
print(end - start)
print(fails_dict)
# In[52]:


df_list = []
with open('tweet_json.txt') as json_file:
    for line in json_file:
        json_data = json.loads(line)
        tweet_id = json_data['id']
        fav_count = json_data['favorite_count']
        ret_count = json_data['retweet_count']
        df_list.append({'tweet_id' : int(tweet_id),
                        'favorite_count' : int(fav_count),
                        'retweet_count' : int(ret_count)})


# In[53]:


tweet_ext_data = pd.DataFrame(df_list)
tweet_ext_data.head(5)


# In[54]:


tweet_ext_data.info()


# In[55]:


image_data = pd.read_csv(r'.\image_predictions\image-predictions.tsv', delim_whitespace=True)
image_data.head(5)


# In[56]:


image_data.info()


# In[57]:


twitter_archive


# In[58]:


twitter_archive.info()


# In[59]:


image_data.head(1)


# In[60]:


twitter_archive.info()


# ## Descriptive Statistics

# In[61]:


tweet_ext_data.describe()


# In[62]:


image_data.describe()


# In[63]:


twitter_archive.describe()


# # Define

# ### Quality Issues
# - Timestamp should be datetime type
# - Duplicated urls in column 'expanded_urls' of table twitter_archive
# - Non-dog names are used in the column name in twitter_archive
# - Column headers in image_date related to 'p1', 'p2', and 'p3' should be more descriptive
# - Inconsistent capitalization in column 'p1, 'p2', and 'p3'.
# - Numerator should be a float data type rather than an integer in 
# - In the source column there is the statement 'rel="nofollows">Twitter for iphone.
# - 24/7 is used as a numerator and denominator in the twitter_archive dataset. This belongs in the text field.
# 
# ### Tidiness Issues
# 
# - Incorrect organization of doggo, floofer, puppo, pupper columns in the twitter archive dataset
# - The columns in the image dataset related to picture identification should be merged in to a single p1, p2 or p3 column based on the tags.

# # Data Cleaning

# ## Copying and converting to a single data type

# ### Copying the dataframes

# In[64]:


cleaned_ext = tweet_ext_data.copy()
cleaned_image = image_data.copy()
cleaned_archive = twitter_archive.copy()


# ### Converting to a single data type

# In[65]:




cleaned_ext['tweet_id'] = cleaned_ext['tweet_id'].astype(str)
cleaned_image['tweet_id'] = cleaned_image['tweet_id'].astype(str)
cleaned_archive['tweet_id'] = cleaned_archive['tweet_id'].astype(str)


# # Code

# ### Cleaned_image dataset

# In[66]:


cleaned_image.head(1)


# ### Capitalizating p1, p2, p3

# In[67]:


list_p1 = list(cleaned_image.p1)
list_p2 = list(cleaned_image.p2)
list_p3 = list(cleaned_image.p3)


# ### Appending the list with a loop

# In[68]:


new_list_p1 = []
for name in list_p1:
    name = cleaned_image.p1.str.title()
    break
    
new_list_p2 = []
for name2 in list_p2:
    name2 = cleaned_image.p2.str.title()
    break

for name3 in list_p3:
    name3 = cleaned_image.p3.str.title()
    break


# ### Appending capitalizations to the dataframe

# In[69]:


cleaned_image.p1 = name
cleaned_image.p2 = name2
cleaned_image.p3 = name3


# ### Renaming columns in related to p1, p2, and p3 and combining

# In[70]:


cleaned_image['Picture_1_Accuracy'] = cleaned_image['p1'].astype(str) + ' | ' + cleaned_image['p1_conf'].astype(str) + ' | ' + cleaned_image['p1_dog'].astype(str)
cleaned_image['Picture_2_Accuracy'] = cleaned_image['p2'].astype(str) + ' | ' + cleaned_image['p2_conf'].astype(str) + ' | ' + cleaned_image['p2_dog'].astype(str)
cleaned_image['Picture_3_Accuracy'] = cleaned_image['p3'].astype(str) + ' | ' + cleaned_image['p3_conf'].astype(str) + ' | ' + cleaned_image['p3_dog'].astype(str)


# ### Dropping the originals

# In[71]:


del cleaned_image["p1"]
del cleaned_image['p1_conf']
del cleaned_image['p1_dog']

del cleaned_image["p2"]
del cleaned_image['p2_conf']
del cleaned_image['p2_dog']

del cleaned_image["p3"]
del cleaned_image['p3_conf']
del cleaned_image['p3_dog']


# ### Cleaned Archive Dataset

# In[72]:


cleaned_archive.head(1)


# ### Converting timestamps datatype

# In[73]:


cleaned_archive['timestamp'] = pd.to_datetime(cleaned_archive['timestamp'], yearfirst='%Y/%m/%d')


# ### Dropping duplicated urls in column expanded_urls

# In[74]:


cleaned_archive['expanded_urls'] = cleaned_archive['expanded_urls'].drop_duplicates()


# ### Converting the numerator column to a float type

# In[75]:


cleaned_archive['rating_numerator'] = cleaned_archive['rating_numerator'].astype(float)


# ### Dropping 24/7 in the numerator and denominator in the twitter archive dataframe

# In[76]:


cleaned_archive.drop(cleaned_archive.index[516], inplace=True)


# ### Cleaning the source url column

# In[77]:


urls = cleaned_archive['source']

def cleanhtml(urls):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', urls)
  return cleantext


# In[78]:


cleaned_archive['platform_used'] = list(map(cleanhtml, urls))


# ### Dropping the original source column

# In[79]:


cleaned_archive = cleaned_archive.drop('source', 1)


# In[80]:


cleaned_archive


# ### Creating a new column

# In[81]:


cleaned_archive['development'] = ''


# ### Replacing all None items with a blank

# In[82]:


cleaned_archive['doggo'] = cleaned_archive['doggo'].replace('None', '')
cleaned_archive['floofer'] = cleaned_archive['floofer'].replace('None', '')
cleaned_archive['pupper'] = cleaned_archive['pupper'].replace('None', '')
cleaned_archive['puppo'] = cleaned_archive['puppo'].replace('None', '')


# ### Moving doggo, floofer, pupper, puppo values to development column

# In[83]:


cleaned_archive['development'] = cleaned_archive['doggo'].str.cat(cleaned_archive[['floofer', 'pupper', 'puppo']])


# ### Dropping doggo, floofer, pupper, puppo

# In[84]:


cleaned_archive = cleaned_archive.drop('doggo', 1)
cleaned_archive = cleaned_archive.drop('floofer', 1)
cleaned_archive = cleaned_archive.drop('pupper', 1)
cleaned_archive = cleaned_archive.drop('puppo', 1)


# ### Removing non-dog names 'None', 'a', 'an', 'the'

# In[85]:


cleaned_archive = cleaned_archive[cleaned_archive.name != 'None']
cleaned_archive = cleaned_archive[cleaned_archive.name != 'a']
cleaned_archive = cleaned_archive[cleaned_archive.name != 'an']
cleaned_archive = cleaned_archive[cleaned_archive.name != 'the']


# # Test

# In[86]:


first = cleaned_ext.merge(right=cleaned_image, how = 'inner', on='tweet_id')
first


# In[87]:


master = first.merge(cleaned_archive, how='inner', on='tweet_id')
master


# In[88]:


master.to_csv('twitter_archive_master.csv', index=False, header=True)


# # Visuals 

# ## Favorite Count

# In[89]:


favorite_count = {'type' : 'bar',
                 'x' : master['timestamp'],
                 'y' : master['favorite_count'],
                 'name' : 'Favorite Count'}

layout = {'title' : 'Favorite Count Over Time',
         'yaxis' : {'title' : 'Favorite Count'},
         'xaxis' : {'title' : 'TimeStamp'},
         'annotations' : [{'xref' : 'paper',
                          'yref' : 'paper',
                          'x' : 0,
                          'y' : 5}]}

plotly.offline.iplot({
    'data' : [favorite_count],
    'layout' : layout})


# ## Retweet Count

# In[90]:


retweet_count = {'type' : 'bar',
                 'x' : master['timestamp'],
                 'y' : master['retweet_count'],
                 'name' : 'Retweet Count'}

layout = {'title' : 'Retweet Count Over Time',
         'yaxis' : {'title' : 'Retweet Count'},
         'xaxis' : {'title' : 'TimeStamp'},
         'annotations' : [{'xref' : 'paper',
                          'yref' : 'paper',
                          'x' : 0,
                          'y' : 5,
                          'showarrow' : False}]}

plotly.offline.init_notebook_mode(connected=True)

plotly.offline.iplot({
    'data' : [retweet_count],
    'layout' : layout})


# # Insights

# - The number of favorites and retweets generally incrase over time. 
# - Between October 2016 and April 2017 we see a huge spike in activity. 
# - We can see that users tend to favorite a tweet as apposed to retweeting. 
