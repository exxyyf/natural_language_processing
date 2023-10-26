#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import requests
import pandas as pd

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KDTree


# In[2]:


url= 'https://www.toptal.com/faq'

soup = BeautifulSoup(requests.get(url).text, "html.parser")


# In[3]:


questions = []
answers = []

a_selector = 'body > main > section > section._1K1faYl9 > div '
q_selector = 'body > main > section > section > h3._3r3EnTQQ'

for x in soup.select(a_selector):
    answers.append(x.text.strip())

for x in soup.select(q_selector):
    questions.append(x.text.strip())


# In[4]:


print(len(questions))
print(len(answers))

data = pd.DataFrame({'q':questions, 'a':answers})
data.head()


# In[5]:


vectorizer = make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components=100), Normalizer())


# In[6]:


vectorizer.fit(pd.concat([data.q, data.a]))


# In[7]:


vectors = vectorizer.transform(data.q)


# In[8]:


vectors[0]


# In[9]:


vectors.shape


# In[10]:


index = KDTree(vectors)


# In[11]:


distances, indices = index.query(vectorizer.transform(['What can I use for payment?']), k=3)


# In[12]:


distances


# In[13]:


indices


# In[14]:


for d, i in zip(distances[0], indices[0]):
    print(d, i, data.q[i])


# In[15]:


distances, indices = index.query(vectorizer.transform(['Where is Toptal located?']), k=3)


# In[16]:


for d, i in zip(distances[0], indices[0]):
    print(d, i, data.q[i])


# In[17]:


indices[0][0]


# In[18]:


def respond(text):
    distances, indices = index.query(vectorizer.transform([text]), k=3)
    if distances[0][0] > 0.65:
        print(f"Unfortunately, I cannot answer this question yet. Maybe, you wanted to know '{data['q'][indices[0][0]]}'")
    else:
        print(data['a'][indices[0][0]])


# In[19]:


respond('Where is Toptal located?')


# In[20]:


respond('Where is the nearest restaurant?')


# In[21]:


respond('What if Toptal expert fails?')


# In[22]:


respond('Can I invite Toptal expert to the company?')


# In[23]:


respond('Who owns the results of work?')


# In[24]:


respond('Who owns the worker?')


# In[42]:


class ToptalFAQBot():
    def reply(self, text):
        self.text = text
        distances, indices = index.query(vectorizer.transform([self.text]), k=3)
        if distances[0][0] > 0.65:
            print(f"Unfortunately, I cannot answer this question yet. Maybe, you wanted to know '{data['q'][indices[0][0]]}'")
        else:
            print(data['a'][indices[0][0]])


# In[43]:


bot = ToptalFAQBot()
bot.reply('Who owns the results of work?')


# In[ ]:




