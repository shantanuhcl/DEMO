#!/usr/bin/env python
# coding: utf-8

# # Training own word2vec model using gensim library on GOT books

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import gensim
import os


# In[3]:


from nltk import sent_tokenize
from gensim.utils import simple_preprocess
#Going to data_new for each file in it,saving the text in corpus and then sentence tokenization of corpus
#for each sentence simple preprocessing taking place & appending it to story
story = []
for filename in os.listdir('data_new'):
    
    f = open(os.path.join('data_new',filename))
    corpus = f.read()
    raw_sent = sent_tokenize(corpus)
    for sent in raw_sent:
        story.append(simple_preprocess(sent))
    


# In[4]:


len(story)


# In[5]:


story


# In[6]:


#min_count=2 means we are considering sentences with min 2 words
#
model = gensim.models.Word2Vec(
    window=10,
    min_count=2
)


# In[7]:


#building vocab:collection of unique words
model.build_vocab(story)


# In[8]:


#by default model.epochs value is 5
model.epochs


# In[9]:


model.train(story, total_examples=model.corpus_count, epochs=model.epochs)


# In[10]:


model.wv.most_similar('daenerys')


# In[11]:


model.wv.doesnt_match(['jon','rikon','robb','arya','sansa','bran'])


# In[12]:


model.wv.doesnt_match(['cersei', 'jaime', 'bronn', 'tyrion'])


# In[13]:


#Vector representation
model.wv['king']


# In[14]:


#similarity percentage
model.wv.similarity('arya','sansa')


# In[15]:


model.wv.similarity('cersei','sansa')


# In[16]:


model.wv.similarity('tywin','sansa')


# In[17]:


#Want to see the vector form of all the words
model.wv.get_normed_vectors()


# In[18]:


model.wv.get_normed_vectors().shape


# In[19]:


y = model.wv.index_to_key


# In[20]:


y


# In[21]:


from sklearn.decomposition import PCA


# In[22]:


pca = PCA(n_components=3)


# In[23]:


X = pca.fit_transform(model.wv.get_normed_vectors())


# In[24]:


#Observe dimensions got reduced to 3 from 100
X.shape


# In[25]:


import plotly.express as px
fig = px.scatter_3d(X[200:300],x=0,y=1,z=2, color=y[200:300])
fig.show()


# In[ ]:




