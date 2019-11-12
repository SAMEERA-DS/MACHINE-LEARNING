#!/usr/bin/env python
# coding: utf-8

# # MARKET BASKET ANALYSIS with NLP(WordCloud)

# In the retail sector, this analysis is used to discover patterns and relationships within transactions and transaction items-sets (Items frequently bought together).

# ## IBM puts it as:
# 
# "Market Basket Analysis is used to increase marketing effectiveness and to improve cross-sell and up-sell opportunities by making the right offer to the right customer. For a retailer, good promotions translate into increased revenue and profits. The objectives of the market basket analysis models are to identify the next product that the customer might be interested to purchase or to browse."

#  This explanation clearly gives the intent of running the analysis : "to improve cross-sell and up-sell opportunities.

# ## Apriori Algorithm:
# The algorithm was first proposed in <b>1994 by Rakesh Agrawal and Ramakrishnan Srikant</b>. Apriori algorithm finds the most frequent itemsets or elements in a transaction database and identifies association rules between the items.

# ## How Apriori works
# To construct association rules between elements or items, the algorithm considers 3 important factors which are, support, confidence and lift. Each of these factors is explained as follows:
# ### Support:
# The support of item I is defined as the ratio between the number of transactions containing the item I by the total number of transactions expressed as :
# <img src="support.png" alt="Support formula" title="Support" />
# ### Confidence:
# This is measured by the proportion of transactions with item I1, in which item I2 also appears. The confidence between two items I1 and I2,  in a transaction is defined as the total number of transactions containing both items I1 and I2 divided by the total number of transactions containing I1.
# <img src="confidence.png" alt="Confidence formula" title="Confidence" />
# ### Lift:
# Lift is the ratio between the confidence and support expressed as :
# <img src="lift.png" alt="Lift formula" title="Lift" />

# ## Example to understand the above concepts:
# Suppose we have a record of 1 thousand customer transactions, and we want to find the Support, Confidence, and Lift for two items e.g. burgers and ketchup.\
# Out of one thousand transactions, 100 contain ketchup while 150 contain a burger. Out of 150 transactions where a burger is purchased, 50 transactions contain ketchup as well.\
# Using this data, we want to find the support, confidence, and lift.
# ### Support:
# Support refers to the default popularity of an item.\
# Support(A) = (Transactions containing (A))/(Total Transactions)\
# For instance if out of 1000 transactions, 100 transactions contain Ketchup then the support for item Ketchup can be calculated as:
# 
# Support(Ketchup) = (Transactions containingKetchup)/(Total Transactions)\
# Support(Ketchup) = 100/1000 = 10%
# ### Confidence:
# Confidence refers to the likelihood that an item B is also bought if item A is bought.\
# Confidence(A→B) = (Transactions containing both (A and B))/(Transactions containing A)\
# We had 50 transactions where Burger and Ketchup were bought together. While in 150 transactions, burgers are bought.\
# Then we can find likelihood of buying ketchup when a burger is bought can be represented as confidence of Burger -> Ketchup and can be mathematically written as:
# 
# Confidence(Burger→Ketchup) = (Transactions containing both (Burger and Ketchup))/(Transactions containing Burger)\
# Confidence(Burger→Ketchup) = 50/150 = 33.3%
# ### Lift:
# Lift(A -> B) refers to the increase in the ratio of sale of B when A is sold.\
# Lift(A→B) = (Confidence (A→B))/(Support (B))\
# The Lift(Burger -> Ketchup) can be calculated as:
# 
# Lift(Burger→Ketchup) = (Confidence (Burger→Ketchup))/(Support (Ketchup))\
# Lift(Burger→Ketchup) = 33.3/10 = 3.33
# 
# 
# Lift basically tells us that the likelihood of buying a Burger and Ketchup together is 3.33 times more than the likelihood of just buying the ketchup.\
# A Lift of 1 means there is no association between products A and B. Lift of greater than 1 means products A and B are more likely to be bought together.\
# Finally, Lift of less than 1 refers to the case where two products are unlikely to be bought together.

# ### Dataset Description:
# This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a registered non-store online retail.The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.

# #### Attribute Information:
# 
# 1)InvoiceNo    : Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.\
# 2)StockCode    : Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.\
# 3)Description  : Product (item) name. Nominal.\
# 4)Quantity     : The quantities of each product (item) per transaction. Numeric.\
# 5)InvoiceDate  : Invoice Date and time. Numeric, the day and time when each transaction was generated.\
# 6)UnitPrice    : Unit price. Numeric, Product price per unit in sterling.\
# 7)CustomerID   : Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.\
# 8)Country      : Country name. Nominal, the name of the country where each customer resides.

# #### First we need to install the following modules to get apriori algorithm and the word cloud outputs.

# In[1]:


get_ipython().system(' pip install mlxtend')


# In[2]:


get_ipython().system(' pip install wordcloud')


# In[3]:


# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori,association_rules


# In[4]:


# Loading the data
data=pd.read_excel("D:\\20-6-19\\PROJECTS\\MARKET BASKET ANALYSIS\\OnlineRetailData.xlsx")
# glimpse of the dataset
data.head()


# In[5]:


# Columns of the dataset
data.columns


# In[6]:


# Shape of the data 
data.shape


# In[7]:


# Information of the dataset
data.info()


# In[8]:


# Missing values count
data.isna().sum()


# We have missing values in <b>CustomerId</b> as these are generated by unix time stamp they have no significance,hence we take <b>InvoiceNo</b> as key column.

# In[9]:


# Exploring the different unique countries of transactions
data.Country.unique()


# In[10]:


# Checking different number of unique countries present in this dataset
x = data['Country'].nunique()
print("There are {} number of unique countries.".format(x))


# In[11]:


# Total number of unique transactions
a=len(data['InvoiceNo'].unique())
print("There are a total of {} unique transactions.".format(a))


# In[12]:


# checking how many unique customer IDs are there
b = data['CustomerID'].nunique()
print("There are {} number of different customers".format(b))


# ### Cleaning the data

# In[13]:


# Removing the extra spaces in the description column using strip()
data['Description'] = data['Description'].str.strip()


# In[14]:


# Changing the type of invoice_no to string to help in removing the transactions cancelled/done on credit
data['InvoiceNo'] = data['InvoiceNo'].astype('str') 


# In[15]:


# Dropping all transactions which were cancelled/done on credit 
data = data[~data['InvoiceNo'].str.contains('C')] 


# In[16]:


# Checking the shape of the data after removing the transactions which were cancelled
data.shape


# ### Visualizing

# In[17]:


# Worcloud for description of all countries
from wordcloud import WordCloud
from wordcloud import STOPWORDS

stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'white', width = 1200, height = 1200).generate(str(data['Description']))

print(wordcloud)
plt.rcParams['figure.figsize'] = (8, 8)
plt.axis('off')
plt.imshow(wordcloud)
plt.title('Most Occuring word in the Description list', fontsize = 20)
plt.show()


# In[18]:


# checking count for countries in the dataset
plt.figure(figsize=(15,7))
sns.countplot(x=data.Country,order=data.Country.value_counts().iloc[:10].index)
#data['Country'].value_counts().head(10).plot.bar(figsize = (10, 7))
plt.title('Top 10 Countries having Online Retail Market', fontsize = 20)
plt.xlabel('Countries')
plt.ylabel('Count')
plt.show()


# In[19]:


# Having a look at the top 10 Countries in terms of Quantities according to the countries

data['Quantity'].groupby(data['Country']).agg('sum').sort_values(ascending = False).head(10).plot.bar(figsize = (10, 7))
plt.title('Top 10 Countries according to Quantity Sold Online', fontsize = 20)
plt.xlabel('Countries')
plt.ylabel('Number of Items Sold')
plt.show()


# ### Splitting the data according to the region of transaction (Taking three countries)

# In[20]:


# Transactions done in France 
basket_France = (data[data['Country'] =="France"] 
          .groupby(['InvoiceNo', 'Description'])['Quantity'] 
          .sum().unstack().reset_index().fillna(0) 
          .set_index('InvoiceNo')) 
basket_France.shape


# In[21]:


# Grouping description based on countries 
z=data['Description'].groupby(data['Country'])
p=[]
q=[]
for name,group in z:
    p.append(name)
    q.append(group)


# In[22]:


# Items description of France country through Wordcloud
wordcloud = WordCloud(background_color = 'white', width = 1200, height = 1200).generate(str(q[15]))
plt.imshow(wordcloud)


# In[23]:


# Transactions done in Portugal 
basket_Por = (data[data['Country'] =="Portugal"] 
          .groupby(['InvoiceNo', 'Description'])['Quantity'] 
          .sum().unstack().reset_index().fillna(0) 
          .set_index('InvoiceNo')) 
basket_Por.shape


# In[24]:


# Items description of Portugal country through Wordcloud
wordcloud = WordCloud(background_color = 'white', width = 1200, height = 1200).generate(str(q[29]))
plt.imshow(wordcloud)


# In[25]:


# Transactions done in Sweden
basket_Sweden = (data[data['Country'] =="Sweden"] 
          .groupby(['InvoiceNo', 'Description'])['Quantity'] 
          .sum().unstack().reset_index().fillna(0) 
          .set_index('InvoiceNo')) 
basket_Sweden.shape


# In[26]:


# Items description of Sweden country through Wordcloud
wordcloud = WordCloud(background_color = 'white', width = 1200, height = 1200).generate(str(q[34]))
plt.imshow(wordcloud)


# In[27]:


# Comparing item descriptions for different countries
def topicWordCloud(WCwidth, WCheight):
    wordcloud = WordCloud(background_color = 'white',width=WCwidth, height=WCheight,random_state=42).generate(str(q[i]))
    return wordcloud

fig = plt.figure(figsize=(40,40))
for i in range(3):
    ax = fig.add_subplot(1,3,i+1)
    wordcloud=topicWordCloud(1000,1000)
    ax.imshow(wordcloud)
    ax.axis('off')


# Here I have chosen 3 countries only,you can do this for any number of countries.

# ### Hot Encoding the data

# In[28]:


# Defining the hot encoding function to make the data suitable for the concerned libraries 
def hot_encode(x): 
    if(x<= 0): 
        return 0
    if(x>= 1): 
        return 1


# In[29]:


# Encoding the datasets (Considering only 3 countries..can be extended)
  
basket_encoded = basket_France.applymap(hot_encode) 
basket_France = basket_encoded
  
basket_encoded = basket_Por.applymap(hot_encode) 
basket_Por = basket_encoded 
  
basket_encoded = basket_Sweden.applymap(hot_encode) 
basket_Sweden = basket_encoded 


# ### Buliding the models and analyzing the results
# #### a) France:

# In[30]:


# Building the model 
frq_items = apriori(basket_France, min_support = 0.05, use_colnames = True) 
  
# Collecting the inferred rules in a dataframe 
rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rules.head()


# From the above output, it can be seen that paper cups and paper plates are bought together in France. This is because the French have a culture of having a get-together with their friends and family atleast once a week. Also, since the French government has banned the use of plastic in the country, the people have to purchase the paper -based alternatives.

# #### b) Portugal:

# In[31]:


# Building the model 
frq_items = apriori(basket_Por, min_support = 0.05, use_colnames = True) 

# Collecting the inferred rules in a dataframe 
rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rules.head()


# On analyzing the association rules for Portuguese transactions, it is observed that Tiffin sets (Knick Knack Tins) and colour pencils go together. These two products typically belong to a primary school going kid which are required to carry their lunch and for creative work respectively, hence logically make sense to be paired together.

# #### c) Sweden:

# In[32]:


# Building the model
frq_items = apriori(basket_Sweden, min_support = 0.05, use_colnames = True) 

# Collecting the inferred rules in a dataframe 
rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rules.head()


# On analyzing the above rules, it is found that boys’ and girls’ cutlery are paired together. This makes practical sense because when a parent goes shopping for cutlery for his/her children, he/she would want the product to be a little customized according to the kid’s wishes.

# In[ ]:




