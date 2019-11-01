#!/usr/bin/env python
# coding: utf-8

# # PREDICTION MODEL

# ### Definition:
# #### Prediction algorithms are used to forecast future events based on historical data.

# <img src="Prediction.png" alt="prediction image" title="Prediction" />

# ### Car Price Prediction Data

# ### Problem Statement:
# A Chinese automobile company Geely Auto aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts.Specifically, they want to understand the factors affecting the pricing of cars in the American market, since those may be very different from the Chinese market.The company wants to know:
# 
# Which variables are significant in predicting the price of a car, how well those variables describe the price of a car.

# ### Business Goal:
# We are required to model the price of cars with the available independent variables so that the company can accordingly manipulate the design of the cars, the business strategy etc; to meet certain price levels based on the fitted model.
# Here we do this by applying Linear Regression model.

# ### Description of Dataset:
# The dataset contains information regarding the various factors influencing the price of a particular car.There are a total of 26 columns/attributes like carname,fueltype,enginelocation,horsepower,peakrpm,citympg etc also with the output variable/attribute price.There are a total of 205 observations. 
### Attribute Information:
Car_ID			    Unique id of each observation (Interger)
Symboling 			Its assigned insurance risk rating, A value of +3 indicates that the auto is risky, -3 that it is probably                     pretty safe.(Categorical)
carCompany			Name of car company (Categorical)
fueltype			Car fuel type i.e gas or diesel (Categorical)
aspiration			Aspiration used in a car (Categorical)
doornumber			Number of doors in a car (Categorical)		
carbody		    	body of car (Categorical)
drivewheel			type of drive wheel (Categorical)
enginelocation		Location of car engine (Categorical)
wheelbase			Weelbase of car (Numeric)
carlength			Length of car (Numeric)	
carwidth			Width of car (Numeric)	
carheight			height of car (Numeric)
curbweight			The weight of a car without occupants or baggage. (Numeric)
enginetype			Type of engine. (Categorical)
cylindernumber		cylinder placed in the car (Categorical)
enginesize			Size of car (Numeric)
fuelsystem			Fuel system of car (Categorical)
boreratio			Boreratio of car (Numeric)
stroke			    Stroke or volume inside the engine (Numeric)
compressionratio	compression ratio of car (Numeric)
horsepower			Horsepower (Numeric)
peakrpm			    car peak rpm (Numeric)
citympg			    Mileage in city (Numeric)
highwaympg			Mileage on highway (Numeric)
price    			Price of car (Numeric)(Dependent variable)
# In[1]:


# Importing necessary packages and functions required
import numpy as np # for numerical computations
import pandas as pd # for data processing,I/O file operations
import matplotlib.pyplot as plt # for visualization of different kinds of plots
get_ipython().run_line_magic('matplotlib', 'inline')
# for matplotlib graphs to be included in the notebook, next to the code
import seaborn as sns # for visualization 
import warnings # to silence warnings
warnings.filterwarnings('ignore')


# In[2]:


# import all libraries and dependencies for data visualization
pd.options.display.float_format='{:.4f}'.format
plt.rcParams['figure.figsize'] = [8,8]
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', -1) 
sns.set(style='darkgrid')


# In[3]:


# import all libraries and dependencies for machine learning
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[4]:


# Reading the automobile consulting company file on which analysis needs to be done
df_auto = pd.read_csv("D:\\20-6-19\\PROJECTS\\CAR PRICE PREDICTION\\CarPrice_Assignment.csv")
df_auto.head()


# #### Understanding the shape of the dataframe

# In[5]:


df_auto.shape


# This shows that there are 205 rows, 26 columns in the data.

# In[6]:


# information of the dataset
df_auto.info()


# In[7]:


# summary statistics of dataset
df_auto.describe()


# ### Cleaning the Data

# ### What is Data Cleaning ? :
# Data comes in all forms, most of it being very messy and unstructured. They rarely come ready to use. Datasets, large and small, come with a variety of issues- invalid fields, missing and additional values, and values that are in forms different from the one we require. In order to bring it to workable or structured form, we need to “clean” our data, and make it ready to use. Some common cleaning includes parsing, converting to one-hot, removing unnecessary data, etc.

# <img src="cleanvsuncleandata.png" alt="clean vs unclean image" title="Cleaning methods" />

# We need to do some basic cleansing activity in order to feed our model the correct data.

# In[8]:


# Dropping car_ID as it is just for reference and is of no use.
df_auto = df_auto.drop('car_ID',axis=1)


# In[9]:


# Missing Values % contribution in DF
df_null = df_auto.isna().mean().round(4) * 100
df_null.sort_values(ascending=False)


# In[10]:


# Outlier Analysis of target variable
plt.figure(figsize = [8,8])
sns.boxplot(data=df_auto['price'], orient="v", palette="Set2")
plt.title("Price Variable Distribution", fontsize = 14, fontweight = 'bold')
plt.ylabel("Price Range", fontweight = 'bold')
plt.xlabel("Continuous Variable", fontweight = 'bold')
plt.show()


# Insight:
# There are some car prices whose values are above 30,000 which can be termed as outliers but we will not remove them instead we will use standarization scaling.

# In[11]:


# Extracting Car Company from the CarName  
df_auto['CarName'] = df_auto['CarName'].str.split(' ',expand=True)


# In[12]:


df_auto.head(5)


# In[13]:


# Checking for Unique Car companies 
df_auto['CarName'].unique()


# We find that in car company names there are some typing mistakes which we will rename correctly as follows:\
# maxda = mazda\
# Nissan = nissan\
# porsche = porcshce\
# toyota = toyouta\
# vokswagen = volkswagen = vw

# In[14]:


# Renaming the typing errors in Car Company names
df_auto['CarName'] = df_auto['CarName'].replace({'maxda': 'mazda', 'nissan': 'Nissan', 'porcshce': 'porsche', 'toyouta': 'toyota', 
                            'vokswagen': 'volkswagen', 'vw': 'volkswagen'})


# In[15]:


# changing the datatype of symboling from integer to string as it is categorical variable as per the dictionary file
df_auto['symboling'] = df_auto['symboling'].astype(str)


# In[16]:


# To check if there are duplicates present in the dataset
df_auto.loc[df_auto.duplicated()]


# In[17]:


# Segregation/Seperation of Numerical and Categorical Variables/Columns in the dataset
cat_col = df_auto.select_dtypes(include=['object']).columns
num_col = df_auto.select_dtypes(exclude=['object']).columns
df_cat = df_auto[cat_col]
df_num = df_auto[num_col]


# In[18]:


df_cat.head()


# In[19]:


df_num.head()


# In[20]:


print(df_cat.shape)
print(df_num.shape)


# ### Visualising the Data

# In[21]:


# Visualizing number of cars for each car name in the dataset
plt.figure(figsize = [15,8])
ax=df_auto['CarName'].value_counts().plot(kind='bar',stacked=False, colormap = 'rainbow')
ax.title.set_text('Carcount')
plt.xlabel("Names of the Car",fontweight = 'bold')
plt.ylabel("Count of Cars",fontweight = 'bold')
plt.show()


# Toyota seems to be the most favoured cars, whereas Mercury seems to be the least favoured cars.

# In[22]:


# Visualizing the distribution of car prices
plt.figure(figsize=(8,8))

plt.title('Car Price Distribution Plot')
sns.distplot(df_auto['price'])
plt.show()


# The plots seems to be right skewed, the prices of almost all cars looks like less than 18000.

# In[23]:


# Pairplot for all the numeric variables
ax = sns.pairplot(df_auto[num_col])


# 1)carlength, carwidth, curbweight ,enginesize ,horsepower seems to have a positive correlation with the output variable price.\
# 2)carheight doesn't show any significant trend with the output variable price.\
# 3)citympg, highwaympg seem to have a significant negative correlation with the output variable price.

# In[24]:


# Boxplot for all the categorical variables
plt.figure(figsize=(20, 15))
plt.subplot(3,3,1)
sns.boxplot(x = 'doornumber', y = 'price', data = df_auto)
plt.subplot(3,3,2)
sns.boxplot(x = 'fueltype', y = 'price', data = df_auto)
plt.subplot(3,3,3)
sns.boxplot(x = 'aspiration', y = 'price', data = df_auto)
plt.subplot(3,3,4)
sns.boxplot(x = 'carbody', y = 'price', data = df_auto)
plt.subplot(3,3,5)
sns.boxplot(x = 'enginelocation', y = 'price', data = df_auto)
plt.subplot(3,3,6)
sns.boxplot(x = 'drivewheel', y = 'price', data = df_auto)
plt.subplot(3,3,7)
sns.boxplot(x = 'enginetype', y = 'price', data = df_auto)
plt.subplot(3,3,8)
sns.boxplot(x = 'cylindernumber', y = 'price', data = df_auto)
plt.subplot(3,3,9)
sns.boxplot(x = 'fuelsystem', y = 'price', data = df_auto)
plt.show()


# 1)DoorNumber isn't affecting the price much.\
# 2)The cars with fueltype as diesel are comparatively expensive than the cars with fueltype as gas.\
# 3)All the types of carbody is relatively cheaper as compared to hardtop,convertible carbody.\
# 4)The cars with rear enginelocation are way expensive than cars with front enginelocation.\
# 5)HigherEnd cars seems to have rwd drivewheel.\
# 6)Enginetype ohcv comes into higher price range cars.\
# 7)The price of car is directly proportional to no.of cylinders in most of the cases.

# In[25]:


# Visualizing some more variables
plt.figure(figsize=(25, 6))

plt.subplot(1,3,1)
plt1 = df_auto['cylindernumber'].value_counts().plot('bar')
plt.title('Number of cylinders')
plt1.set(xlabel = 'Number of cylinders', ylabel='Frequency of Number of cylinders')

plt.subplot(1,3,2)
plt1 = df_auto['fueltype'].value_counts().plot('bar')
plt.title('Fuel Type')
plt1.set(xlabel = 'Fuel Type', ylabel='Frequency of Fuel type')

plt.subplot(1,3,3)
plt1 = df_auto['carbody'].value_counts().plot('bar')
plt.title('Car body')
plt1.set(xlabel = 'Car Body', ylabel='Frequency of Car Body')
plt.show()


# 1)The number of cylinders used in most cars is four.\
# 2)Number of Gas fueled cars are way more than diesel fueled cars.\
# 3)Sedan is the most prefered car type.

# #### Derived Metrics
# We will use the mean of the car prices("Average Price") and visualize some variables using it.

# In[26]:


plt.figure(figsize=(50, 5))
df_autox = pd.DataFrame(df_auto.groupby(['CarName'])['price'].mean().sort_values(ascending = False))
df_autox.plot.bar()
plt.title('Car Company Name vs Average Price')
plt.show()


# Jaguar,Buick and porsche seems to have the highest average price.

# In[27]:


plt.figure(figsize=(20, 6))
df_autoy = pd.DataFrame(df_auto.groupby(['carbody'])['price'].mean().sort_values(ascending = False))
df_autoy.plot.bar()
plt.title('Carbody type vs Average Price')
plt.show()


# hardtop and convertible seems to have the highest average price.

# In[28]:


#Binning the Car Companies based on avg prices of each car Company using groupby and merge functions
df_auto['price'] = df_auto['price'].astype('int')
df_auto_temp = df_auto.copy()
t = df_auto_temp.groupby(['CarName'])['price'].mean()
df_auto_temp = df_auto_temp.merge(t.reset_index(), how='left',on='CarName')
bins = [0,10000,20000,40000]
label =['Budget_Friendly','Medium_Range','TopNotch_Cars']
df_auto['Cars_Category'] = pd.cut(df_auto_temp['price_y'],bins,right=False,labels=label)
df_auto.head()


# ### Significant variables after Visualization
# We find the follwing variables to be significant after all the visualizations\
# Cars_Category , Engine Type, Fuel Type\
# Car Body , Aspiration , Cylinder Number\
# Drivewheel , Curbweight , Car Length,\
# Car width , Engine Size,\
# Boreratio , Horse Power , Wheel base\
# citympg , highwaympg.

# In[29]:


sig_col = ['price','Cars_Category','enginetype','fueltype', 'aspiration','carbody','cylindernumber', 'drivewheel',
            'wheelbase','curbweight', 'enginesize', 'boreratio','horsepower', 
                    'citympg','highwaympg', 'carlength','carwidth']
df_auto = df_auto[sig_col]


# In[30]:


df_auto.shape


# ### Data Preparation
# #### Dummy Variables
# We need to convert the categorical variables to numeric.For this, we will use something called dummy variables.

# In[31]:


sig_cat_col = ['Cars_Category','enginetype','fueltype','aspiration','carbody','cylindernumber','drivewheel']


# In[32]:


# Get the dummy variables for the categorical feature and store it in a new variable - 'dummies'
dummies = pd.get_dummies(df_auto[sig_cat_col])
dummies.shape


# In[33]:


# To get k-1 dummies out of k categorical levels by removing the first level.
dummies = pd.get_dummies(df_auto[sig_cat_col], drop_first = True)
dummies.shape


# In[34]:


# Add the results to the original dataframe
df_auto = pd.concat([df_auto, dummies], axis = 1)
df_auto.shape


# In[35]:


df_auto.sample(5)


# In[36]:


# Drop the original cat variables as dummies are already created
df_auto.drop( sig_cat_col, axis = 1, inplace = True)
df_auto.shape


# ## Splitting the Data into Training and Testing Sets

# In[37]:


df_auto.sample(10)


# In[38]:


np.random.seed(0)
# We specify this so that the train and test data set always have the same rows, respectively
df_train, df_test = train_test_split(df_auto, test_size = 0.3, random_state = 100)
# We divide the dataframe into 70/30 ratio


# In[39]:


print(df_train.shape)
print(df_test.shape)


# ### Rescaling the Features
# For Simple Linear Regression, scaling doesn't impact model. So it is extremely important to rescale the variables so that they have a comparable scale. If we don't have comparable scales, then some of the coefficients as obtained by fitting the regression model might be very large or very small as compared to the other coefficients. There are two common ways of rescaling:
# 
# 1)Min-Max scaling\
# 2)Standardisation (mean-0, sigma-1)\
# Here, we will use Standardisation Scaling.

# In[40]:


scaler = preprocessing.StandardScaler()


# In[41]:


sig_num_col = ['wheelbase','carlength','carwidth','curbweight','enginesize','boreratio','horsepower','citympg','highwaympg','price']


# In[42]:


# Apply scaler() to all the columns except the 'dummy' variables
df_train[sig_num_col] = scaler.fit_transform(df_train[sig_num_col])


# In[43]:


df_train.sample(5)


# In[44]:


# Checking the correlation coefficients to see which variables are highly correlated
plt.figure(figsize = (25, 20))
sns.heatmap(df_train.corr(), cmap="RdYlBu",annot=True)
plt.show()


# From the above correlation heatmap we find that there a total of 13 variables which are highly correlated(both positively-9,negatively-4) with the price variable.

# #### Scatterplot for few correlated variables vs price.

# In[45]:


col = ['highwaympg','citympg','horsepower','enginesize','curbweight','carwidth']


# In[46]:


# Scatter Plot of independent variables vs dependent variables
plt.figure(figsize=(17, 10))
plt.subplot(2,3,1)
sns.scatterplot(x = 'highwaympg', y = 'price', data = df_auto)
plt.subplot(2,3,2)
sns.scatterplot(x = 'citympg', y = 'price', data = df_auto)
plt.subplot(2,3,3)
sns.scatterplot(x = 'horsepower', y = 'price', data = df_auto)
plt.subplot(2,3,4)
sns.scatterplot(x = 'enginesize', y = 'price', data = df_auto)
plt.subplot(2,3,5)
sns.scatterplot(x = 'curbweight', y = 'price', data = df_auto)
plt.subplot(2,3,6)
sns.scatterplot(x = 'carwidth', y = 'price', data = df_auto)
plt.show()


# We can see there is a line that we can fit in the above plots.

# In[47]:


# Dividing into X and Y sets for model building
y_train = df_train.pop('price')
X_train = df_train


# In[48]:


y_train.sample(2)


# In[49]:


X_train.sample(2)


# In[50]:


# Shapes of X_train,y_train
print(X_train.shape)
print(y_train.shape)


# ### Building a linear model

# In[51]:


# Building a simple linear model with the most highly correlated variable enginesize
X_train_1 = X_train['enginesize']
# Add a constant
X_train_1c = sm.add_constant(X_train_1)
# Create a first fitted model
lr_1 = sm.OLS(y_train, X_train_1c).fit()


# In[52]:


# Check parameters created
lr_1.params


# In[53]:


# Let's visualise the data with a scatter plot and the fitted regression line
plt.scatter(X_train_1c.iloc[:, 1], y_train)
plt.plot(X_train_1c.iloc[:, 1], 0.8679*X_train_1c.iloc[:, 1], 'g')
plt.show()


# In[54]:


# Print a summary of the linear regression model obtained
print(lr_1.summary())


# With simple linear regression i.e., enginesize and price we get adjusted R square value of 75%.

# ### Adding more variables
# The adjusted R-squared value obtained is 0.75. Since we have so many variables,let's add the other highly correlated variables, i.e. curbweight,horsepower.

# In[55]:


X_train_2 = X_train[['enginesize','horsepower', 'curbweight']]
# Add a constant
X_train_2c = sm.add_constant(X_train_2)
# Create a second fitted model
lr_2 = sm.OLS(y_train, X_train_2c).fit()


# In[56]:


lr_2.params


# In[57]:


print(lr_2.summary())


# The adjusted R-squared incresed from 0.75 to 0.81

# ### Considering all 13 correlated variables as from the correlation heatmap
# The adjusted R-squared value obtained with 3 highly correlated variables is 0.81. Since we have so many correlated variables, we can clearly do better than this. So lets consider all the 13 highly correlated variables in order(from high to low), i.e.,(positively correlated-9) enginesize,curbweight,horsepower,carwidth,Cars_Category_TopNotch_Cars,carlength,drivewheel_rwd,(negatively correlated-4) drivewheel_fwd,cylindernumber_four,citympg,highwaympg and fit the multiple linear regression model.

# In[58]:


X_train_3 = X_train[['enginesize', 'curbweight','horsepower', 'carwidth','Cars_Category_TopNotch_Cars','carlength','drivewheel_rwd','drivewheel_fwd','cylindernumber_four','citympg','highwaympg']]
# Add a constant
X_train_3c = sm.add_constant(X_train_3)
# Create a third fitted model
lr_3 = sm.OLS(y_train, X_train_3c).fit()


# In[59]:


lr_3.params


# In[60]:


print(lr_3.summary())


# We have achieved adjusted R-squared of 0.91 by manually picking the highly correlated variables.

# ### Making Predictions Using the Final Model
# Now that we have fitted the model, it's time to go ahead and make predictions using the final model.

# In[61]:


# Applying the scaling on the test sets
df_test[sig_num_col] = scaler.transform(df_test[sig_num_col])
df_test.shape


# In[62]:


# Dividing test set into X_test and y_test
y_test = df_test.pop('price')
X_test = df_test


# In[63]:


# Adding constant
X_test_1 = sm.add_constant(X_test)

X_test_new = X_test_1[X_train_3c.columns]


# In[64]:


# Making predictions using the final model
y_pred = lr_3.predict(X_test_new)


# ### Model Evaluation
# Let's now plot the graph for actual versus predicted values.

# In[65]:


# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=18)   
plt.xlabel('y_test ', fontsize=15)                       
plt.ylabel('y_pred', fontsize=15)    
plt.show()


# ### RMSE Score

# In[66]:


r2_score(y_test, y_pred)


# The R2 score of Training set is 0.91 and Test set is 0.89 which is very much close. Hence, we can say that our model is good enough to predict the Car prices with the above variables.

# ### Equation of Line to predict the Car prices values
# Carprice=0.0121+0.0427×enginesize+0.1971×curbweight+0.1961×horsepower+0.1642×carwidth+1.1336×Cars_Category_TopNotch+0.0480×carlength+0.1203×drivewheel_rwd-0.0262×drivewheel_fwd-0.2338×cylindernumber_four+0.0738×citympg-0.0423×highwaympg
