import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

'''collecting data'''
data_set=pd.read_excel('file.csv.xlsx')
##print data_set.head()
print '# of passengers in original data : '+str(len(data_set.index))  
'''# of passengers in original data : 17'''

'''Analyse data '''

sns.countplot(x='Survived',data=data_set) #plot for how many survived and how many not survived

sns.countplot(x='Survived',hue='Gender',data=data_set) # M and F Survivers
sns.countplot(x='Survived',hue='Pclass',data=data_set) # based on Passenger class
data_set['Age'].plot.hist()
data_set['Fare'].plot.hist(figsize=(10,5))

plt.savefig('based on Fare')

data_set.info()   # to get info of total sheet of data
sns.countplot(x='SibSp',data=data_set)

plt.show()

''' Data wrangling '''

print data_set.isnull()

'''
# of passengers in original data : 17
    Passngrid  Survived  Pclass   Name  ...  Ticket   Fare  Cabin  Embarked
0       False     False   False  False  ...   False  False   True     False
1       False     False   False  False  ...   False  False  False     False
2       False     False   False  False  ...   False  False  False     False
3       False     False   False  False  ...   False  False  False     False
4       False     False   False  False  ...   False  False  False     False
5       False     False   False  False  ...   False  False   True     False
6       False     False   False  False  ...   False  False   True     False
7       False     False   False  False  ...   False  False   True     False
8       False     False   False  False  ...   False  False   True     False
9       False     False   False  False  ...   False  False   True     False
10      False     False   False  False  ...   False  False   True     False
11      False     False   False  False  ...   False  False  False     False
12      False     False   False  False  ...   False  False  False     False
13      False     False   False  False  ...   False  False   True     False
14      False     False   False  False  ...   False  False   True     False
15      False     False   False  False  ...   False  False  False     False
16      False     False   False  False  ...   False  False  False     False'''

print data_set.isnull().sum()

'''
Passngrid    0
Survived     0
Pclass       0
Name         0
Gender       0
Age          4
SibSp        0
Parch        0
Ticket       0
Fare         0
Cabin        9
Embarked     0
dtype: int64  '''

sns.heatmap(data_set.isnull(),yticklabels=False,cmap='magma')
plt.show()
'''   ## availble colors
Possible values are: Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap,
CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r,
PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r,
PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu,
RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r,
Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r,
autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r,
coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r,
gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r,
gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r,
hot, hot_r, hsv, hsv_r, icefire, icefire_r, inferno, inferno_r, jet, jet_r, magma, magma_r, mako, mako_r,
nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow,
rainbow_r, rocket, rocket_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20,
tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, viridis, viridis_r, vlag, vlag_r,
winter, winter_r'''

# remove the columns which has NULL values
data_set.drop('Cabin',axis=1,inplace=True)
data_set.dropna(inplace=True)
sns.heatmap(data_set.isnull(),yticklabels=False,cbar=False)   #just to check either all null values got removed using graph
print data_set.isnull().sum()  # check all columns if any null values contains

print pd.get_dummies(data_set['Gender'])
''' have to keep only one column so remove in next line of code
    female  male
0        0     1
1        1     0
2        1     0
3        1     0'''
sex=pd.get_dummies(data_set['Sex'],drop_first=True)   # removes first column
'''
    male
0      1
1      0
2      0
3      0'''
embark_=pd.get_dummies(data_set['Embarked'])
print embark_
'''
    C  Q  S
0   0  0  1
1   1  0  0
2   0  1  0
3   0  1  0'''
embark= pd.get_dummies(data_set['Embarked'],drop_first=True)
embark
'''
    Q  S
0   0  1
1   0  0
2   1  0
3   1  0
5   0  0
6   1  0'''
pd.get_dummies(data_set['Pclass'])
print pcl
'''
    1  2  3
0   0  1  0
1   1  0  0
2   0  1  0
3   1  0  0'''
pcl=pd.get_dummies(data_set['Pclass'],drop_first=True)
print pcl
'''
    2  3
0   1  0
1   0  0
2   1  0
3   0  0
5   0  1'''
print data_set.head(5)
data_set=pd.concat([data_set,sex,embark,pcl],axis=1)
print data_set.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis=1,inplace=True)
print data_set.head(5)

'''Train Data '''
'''================='''
#step 1 : have to define dependent and independent variable ;ArithmeticError so here predicting is passenger is Survived or not so Y is 'Survived'.i.e Dependent variable
# And rest all can consider as Independent varibles X

X=data_set.drop('Survived',axis=1)
y=data_set['Survived']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)  # 70 and 30 ratio

# create an instance for logistic regression model

logmodel = LogisticRegression()

# fit our model
logmodel.fit(X_train,y_train)

# Now predict our model
predictions=logmodel.predict(X_test)

# now have to evaluate how our model have been performing
# calculate the accuracy and classification report

from sklearn.metrics import classification_report

print classification_report(y_test,predictions)  # will generate some report
print X_test[10:11]
predict_person = logmodel.predict(X_test[10:11])

print 'myprediction',predict_person# to test a perticular value either 0 or 1
# calculate Accuracy
# concept of Accuracy : 2 by 2 matrix 
''' Calculate accuracy to check how accurate your results are '''

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,predictions)
print cm

'''
array[[105,21]
      [25,63]]'''

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test,predictions)  # 0.786  or 78%
 
print acc_score

import seaborn as sns
#plt.figsize(figsize=(9,9))
sns.heatmap(cm,annot=True,fmt =".3f",linewidths=.5,square=True,cmap='Blues_r')
plt.xlabel('Predicted label')
plt.ylabel('True label')
title = 'Accuracy Score : {0}'.format(acc_score)
plt.title(title,size = 14)
plt.savefig('CM_After_prediction')
plt.show()

'''
fmt ='.3f' is for converting 105 complex value to integer value in CM Diagram
'''
plt.savefig('checking_null_values_using_heat_map')
plt.show()

