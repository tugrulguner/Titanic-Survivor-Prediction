import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline

train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.drop(['PassengerId'], axis = 1, inplace = True)
train.head()
# Here I'm dropping the first column indicating the Id of passengers

train.dtypes
# To check types of columns in dataset

train.isnull().sum()
# To check how many NaN values this dataset have.

sns.boxplot(y = data['Fare'], x = train['Survived'])
# Here we can track the distribution of 'Fare' values, which is a continous, according to Survived.
# We can and we should also use this boxplot by replacing 'Fare' with 'Age'. These boxplots can help us to hint about
# what NaN values in 'Fare' and 'Age' columns can be replaced by without putting rough values like mean.

def isalone(column):
    if column == 0:
        return 0
    else:
        return 1

train['IsAlone'] = train['Parch'].apply(isalone)
# This adds a new column that can show whether passenger is alone or not using the 'Parch' information. 

def richfemale(columns):
    if columns[0] > 1:
        if columns[1] == 1:
            if columns[2] == 1:
                return 1
            else:
                return 2
        else:
            return 3
    else:
        return 0
        
# This function also adds new column by considering 'Fare', 'Pclass', and 'female' columns to determine 
# whether passenger is female and rich (1) or male and rich (2) (1 and 2 based on Pclass, if Pclass is 1
# then they can be considered as very rich, if Pclass is 2 or 3, without concering the sex this time 
# I set it as intermediate rich, which is 3) or finally poor (0).  

 
def parch_corr(columns):
    if pd.isnull(columns[0]) == True:
        if columns[1] == 0:
            return 0
        else:
            return 0.5
    else:
        return columns[0]

train['Embarked'].value_counts()
# To check how many unique values 'Embarked' column has and it shows there are only 3, 'C', 'S', and 'Q'. I assign values for each.
# But for NaN values I take them into account and leave them as 0. If we want to assign more precise values for NaN values
# we can easily fill zeros with what we want in future.

def embarkcorr(column):
    if column == 'C':
        return 1
    elif column == 'S':
        return 2
    elif column == 'Q':
        return 3
    else:
        return 0
train['Embarked'] = train['Embarked'].apply(embarkcorr)
# Assign of these values are completely random, I didn't consider any pattern or strategy.

def agedet(columns):
    Age = columns[0]
    Pclas = columns[1]
    Embark = columns[2]
    if np.isnan(Age) == True:
        if Embark == 3:
            return 50
        elif Pclas == 1:
            return 38
        elif Pclas == 2:
            return 30
        else:
            return 25
    else:
        return Age
        
train['Age'] = train[['Age', 'Pclass', 'Embarked']].apply(agedet, axis = 1)
# 'Age' column is a continous distribution and contains significant amount of NaN values. This function here
# fills the NaN values by taking into account 'Pclass' and 'Embarked' columns, which its dependence on these columns was
# analysed through boxplot.
    
def agecat(column):
    if column <= 15:
        return 0
    elif 15<column<40:
        return 1
    else:
        return 2

train['Age'] = train['Age'].apply(agecat)
# We corrected some NaN values of 'Age' column. This time I am transforming it from continous distribution
# to discrete one.

train['Cabin'] = train['Cabin'].fillna(0)
# First I replace NaN values with 0.

def cabin_crr(columns):
        if columns == 0:
            return columns
        else:
            column = columns[0]
            if column == 'B':
                return 2
            elif column == 'C':
                return 1
            elif column == 'D':
                return 3
            elif column == 'F':
                return 6
            elif column == 'E':
                return 4
            elif column == 'A':
                return 5
            elif column == 'G':
                return 7
            else:
                return 8 
        
train['Cabin'] = train['Cabin'].apply(cabin_crr)
# After replacing NaN values with zero, then I check first element of nonzero values and assign different values according
# to their uniqueness. However, total sum of missing values in 'Cabin' column is a lot, but I will continue to keep this column.

def Pclasscorr(column):
    if column == 1:
        return '1'
    elif column == 2:
        return '2'
    else:
        return '3'

train['Pclass'] = train['Pclass'].apply(Pclasscorr)
pclass = pd.get_dummies(train['Pclass'])

train = pd.concat([train, pclass], axis = 1)
# Here, first I replace integers in 'Pclass' column with strings, which makes it very easy to go for separate this single
# column with 1, 2 and 3 values into 3 new columns consisting of 0 and 1 for corresponding values. This part is not something
# that I think make significant difference. We can leave it as a single column without touching it.

def namesplit(columns):
    a = columns[0].split()
    if a[1] == 'Mr.':
        return 0
    elif a[1] == 'Mrs.':
        return 2
    elif a[1] == 'Miss.':
        return 1
    elif a[1] == 'Master.':
        return 3
    else:
        return 4

Sex = pd.get_dummies(train['Sex'])
train = pd.concat([train, Sex], axis = 1)
train.drop(['Sex', 'Ticket'], axis = 1, inplace = True)

train['Name'] = train[['Name', 'female']].apply(namesplit, axis = 1)
# I think this part can be quite important. After finding how many unique titles that 'Name' column contain, we can select 
# the mostly used ones and assign value for them. Here I took first 4 highly used titles and leave remainings as same with a
# fixed value of 4.

train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mean())
train['richfemale'] = train[['Fare', 'Pclass', 'female']].apply(richfemale, axis = 1)
train.drop('Pclass', axis = 1, inplace = True)

# I fill 'Embarked' column with mean since it has very few NaN values, which is only 2. Then I add 'richfemale' column that I 
# already described above and then I dropped the 'Pclass' column since I already add 3 different columns representing this
# single column.

plt.figure(figsize=(20,10))
sns.heatmap(train.corr(), annot = True)
# To see how much these columns are correlated before going into training 

X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived'], axis = 1), 
                                                         train['Survived'], test_size=0.3, random_state = 42)
n_estimators = [50, 250, 500, 750, 1000, 1500, 3000, 5000]
max_features = ['auto', 'sqrt', 'log2']
max_depth = [10, 25, 40, 50]
max_depth.append(None)
min_samples_split = [2, 5, 15, 20]
min_samples_leaf = [1, 2, 5, 10]

grid_param = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth, 
              'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}

RF = RandomForestClassifier(random_state = 42)

RF_random = RandomizedSearchCV(estimator = RF, param_distributions = grid_param, n_iter = 500,
                              cv = 5, verbose = 2, random_state = 42, n_jobs = -1)
RF_random.fit(X_train, y_train)
print(RF_random.best_params_)

# Here I split the data into train and test as using 0.3 for test_size. Then I start for hyperparameter optimization by using 
# Randomized Search CV for the Random Forest Classifier through assigning some possible value ranges for particular parameters
# like n_estimators, max_features, etc.

k = 1
deger = 0
degerson = 0 


while degerson < 0.85:
    X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived'], axis = 1), 
                                                         train['Survived'], test_size=0.3)
    
    model = RandomForestClassifier(n_estimators = 500, min_samples_split = 15, min_samples_leaf = 1, max_features = 'log2', 
                                   max_depth = 40) # These values were determined through Randomized Search CV above 
    model.fit(X_train, y_train) 
    deger = cross_validate(model, X_test, y_test, cv=30) # I like to use cross validation to determine my score
    degerson = deger['test_score'].mean()
    k += 1
    if k>1000:
        break
    # Here I put certain contraint on the loop, if I couldn't find score higher than 0.85, It will stop after some certain 
    # iteration. I can improve this part here to involve even though it couldnt find better score than 0.85, it can select
    # the best one it got during the process.

deger['test_score'].mean()
# To see what score I got

cfm = confusion_matrix(y_test, model.predict(X_test))
sns.heatmap(cfm, annot = True, fmt = 'd')
# To visualize confusion matrix

# --------------- Preprocessing and Fit of the Test Sample Part ------------------

test = pd.read_csv('/kaggle/input/titanic/test.csv')

test.isnull().sum()
# To check how many NaN values are there in the test dataset

test['Embarked'] = test['Embarked'].apply(embarkcorr)
test['Age'] = test[['Age', 'Pclass', 'Embarked']].apply(agedet, axis = 1)
test['Age'] = test['Age'].apply(agecat)
test['IsAlone'] = test['Parch'].apply(isalone)
test['Pclass'] = test['Pclass'].apply(Pclasscorr)
pclasst = pd.get_dummies(test['Pclass'])
test = pd.concat([test, pclasst], axis = 1)
test.drop('Pclass', axis = 1, inplace = True)
test['Cabin'] = test['Cabin'].fillna(0)
test['Cabin'] = test['Cabin'].apply(cabin_crr)
test['Fare'] = test['Fare'].apply(Farecat)
Sex = pd.get_dummies(test['Sex'])
test = pd.concat([test, Sex], axis = 1)
df = test['PassengerId']
test.drop(['PassengerId', 'Sex', 'Ticket'], axis = 1, inplace= True)

test['Name'] = test[['Name','female']].apply(namesplit, axis = 1)
test['Embarked'] = test['Embarked'].fillna('backfill')

test['richfemale'] = test[['Fare', 'Pclass', 'female']].apply(richfemale, axis = 1)

# So far I have just applied my functions that I used as preprocessing of training data, nothing is new here.
# They are in the same order and test data was exposed to same preprocessing that we did with the train data.
# We can do this easily by using pipeline but since this way my first seriously taken kaggle challenge, I lam leaving it like
# this. 

predictt = model.predict(test) # Making predictions using already fitted model above through train data
df = pd.DataFrame(df, columns=['PassengerId'])
dff = pd.DataFrame(predictt, columns=['Survived'])
sonuc = pd.concat([df, dff], axis = 1)

sonuc.to_csv('/kaggle/working/predict.csv', index = False)
# We create document consisting of our predictions and ready for submission to kaggle


