# **Project: Risk Analysis**

<br>

## **Abstract:**

#prediction #bigdata #tensorflow #neuralnetwork #featureengeneering

Our goal is try to predict the ability of a client to repay a loan.
This is a first and complete long project using Tensorflow to use Big Data and a lot of feature engineering. 

An extensive Exploratory Data Analysis was carried out and showed that some features as 'total number of credit lines currently in the borrower's credit file' and 'Number of derogatory public records' have direct correlation with our target. 

To predict, was createed a Deep Learning Model that seems work just well. To improve the performance was needed to try different parameters for the model. At the end we find that imbalanced data could bring down our results. We treated through iversampling and the model very similar. At the end we compare the evaluation of the neural network model with a more simple random forest classification, what also have performed really close the neural network.

<br>

## **Relevant info on the subject matter:**

[All Lending Club loan data, by Nathan George.](https://www.kaggle.com/wordsforthewise/lending-club)

<br>

## **Walk through:**

1. [Business Issue Understanding](#1);
2. [Data Understanding](#2);
3. [EDA - Exploratory Data Analysis](#3);
4. [Data Selecting/Preparation/Feature Engineering](#4);
5. [Modeling](#5);
6. [Evaluation](#6);
7. [Return and adjust if necessary](#7);
8. [Results/Findings](#8);
9. [Conclusion/Discussion and next steps](#9).
   
<br>

## **Author:**

Luiz Furtado <br>
[LinkedIn](https://www.linkedin.com/in/luiz-furtado-dev/) | [GitHub](https://github.com/TSLSouth)

<br>

---

<br>

## **1. Business Understanding** <a id='1'></a>

<br>

## _1.1 Assess Situation:_
LendingClub is a US peer-to-peer lending company, headquartered in San Francisco, California. It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. LendingClub is the world's largest peer-to-peer lending platform.

<br>

## _1.2 Goal:_

Given historical data on loans out with information on whether or not the borrower defaulted (charge-off), we will build a model that can predict wether or nor a borrower will pay back their loan. 

<br>

## _1.3 Data Mining Goals:_

That way in the future when we get a new potential customer we can assess whether or not they are likely to pay back the loan.

This is a binary classification problem.

<br>

## _1.4 Project Plan:_

Use the open data that come from the Lending Club.

Do a Exploratory Data Analysis to understand what makes a good payer.

Use Keras to apply a classification model to predicit when a borrower will repay or not.

<br>

---

<br>

## **2. Data Understanding** <a id='2'></a>

<br>

## _2.1 Attribute Information:_
  
  <br>

```python
df_dic = pd.read_csv('lending_club_info.csv',index_col='LoanStatNew')
df_dic
```
Description LoanStatNew

loan_amnt: The listed amount of the loan applied for by t...

term: The number of payments on the loan. Values are...

int_rate: Interest Rate on the loan

installment: The monthly payment owed by the borrower if th...

grade: LC assigned loan grade

sub_grade: LC assigned loan subgrade

emp_title: The job title supplied by the Borrower when ap...

emp_length: Employment length in years. Possible values ar...

home_ownership: The home ownership status provided by the borr...

annual_inc: The self-reported annual income provided by th...

verification_status: Indicates if income was verified by LC, not ve...

issue_d: The month which the loan was funded

loan_status: Current status of the loan

purpose: A category provided by the borrower for the lo...

title: The loan title provided by the borrower

zip_code: The first 3 numbers of the zip code provided b...

addr_state: The state provided by the borrower in the loan...

dti: A ratio calculated using the borrower’s total ...

earliest_cr_line: The month the borrower’s earliest reported cre...

open_acc: The number of open credit lines in the borrowe...

pub_rec: Number of derogatory public records

revol_bal: Total credit revolving balance

revol_util: Revolving line utilization rate, or the amount...

total_acc: The total number of credit lines currently in ...

initial_list_status: The initial listing status of the loan. Possib...

application_type: Indicates whether the loan is an individual ap...

mort_acc: Number of mortgage accounts.

pub_rec_bankruptcies: Number of public record bankruptcies

<br>
<br>

## _2.2 Useful tools to this data set:_
  
  <br>

We can use this tool to grab what's any feature description any time:

```python
print(df_dic.loc['revol_util']['Description'])

Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.
```

Or a function to do the same:

```python
def feat_info(col_name):
    '''func to grab any feature description any time'''
    print(df_dic.loc[col_name]['Description'])

feat_info('mort_acc')

Number of mortgage accounts.
```

<br>

Describe:

```python
df = pd.read_csv('lending_club_loan_two.csv')
df.head()
```

![img describe](https://github.com/TSLSouth/Risk-Analysis-Prediction-with-Tensorflow/blob/main/img/describe.png?raw=true)

<br>

There's missing data and some categorical features:

```python
df.info()

RangeIndex: 396030 entries, 0 to 396029
Data columns (total 27 columns):
 #   Column                Non-Null Count   Dtype  
---  ------                --------------   -----  
 0   loan_amnt             396030 non-null  float64
 1   term                  396030 non-null  object 
 2   int_rate              396030 non-null  float64
 3   installment           396030 non-null  float64
 4   grade                 396030 non-null  object 
 5   sub_grade             396030 non-null  object 
 6   emp_title             373103 non-null  object 
 7   emp_length            377729 non-null  object 
 8   home_ownership        396030 non-null  object 
 9   annual_inc            396030 non-null  float64
 10  verification_status   396030 non-null  object 
 11  issue_d               396030 non-null  object 
 12  loan_status           396030 non-null  object 
 13  purpose               396030 non-null  object 
 14  title                 394275 non-null  object 
 15  dti                   396030 non-null  float64
 16  earliest_cr_line      396030 non-null  object 
 17  open_acc              396030 non-null  float64
 18  pub_rec               396030 non-null  float64
 19  revol_bal             396030 non-null  float64
...
 25  pub_rec_bankruptcies  395495 non-null  float64
 26  address               396030 non-null  object 
dtypes: float64(12), object(15)
```

<br>

---

<br>

## **3. EDA - Data Exploratory Data Analysis** <a id='3'></a>

<br>

Let's take a overview on loan status:

```python
sns.countplot(x='loan_status',data=df)
```

![img loan status](https://github.com/TSLSouth/Risk-Analysis-Prediction-with-Tensorflow/blob/main/img/loan%20status.png?raw=true)

<br>

Loan amount count:

```python
plt.figure(figsize=(16,6))
sns.histplot(df['loan_amnt'],bins=40)
```

![img loan amount count](https://github.com/TSLSouth/Risk-Analysis-Prediction-with-Tensorflow/blob/main/img/loan%20amount%20count.png?raw=true)

<br>

Correlation heatmap:

```python
plt.figure(figsize=(12,12))
sns.heatmap(df.corr(),annot=True,cmap='viridis')
```

![img corr](https://github.com/TSLSouth/Risk-Analysis-Prediction-with-Tensorflow/blob/main/img/corr.jpeg?raw=true)

<br>

Correlation between loan total amount and installment. 

```python
plt.figure(figsize=(6,6))
sns.scatterplot(x='installment',y='loan_amnt',data=df)
```

![img installment](https://github.com/TSLSouth/Risk-Analysis-Prediction-with-Tensorflow/blob/main/img/installment.png?raw=true)

<br>

Loan status vs. loan amount shows no significant difference on results to when fully paid or charge off:

```python
plt.figure(figsize=(6,6))
sns.boxplot(x='loan_status',y='loan_amnt',data=df)
```

![img status vs. amount](https://github.com/TSLSouth/Risk-Analysis-Prediction-with-Tensorflow/blob/main/img/status%20vs%20amount.png?raw=true)

<br>

This correlation above described:

```python
df.groupby('loan_status')['loan_amnt'].describe()
```

![img status vs. amount describe](https://github.com/TSLSouth/Risk-Analysis-Prediction-with-Tensorflow/blob/main/img/status%20vs%20amount%20describe.png?raw=true)

<br>

Analysing grades:

```python
plt.figure(figsize=(12,4))
sns.countplot(x='grade',hue='loan_status',data=df)
```
![img grades](https://github.com/TSLSouth/Risk-Analysis-Prediction-with-Tensorflow/blob/main/img/grades.png?raw=true)

<br>

With subgrades we can see some correlations:
1) F and G looks doesn't get paid enough
2) Proportions of chagerd off loans increases from A1 to C4.

```python
plt.figure(figsize=(12,4))
sns.countplot(x='sub_grade',hue='loan_status',order=sorted(df['sub_grade'].unique()),data=df)
```

![img subgrades](https://github.com/TSLSouth/Risk-Analysis-Prediction-with-Tensorflow/blob/main/img/subgrades.png?raw=true)

<br>

Isolating subgrades F and G to look closely:

```python
plt.figure(figsize=(12,4))
f_g = df[(df['grade']=='F') | (df['grade']=='G')]
sns.countplot(x='sub_grade',hue='loan_status',order=sorted(f_g['sub_grade'].unique()),data=df)
```

![img F&G](https://github.com/TSLSouth/Risk-Analysis-Prediction-with-Tensorflow/blob/main/img/F%20nad%20G.png?raw=true)

<br>

Loan status vs. other features shows that with more income and more mortgage accounts results in more fully paid loans:

1. Transformming categorical loan_status in number:

```python
df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})

df[['loan_repaid','loan_status']]
```

![img repaid](https://github.com/TSLSouth/Risk-Analysis-Prediction-with-Tensorflow/blob/main/img/repaid.png?raw=true)

<br>

2. Looking at this correlations:

```python
df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
```

![img repaid corr](https://github.com/TSLSouth/Risk-Analysis-Prediction-with-Tensorflow/blob/main/img/repaid%20corr.png?raw=true)

<br>

---

<br>

## **4. Data Preparation** <a id='4'></a>

<br>

## _4.1 Removing or filling any missing data._

## _4.2 Removing unnecessary or repetitive features._

## _4.3 Converting categorical string features to dummy variables._

<br>
<br>

% of missing values in each column:

```python
df.isnull().sum()/len(df)*100

loan_amnt               0.000000
term                    0.000000
int_rate                0.000000
installment             0.000000
grade                   0.000000
sub_grade               0.000000
emp_title               5.789208
emp_length              4.621115
home_ownership          0.000000
annual_inc              0.000000
verification_status     0.000000
issue_d                 0.000000
loan_status             0.000000
purpose                 0.000000
title                   0.443148
dti                     0.000000
earliest_cr_line        0.000000
open_acc                0.000000
pub_rec                 0.000000
revol_bal               0.000000
revol_util              0.069692
total_acc               0.000000
initial_list_status     0.000000
application_type        0.000000
mort_acc                9.543469
pub_rec_bankruptcies    0.135091
address                 0.000000
loan_repaid             0.000000
```

<br> 

Dropping columns with missing data and very low correlation or categorical that we are not interested.

There are too many unique job titles to try to convert this to a dummy variable feature. So, we drop it

```python
df['emp_title'].nunique()
173105

df = df.drop('emp_title',axis=1)
```

<br>

Charge off rates are extremely similar across all employment lengths. We drop it:

```python
emp_co = df[df['loan_status']=='Charged Off'].groupby('emp_length').count()['loan_status']

emp_fp = df[df['loan_status']=='Fully Paid'].groupby('emp_length').count()['loan_status']

emp_len = emp_co/emp_fp

emp_len

emp_length
1 year       0.248649
10+ years    0.225770
2 years      0.239560
3 years      0.242593
4 years      0.238213
5 years      0.237911
6 years      0.233341
7 years      0.241887
8 years      0.249625
9 years      0.250735
< 1 year     0.260830

df = df.drop('emp_length',axis=1)
```
<br>

Similar columns: purpose and title. Droped:

```python
df['title'].head(10)

0                   Vacation
1         Debt consolidation
2    Credit card refinancing
3    Credit card refinancing
4      Credit Card Refinance
5         Debt consolidation
6           Home improvement
7       No More Credit Cards
8         Debt consolidation
9         Debt Consolidation


df['purpose'].head(10)

0              vacation
1    debt_consolidation
2           credit_card
3           credit_card
4           credit_card
5    debt_consolidation
6      home_improvement
7           credit_card
8    debt_consolidation
9    debt_consolidation

df = df.drop('title',axis=1)
```

Filling missing values with the mean value corresponding to its total_acc value:

```python
def fill_mort_acc(total_acc,mort_acc):
    '''
    Accepts the total_acc and mort_acc values.
    Checks if the mort_acc is NaN.
    If so, it returns the avg mort_acc value
    for the corresponding total_acc value for that row.
    
    '''
    if np.isnan(mort_acc):
        return perc_of_total[total_acc]
    else:
        return mort_acc

df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)

df.isnull().sum()

loan_amnt                 0
term                      0
int_rate                  0
installment               0
grade                     0
sub_grade                 0
home_ownership            0
annual_inc                0
verification_status       0
issue_d                   0
loan_status               0
purpose                   0
dti                       0
earliest_cr_line          0
open_acc                  0
pub_rec                   0
revol_bal                 0
revol_util              276
total_acc                 0
initial_list_status       0
application_type          0
mort_acc                  0
pub_rec_bankruptcies    535
address                   0
loan_repaid               0
```
<br>

Revol_util and the pub_rec_bankruptcies have missing data points. But they account for less than 0.5% of the total data. Let's drop the rows with missing values and complete missing data adjustment:

```python
df.isnull().sum()/len(df)*100

revol_util              0.069692
pub_rec_bankruptcies    0.135091

df = df.dropna()

loan_amnt               0
term                    0
int_rate                0
installment             0
grade                   0
sub_grade               0
home_ownership          0
annual_inc              0
verification_status     0
issue_d                 0
loan_status             0
purpose                 0
dti                     0
earliest_cr_line        0
open_acc                0
pub_rec                 0
revol_bal               0
revol_util              0
total_acc               0
initial_list_status     0
application_type        0
mort_acc                0
pub_rec_bankruptcies    0
address                 0
loan_repaid             0
```

<br>

Calling categorical columns using dtypes and let's go through all the categorical features to see what we should do with them.

```python
df.select_dtypes(['object']).columns

Index(['term', 'grade', 'sub_grade', 'home_ownership', 'verification_status',
       'issue_d', 'loan_status', 'purpose', 'earliest_cr_line',
       'initial_list_status', 'application_type', 'address'],
      dtype='object')
```
<br>

Grade feature is part of sub_grade, so let's drop it:

```python
df = df.drop('grade',axis=1)
```
<br>

Transforming subgrades to dummies:

```python
subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)

df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)

df.columns

Index(['loan_amnt', 'term', 'int_rate', 'installment', 'home_ownership',
       'annual_inc', 'verification_status', 'issue_d', 'loan_status',
       'purpose', 'dti', 'earliest_cr_line', 'open_acc', 'pub_rec',
       'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',
       'application_type', 'mort_acc', 'pub_rec_bankruptcies', 'address',
       'loan_repaid', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5',
       'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2',
       'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2', 'G3', 'G4',
       'G5'],
      dtype='object')
```

<br>

home_ownership can be converted to dummy variables, but will be interesting replace NONE and ANY withone word = OTHER. So we end up with just 4 categories and transform that to dummies:

```python
df['home_ownership'].value_counts()

MORTGAGE    198022
RENT        159395
OWN          37660
OTHER          110
NONE            29
ANY              3

df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

ownership_dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,ownership_dummies],axis=1)

df.select_dtypes(['object']).columns

Index(['verification_status', 'issue_d', 'loan_status', 'purpose',
       'earliest_cr_line', 'initial_list_status', 'application_type',
       'address'],
      dtype='object')
```

<br>

We can transform 'verification_status', 'application_type','purpose', 'initial_list_status' to dummies variables at once:

```python
dummies = pd.get_dummies(df[['verification_status', 'purpose', 'initial_list_status', 'application_type']],drop_first=True)

df = df.drop(['verification_status', 'purpose', 'initial_list_status', 'application_type'],axis=1)

df = pd.concat([df,dummies],axis=1)

df.columns

Index(['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'issue_d',
       'loan_status', 'dti', 'earliest_cr_line', 'open_acc', 'pub_rec',
       'revol_bal', 'revol_util', 'total_acc', 'mort_acc',
       'pub_rec_bankruptcies', 'address', 'loan_repaid', 'A2', 'A3', 'A4',
       'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5', 'D1',
       'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3',
       'F4', 'F5', 'G1', 'G2', 'G3', 'G4', 'G5', 'OTHER', 'OWN', 'RENT',
       'verification_status_Source Verified', 'verification_status_Verified',
       'purpose_credit_card', 'purpose_debt_consolidation',
       'purpose_educational', 'purpose_home_improvement', 'purpose_house',
       'purpose_major_purchase', 'purpose_medical', 'purpose_moving',
       'purpose_other', 'purpose_renewable_energy', 'purpose_small_business',
       'purpose_vacation', 'purpose_wedding', 'initial_list_status_w',
       'application_type_INDIVIDUAL', 'application_type_JOINT'],
      dtype='object')
```

<br> 

issue_d: we wouldn't know beforehand whether or not a loan would be issued when using our model, so, in theory, we wouldn't have an issue_date. Let's drop it:

```python
df = df.drop('issue_d',axis=1)
```

<br>

earliest_cr_year: this appears to be a historical time stamp feature. Let's extract the year from this and convert it to a numeric feature and set a new column to this:

```python
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))
df = df.drop('earliest_cr_line',axis=1)
```

<br>

adress: let's feature engineer a zip code column from the address. #We have just a few zip codes so we can transform in to dummies:

```python
df['zip_code'] = df['address'].apply(lambda address:address[-5:])

df['zip_code'].value_counts()

70466    56880
22690    56413
30723    56402
48052    55811
00813    45725
29597    45393
05113    45300
11650    11210
93700    11126
86630    10959

dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop(['zip_code','address'],axis=1)
df = pd.concat([df,dummies],axis=1)
```
<br>

Dropping laon status that is the same of loan_repaid:

```python
df = df.drop('loan_status',axis=1)
```

<br>

All the preprocessing is done!

<br>

---

<br>

## **5. Modeling** <a id='5'></a>

<br>

## _5.1 Train test split:_

```python
import sklearn

from sklearn.model_selection import train_test_split

X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```

<br>

## _5.2 Normalizing the data:_

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

```

<br>

## _5.3 Creating the model with Tensor flow:_

```python
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
```

<br>

We have 78 features, so we'll use this number in our first layer:

```python
X_train.shape

(276653, 78)
```

<br>

Model:


```python
model = Sequential()

# input layer
model.add(Dense(78, activation='relu')) # first layer: density = number of features

model.add(Dropout(0.2)) # to prevent overfitting in each layer

# hidden layer 1
model.add(Dense(156, activation='relu')) # second layer = density is half of the first
model.add(Dropout(0.2))

# hidden layer 2
model.add(Dense(78, activation='relu')) # third layer = dense is half of the second
model.add(Dropout(0.2))

# output layer for binary problem
model.add(Dense(units=1,activation='sigmoid'))

# compile model for binary classification
model.compile(loss='binary_crossentropy', optimizer='adam')
```

<br>

Fitting the model:

```python
model.fit(x=X_train, 
          y=y_train, 
          epochs=25,
          batch_size=256,
          validation_data=(X_test, y_test), 
          verbose=0,
         )
```

<br>

Saving the model:

```python
from tensorflow.keras.models import load_model

model.save('complete_data_project_model_one.h5') 
```

<br>

Classifying > 0.5:

```python
pred = (model.predict(X_test) > 0.5).astype('int32') # > 0.5).astype('int32')  to classify
```

<br>

---

<br>

## **6. Evaluation** <a id='6'></a>

<br>

Model loss vs. test loss:

```python
losses = pd.DataFrame(model.history.history)

losses[['loss','val_loss']].plot()
```

![img historical losses]()

<br>

Classification reports:
```python
print(classification_report(y_test,pred))

              precision    recall  f1-score   support

           0       0.97      0.44      0.61     23363
           1       0.88      1.00      0.93     95203

    accuracy                           0.89    118566
   macro avg       0.92      0.72      0.77    118566
weighted avg       0.90      0.89      0.87    118566

<------->

confusion_matrix(y_test,pred)

array([[10343, 13020],
       [  332, 94871]], dtype=int64)

<------->

df['loan_repaid'].value_counts()

1    317696
0     77523

<------->

317696/len(df)

0.8038479931379817
```

<br>

---

<br>

## **7. Return and adjust if necessary** <a id='7'></a>

<br>

First Run: 

- Our model is performming better than just a straight guess (89% from model vs. 80% straight guess) but it's no significant better.

- And our f1-score is still low at first attempt (0.61)

- So let's try to improve it playing with parameters: layers, density and dropout.

<br>

Second Run:

- After try many possibilities of layers and dropouts the result of f1-score on Charge Off classification was yet not so good. 

- Probably it's because we have an imbalanced dataset over our target feature. Let´s try to fix that and run again.

<br>

Third Run:

- After the third run with a more balanced data the model looks a bit better with a recall much better (0.80) but loss in precision. The F1-Score still remain 0.61 a the end:

```python
              precision    recall  f1-score 

           0       0.49      0.80      0.61    
           1       0.94      0.79      0.86     
```
<br>

As last model intent we'll compare with a Random Forest model:

```python
              precision    recall  f1-score  

           0       0.91      0.43      0.59     
           1       0.88      0.99      0.93     
```

<br>

---

<br>

## **8. Results** <a id='8'></a>

<br>

Our first attempt model with neural network and without oversampling the dataset is performming better than just a straight guess (89% from model vs. 80% straight guess) and that could be used on deployment. 

<br>

---

<br>

## **References/Acknowledgments**

<br>

References: 

<br>

https://www.kaggle.com/wordsforthewise/lending-club

<br>

Thanks to: 

<br>

- I2A2 - Institute of Applied Artificial Intelligence, Canada to have accepted me in their amazing courses.  <br>
- Pierian Data Training, USA to their great online courses.
- And to all friends that welcome me in the data science world and help me so much with their kindness, knowledge, experience and orientation.

<br>
