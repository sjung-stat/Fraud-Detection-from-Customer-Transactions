Fraud Detection from Customer Transactions
================

> Investigating customer transactions dataset and building models to
> detect fraud transactions

### Introduction

The emergence of digital payments has made our lives a lot easier. For
example, it helps us lighten our wallets because we can carry only a
piece of plastic card. Also, it allows us to shop online at home. In
addition to that, digital payments have lots of advantages. But at the
same time, a serious problem has arisen: fraudulent transactions.
According to [Axis
Bank](https://application.axisbank.co.in/webforms/axis-support/sub-issues/FND-Fraud-ccdcsa-1.aspx),
a fraudulent transaction is “any transaction in the account that was not
authorized directly by the card/account holder.”

![](transactionfraud.png "Title")

pc:<https://www.elenavandesande.com/2016/10/24/how-to-detect-and-prevent-transaction-laundering/>

Individuals should make an effort to keep their personal information
private and avoid fraud and abuse of personal accounts. But, such
fraudulent activities should be prevented at a company level as well. To
prevent such problems, many payment service companies are trying to
develop methods to detect suspicious transactions. And [Vesta
Corporation](https://trustvesta.com/) is one of those companies.

According to [this
article](https://www.nerdwallet.com/article/credit-cards/merchants-victims-credit-card-fraud),
“when merchants are victims of credit card fraud”, the one that will pay
is “rarely the consumer. Instead, liability usually comes down to the
merchant or the bank that issued the card.” However, it also states that
“U.S. credit card fraud over time raises retail prices for consumers, as
businesses pass along the cost of fraud.” So the consumer will be
responsible for it in the long term. Therefore, this fraudulent
transactions harm the whole economic system.

-----

### Objective

I recently tried to purchase something online with my credit card. The
merchandise was something that I had not bought before and the price was
unusually expensive. Also, I was very far away from my home when I was
placing an order online with the credit card. As a result, the
transaction was denied and the bank sent me a notification to verify if
the attempt was made by me. I was impressed that they could catch this
unusual transaction in advance and deny them. And I also wanted to see
if my own machine learning model could perform well.

In this independent project, we deal with customer transactions dataset
provided by Vesta Corporation to predict which transactions are fraud.
To accomplish this, we will build a machine learning model, CatBoost.
You can find more information about CatBoost
[here](https://catboost.ai/). You can find the comparison of CatBoost to
the other two famous boosted-based algorithms, XGBoost and LightGBM,
[here](https://medium.com/riskified-technology/xgboost-lightgbm-or-catboost-which-boosting-algorithm-should-i-use-e7fda7bb36bc).

#### About the Data

You can download the dataset used in this project
[here](https://www.kaggle.com/c/ieee-fraud-detection/data). Please note
that we use the train\_identity and train\_transaction datasets only. 
- The training set is comprised of 590k of transaction information and 435
features such as the transaction amount, product code, and address. 
- The target variable is “isFraud”. 
- The Descriptions of the features can be found [here](https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203).

But please note that detailed description is not available at this moment.

#### Why use CatBoost for this project?

First, CatBoost can handle missing values internally. Majority of
features from the dataset have missing values. And you will see that the
proportion of missing values of 13 of them are higher than 90%. Since we
do not have to take care of missing values manually, it saves a lot of
trouble. Also, the training time of the algorithm is relatively faster
than other classification algorithms. Since we have approximately 590k
observations, a faster computation is a great advantage. And CatBoost is
very useful when variables are categorical, because it has its own
method to deal with categorical features. Since we already have 400+
features, it would be computationally very heavy to use One Hot
Encoding. So this makes the training a lot easier.

-----

### Conclusion

![](CatBoost%20result.png "Title")

I used the “Log Loss” metric for this classification modeling and I
achieved logloss=0.05298 with 10,000 iterations. The dataset is very
imbalanced. Given this proportion, the approximate log loss of a baseline
model is 0.152 with 96.5%/3.5% split for target variable. So the result from validation set is great. However, the computation was
still too heavy for my machine; it took 10+ hours to train the data.

![](feature%20importance.png "Title")

Also, according to this
[document](https://catboost.ai/docs/concepts/fstr.html#fstr), the
feature importance “shows how much on average the prediction changes if
the feature value changes. The bigger the value of the importance the
bigger on average is the change to the prediction value, if this feature
is changed.” The feature importance part gives us an interesting result;
TransactionDT (transaction timedelta from a given reference datetime),
TransactionAMT (transaction amount), card1 (unknown), and card2
(unknown) are the most important features. So, even though the detailed
information regarding card1 and card2 are unknown, they play an
essential role when detecting fraudulent transactions.

Please note that due to the general performance of my computer, I am
currently not able to perform the parameter tuning. Once I upgrade my
machine, I will try to optimize the model by tuning hyperparameters and
then improve its performance again.

-----

### Installation

This project builds a CatBoost model so your computer needs to have the
“catboost” library. Please refer to this [installation
guide](https://catboost.ai/docs/concepts/r-installation.html) to install
the library on your computer.

-----

### Versions

For this project, I use

  - R version 4.0.0
  - cmake version 3.18.1
  - Visual Studio 2019 version 16.6
  - Git version 2.28.0

-----

### Contact Information

  - If you have any questions, feel free to email me at
    <sjung.stat@gmail.com>
  - You can find my LinkedIn profile
    [here](https://www.linkedin.com/in/sjung-stat/)
