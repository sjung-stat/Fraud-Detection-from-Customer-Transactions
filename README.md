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
authorized directly by the card/account holder.” To prevent such
problems, many payment service companies are trying to develop methods
to detect suspicious transactions. And [Vesta
Corporation](https://trustvesta.com/) is one of those companies.

In this independent project, we deal with customer transactions dataset
provided by Vesta Corporation.

We briefly explore the dataset using visualization tools. And then we
clean the dataset so that we can make use of the dataset more
effectively. And lastly, we build two machine learning models, LightGBM
and CatBoost, to predict which transactions are fraud.

![](transactionfraud.png "Title") pc:
<https://www.elenavandesande.com/2016/10/24/how-to-detect-and-prevent-transaction-laundering/>

-----

### Installation

  - You can download the dataset used in this project
    [here](https://www.kaggle.com/c/ieee-fraud-detection/data).
  - Please note that we use the train\_identity and train\_transaction
    datasets only.
  - This project builds a LightGBM model so your computer needs to have
    the “lightgbm” library. Please refer to this [installation
    guide](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html)
    to install the library on your computer.

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
