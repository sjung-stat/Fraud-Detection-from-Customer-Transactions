Fraud Detection from Customer Transaction
================
Seyoung Jung
08/03/2020

-----

# 1 Introduction

Today, due to the development of digital payments, less people are
carrying cash. According to this news article titled [More Americans say
they don’t carry
cash](https://www.cnbc.com/2019/01/15/more-americans-say-they-dont-carry-cash.html),
“in an average week, roughly 3 in 10 adults said they make zero
purchases using cash”. Also it mentions that “millennials are paving the
way among people ditching bills and coins in favor or credit, debit and
digital payments, through apps like Apple Pay, Venmo, and Zelle.”
Digital payments are very convenient; you don’t have to carry your fat
wallet anymore, and most importantly, it allows you to shop online.
Especially during this ongoing global pandemic of COVID-19, the online
consumer market is growing exponentially to avoid physical contact with
others.

However, this digital payment system has a serious problem; exposure to
fraud transactions. And this
[article](https://www.paymentssource.com/opinion/coronavirus-increases-exposure-for-digital-payments-fraud)
suggests that “coronavirus increases exposure for digital payments
fraud.” Such fraud tranasctions So, many experts are trying to develop
methods to decrease the number of fraud transactions.

In this project, we build models that predict whether a transaction is a
fraud. The dataset used for this project can be found
[here](https://www.kaggle.com/c/ieee-fraud-detection/data). In this
project, we use only training set to see how well our models classify
fraud transactions.

-----

# 2 Preparation

We will load packages we need for this project.

``` r
library(readr)        # read_csv
library(dplyr)        # glimpse
library(stringr)      # str_c, str_sub
library(ggplot2)      # ggplot
library(scales)       # label = comma
library(gridExtra)    # grid.arrange
library(tidyr)
library(janitor)      # remove_constant
library(magrittr)     # %<>% pipeline
library(forcats)      # fct_lump
library(lightgbm)
library(Matrix)       #s parse.model.matrix
```

And read dataset into R. Before we join the two dataset, we will create
a new column called “isID” to identify which rows are from id\_data.

``` r
txn_data <- read_csv("fraud_detection_transaction.csv", n_max=600000)
id_data <- read_csv("fraud_detection_identity.csv", n_max=150000)
txn_data$isID <- NA
id_data$isID <- 1
```
