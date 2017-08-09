# -*- coding: utf-8 -*-
"""
Created on Wed Aug 09 10:34:09 2017

Script for data munging, load data, fill in missing values, format for analysis

@author: Richard
"""

import pandas as pd
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
libdir = os.path.join(parentdir,"ml_library")
sys.path.insert(0,libdir) 
from logistic_regression import logistic_regression


def data_munging():
    ''' perform initial data munging tasks'''
    train_data = pd.read_csv("../data/train.csv")
    
    # Use dummy variables for categorical variables, encode missing values 
    # we are not filling missing values, because the fact that they are missing
    # might contain some information
    train_dummy = pd.get_dummies(train_data, dummy_na = True, 
                   columns = ["Gender", "Married", "Dependents", "Education",
                              "Self_Employed", "Credit_History",
                              "Property_Area"])
    
    # look for missing values:
    np.sum(pd.isnull(train_dummy))
    
    # only missing LoanAmount for 22 rows / 614 rows ~ 3%.
    # fill with mean loan amount
    mean_loan = np.mean(train_dummy['LoanAmount'])
    train_dummy['LoanAmount'][pd.isnull(train_dummy['LoanAmount'])] = mean_loan
    
    # Set missing loan terms to 30 years:
    train_dummy['Loan_Amount_Term'][pd.isnull(train_dummy['Loan_Amount_Term'])] = 360
    
    # Now create additional variables as determined in R exploration.
    train = train_dummy # simplify notation
    train['LoanAmount'] = 1000*train['LoanAmount']
    train['MonthlyPayment'] = train['LoanAmount']/train['Loan_Amount_Term']
    train['TotalIncome'] = train['ApplicantIncome'] + train['CoapplicantIncome']
    train['PaymentRatio'] = train.MonthlyPayment / train.TotalIncome

    return train

# Now run logistic regression, calculate performance metrics
    
def calc_learning_curve(features, target, train_ratio):
    ''' calculate learning curve for provided training set '''
    m,n = features.shape
    num_train = int(m * train_ratio)
    train_idx = np.random.randint(0, m, num_train)
    train_features = features[train_idx,:]
    train_target = target[train_idx,:]
    
    cv_features = features[~train_idx,:]
    cv_target = target[~train_idx,:]
    
    # run loop with different sized training examples
    train_err = np.zeros(num_train)
    test_err = np.zeros(num_train)
    train_size = np.array(range(1,num_train+1))
    lr = logistic_regression()
    for i in range(num_train):
        lr.train(train_features[1:i,:], train_target[:,i,:])
    
    


# look at learning curve...
# add polynomial features.
