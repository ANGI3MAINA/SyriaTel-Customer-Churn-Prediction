# SyriaTel-Customer-Churn-Prediction
### By : Angela Wanjiru maina

## Overview
**Welcome to the GitHub repository for the SyriaTel Customer Churn Prediction project. This repository contains all the code, data, and documentation for developing a predictive model to address the high customer churn rate faced by SyriaTel. The project aims to identify customers at risk of churning, understand the underlying factors driving this behavior, and provide actionable insights to improve customer retention. Using a comprehensive dataset sourced from Kaggle, the repository includes steps for data preparation, model development, evaluation, and recommendations. The ultimate goal is to build a robust model with high accuracy and recall to help SyriaTel reduce churn and enhance profitability. Explore the repository to find detailed explanations, notebook, presentations and results that guide the entire process from data exploration to deploying the predictive model.

## 1. Business Understading

### Problem Overview

SyriaTel, a telecommunications company, is experiencing a significant customer churn problem, with many clients opting to leave and switch to competitors. To tackle this challenge, SyriaTel seeks to create a predictive model to anticipate customer churn. By doing so, the company hopes to better understand the key factors driving customer departures, improve retention rates, and ultimately boost profitability.

### Goals

1.Identify the primary factors leading to customer churn.

2.Build a model to predict which customers are most likely to leave.

3.Offer recommendations to proactively retain customers and reduce churn.


### Success Criteria

1.A reliable customer churn prediction model with high accuracy and a recall score of at least 0.75.

2.Identification of critical features that contribute to customer churn.

3.Practical recommendations for SyriaTel to lower churn rates and enhance customer loyalty.

## 2. Data Understanding
The dataset for this project is sourced from Kaggle (https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset) and was obtained directly from their repository.

This dataset is a single, comprehensive collection of data relevant to customer churn analysis. It includes various features that provide insights into customer behavior and interactions with the service provider. Its value lies in its thorough coverage of factors like service usage, billing details, and customer support engagements—all of which are critical for predicting churn.

By analyzing these features, we can develop models to identify customers at risk of leaving, allowing the company to take early action to retain them.

#### Categorical Features:

`state`: The state where the customer resides.

`phone number`: The phone number of the customer.

`international plan`: Whether the customer has an international plan (Yes or No).

`voice mail plan`: Whether the customer has a voice mail plan (Yes or No).

#### Numeric Features:

`area code`: The area code associated with the customer's phone number.

`account length`: The number of days the customer has been an account holder.

`number vmail messages`: The number of voice mail messages received by the customer.

`total day minutes`: The total number of minutes the customer used during the day.

`total day calls`: The total number of calls made by the customer during the day.

`total day charge`: The total charges incurred by the customer for daytime usage.

`total eve minutes`: The total number of minutes the customer used during the evening.

`total eve calls`: The total number of calls made by the customer during the evening.

`total eve charge`: The total charges incurred by the customer for evening usage.

`total night minutes`: The total number of minutes the customer used during the night.

`total night calls`: The total number of calls made by the customer during the night.

`total night charge`: The total charges incurred by the customer for nighttime usage.

`total intl minutes`: The total number of international minutes used by the customer.

`total intl calls`: The total number of international calls made by the customer.

`total intl charge`: The total charges incurred by the customer for international usage.

`customer service calls`: The number of customer service calls made by the customer.
