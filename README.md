# Data-BookingTrend
Project Overview

This project develops predictive machine learning models using booking data to identify key factors associated with customer cancellations. The objective is to support data-driven strategies that improve operational efficiency and enhance customer retention.
The dataset was provided by Hotel Haven, a luxury hotel chain with multiple locations. Hotel Haven has faced challenges in accurately predicting booking cancellations, which has negatively impacted resource planning and operational management. By analyzing historical booking patterns, this project aims to improve cancellation forecasting and inform proactive retention strategies.
Methodology

The analysis followed an end-to-end data science workflow:

Data Cleaning and Exploration:
Data quality checks, preprocessing, and exploratory data analysis (EDA) were conducted to understand booking patterns, customer behavior, and key drivers of cancellations.

Feature Engineering:
Relevant features were engineered to capture booking characteristics and customer attributes that influence cancellation behavior.

Data Preprocessing:
Numerical features were scaled, and class imbalance was addressed using SMOTE (Synthetic Minority Oversampling Technique) to improve model learning and predictive performance.

Model Development:
A baseline Logistic Regression model was trained using an 80/20 train–test split. The baseline model achieved an accuracy and F1-score of 77%.
Advanced Modeling:
More complex models were evaluated, including Random Forest and multi-model approaches. The Random Forest model outperformed other models, achieving an accuracy and F1-score of 89%.

Model Optimization:
Hyperparameter tuning was applied to the Random Forest model, resulting in an Area Under the ROC Curve (AUC) of 96%, indicating strong predictive performance and robust discrimination between canceled and non-canceled bookings.
Results and Impact

The optimized Random Forest model demonstrates a high capacity to predict customer cancellations with strong accuracy and reliability. With effective deployment, this model can help Hotel Haven:

Anticipate cancellations more accurately

Improve resource allocation and operational planning

Implement targeted retention strategies

Reduce revenue loss associated with last-minute cancellations
