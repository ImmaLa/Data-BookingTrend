import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing data 
df = pd.read_csv("bookingg.csv")

# Data preprocessing
## Viewing Data Features 

df.head()
df.tail()

# Data Cleaning, checking for missing values and duplications

df.info()
df.isnull().sum()

# Droping irrelevant colunms
df = df.drop(columns=["Booking_ID"])

# Removing space from colunm names
df.columns = df.columns.str.replace('number of adults','Number_of_adults')
df.columns = df.columns.str.replace('number of children','Number_of_children')
df.columns = df.columns.str.replace('number of weekend nights','Number_of_weekend_nights')
df.columns = df.columns.str.replace('number of week nights','Number_of_week_nights')
df.columns = df.columns.str.replace('type of meal','Type_of_meal')
df.columns = df.columns.str.replace('car parking space','Car_parking_space')
df.columns = df.columns.str.replace('room type','Room_Type')
df.columns = df.columns.str.replace('lead time','Lead_time')
df.columns = df.columns.str.replace('market segment type','Market_segment_type')
df.columns = df.columns.str.replace('repeated','Repeated')
df.columns = df.columns.str.replace('P-C','PC')
df.columns = df.columns.str.replace('P-not-C ','PnotC ')
df.columns = df.columns.str.replace('average price','Average_price')
df.columns = df.columns.str.replace('special requests','Special_requests')
df.columns = df.columns.str.replace('booking status','Booking_status')
df.columns = df.columns.str.replace('date of reservation','date_of_reservation')


# Descriptive Statistics for numerical data
df.describe()

# Statistical Summary for the  categorical data
cat_data = data.select_dtypes(include=["object", "category", "bool"])
cat_data.describe()

# Plotting the outliers
numerical_col = ["lead time", "average price"]
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
ax = ax.flatten()
for idx, column_name in enumerate(numerical_col):
    sns.histplot(data[column_name], bins=30, kde=True, color="teal", ax=ax[idx])
    ax[idx].set_title(f"Histogram for {column_name}")

plt.tight_layout()
plt.show()
sns.boxplot(y=data['lead time'])
plt.show()

sns.boxplot(y=data['average price'])
plt.show()

# Handling  outliers¶
affected_column=["lead time", "average price"]
q1 = data[affected_column].quantile(0.25)
q3 = data[affected_column].quantile(0.75)
Iqr = q3 - q1

Lower_bound = q1-(1.5 * Iqr)
Upper_bound = q1 + (1.5 * Iqr)

Lower_bound
Upper_bound

df[affected_column] = df[affected_column].clip(lower =Lower_bound, upper = Upper_bound, axis = 1)

### Verifying whether outlier is gone
sns.boxplot(y=df['Lead_time'])
plt.show()

sns.boxplot(y=df['Average_price'])
plt.show()

# Statistical analysis

# Style settings
sns.set(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.figsize"] = (10, 5)

# Target variable
target = "Booking_status"
# CANCELLATION RATE (in percentage)
cancel_percent = (
    data[target].value_counts(normalize=True) * 100
).reset_index()

cancel_percent.columns = [target, "Percentage"]

# Plotting the cancellation rate using Seaborn
plt.figure(figsize=(6, 4))
sns.barplot(
    data=cancel_percent,
    x=target,
    y="Percentage",
    hue=target,            
    palette="viridis",
    dodge=False,          
    legend=False          
)
plt.title("Percentage of Cancellations vs Non-Cancellations")
plt.ylabel("Percentage (%)")
plt.xlabel("Booking_Status")

plt.legend(title="Booking_Status", labels=["Canceled", "Not_Canceled"])

plt.show()

# Display cancellation percentage table
print("\n=== Cancellation Percentage Table ===")
print(cancel_percent)

categorical_cols = [
    'Type_of_meal',
    'Car_parking_space',
    'Room_Type',
    'Market_segment_type',
    'Repeated'
]
for col in categorical_cols:
    plt.figure(figsize=(10, 5))
    sns.countplot(
        data=data,
        x=col,
        hue=target,
        palette="viridis",
        dodge=True 
    )
    plt.title(f"Bar Chart of {col} by Cancellation Status")
    plt.xticks(rotation=45)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.legend(title="Booking Status", labels=["Not Canceled", "Canceled"])
    plt.show()

for col in columns_to_plot:   
    plt.figure(figsize=(8, 5))      
    ax = sns.barplot(
        data=data,
        x=target,
        y=col,
        hue=target,
        palette="viridis",
        width=0.6
    )
    plt.title(f"Average {col} by Booking_Status")
    plt.xlabel("Booking_Status")
    plt.ylabel(f"Average {col} (Days)")
    plt.show()

# Model development 
## Scale numerical Features
## Scale numerical columns to bring them to a similar range to improve model performance
Col_to_scale = ["Lead_time", "Average_price"]
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

df[Col_to_scale] = scaler.fit_transform(df[Col_to_scale])

df.head()
#Checking class distribution to handle class imbalanced
df["Booking_status"].value_counts()
df["Booking_status"].value_counts(normalize=True)*100
sns.countplot(x=df["Booking_status"])

#Target varable is inbalnace
#Handling imbalance¶
#Oversampling with smooth
!pip install imbalanced-learn

from imblearn.over_sampling import SMOTE #Synthetic Minority Oversampling Technique 
## Separate data from the feature
x = df.drop("Booking_status", axis=1) 
y = df["Booking_status"]
y.head
x.head()
smote = SMOTE(random_state=42)
x_resampled, y_resampled = smote.fit_resample(x, y)
y_resampled.value_counts()
y_resampled.value_counts(normalize=True)*1
sns.countplot(x=y_resampled)
resampled_data = pd.DataFrame(x_resampled, columns=x.columns)
resampled_data["Booking_status"] = y_resampled
resampled_data.head()


# Saving the dataset
df.to_csv("clean_customer_booking_status_balanced.csv", index=False)
resampled_data.head()

# Data Splitting
from sklearn.model_selection import train_test_split

### Split the data 80% for training and 20% for testing
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size = 0.2, random_state=42)
print(f"training data size{len(x_train)}")
print(f"testing data size{len(x_test)}")

y_train.value_counts(normalize=True)*100
y_test.value_counts(normalize=True)*100


# Model training

## Training using Logistics Regression(Baseline model)
from sklearn.linear_model import LogisticRegression

###model initialiation
model = LogisticRegression(random_state =42)
## model traing
model.fit(x_train, y_train)

### Predicting on the test set
y_pred = model.predict(x_test
y_pred[:5]
print(y_test[:5])
                       
from sklearn.metrics import classification_report, confusion_matrix
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Logistic Regression Clasiffication Report")
print(report)
y_test.value_counts()
len(y_test)

# Training with random forest(Advanced model)
from sklearn.ensemble import RandomForestClassifier

###model initialiation
RandomForest_model = RandomForestClassifier(random_state =42)
## model traing
RandomForest_model.fit(x_train, y_train)

### Predicting on the test set
y_pred = RandomForest_model.predict(x_test)
report = classification_report(y_test, y_pred)
print("Logistic Regression Classification Report")
print(report)

### retraining the model with specification, class weigt = balanced
###model initialiation
RandomForest_model = RandomForestClassifier(random_state =42, class_weight = "balanced")

## model traing
RandomForest_model.fit(x_train, y_train)

### Predicting on the test set
y_pred = RandomForest_model.predict(x_test)
report = classification_report(y_test, y_pred)
print("Logistic Regression Classification Report")
print(report)

# Training with multiple classification model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

!pip install xgBoost
from xgboost import XGBClassifier

models = {"Logistic Regression": LogisticRegression(random_state =42, class_weight = "balanced"),
        "Random Forest": RandomForestClassifier(random_state =42, class_weight = "balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state =42),
        "Ada Boost": AdaBoostClassifier(random_state =42),
        "Decision Tree": DecisionTreeClassifier(random_state =42, class_weight = "balanced"),
        "KNeighbors": KNeighborsClassifier(),
        "SVC":SVC(random_state =42, class_weight = "balanced"),
        "XGBClassifier": XGBClassifier(eval_metric="mlogloss")
        }
result = {}
from sklearn.metrics import accuracy_score
result={}

for model_name, model in models.items():
    print(f"Training {model_name}....")
    ## model traing
    model.fit(x_train, y_train)
    
    ### Predicting on the test set
    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred)
    
    # Evalaute the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    ## store the result
    result[model_name] = {
        "accuracy": accuracy, 
        "classification_report": report,
        "confusion_matrix": cm
        }
    print(F"{model_name} clasification report")
    print(report)
    sns.heatmap(cm, annot=True, fmt="g")
    plt.show()

## I would consider the random forest model which has 89% accuracy

## Hyperparameter tuning for random forest model
RandomForestClassifier(random_state =42, class_weight = "balanced")
from sklearn.model_selection import GridSearchCV

## Definde the parameter grid
parameter_grid = {
    "n_estimators" : [50, 100, 200],
    "max_depth" : [None, 10, 20,30],
    "min_samples_split" : [2, 5, 10],
    "min_samples_leaf" : [1, 2, 4]
}

grid_search=GridSearchCV(RandomForestClassifier(random_state =42, class_weight = "balanced"),
                         param_grid = parameter_grid,
                         scoring="f1",
                         cv=5,
                         n_jobs=-1
                        )
grid_search.fit(x_train, y_train)

## Best parameter
print(F"Best Hyperparameters: {grid_search.best_params_}")

## Train best model
best_RandomForest = grid_search.best_estimator_

y_pred_best_randomforest = best_RandomForest.predict(x_test)

print("Fine tune radom forest classifier")
report = classification_report(y_test, y_pred_best_randomforest)
print(report)

## Model evalution
Matrix = confusion_matrix(y_test, y_pred_best_randomforest)

sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predictions")
plt.ylabel("Actual")
plt.show()

# ROC curve and AUC score to evaluate the models ability to distingush between classes
from sklearn.metrics import roc_curve, roc_auc_score

##compute the posibilities for roc
y_probs = best_randomforest.predict_proba(x_test)[:, 1]

# ROC curve
fpr, tpr, threshold =roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC: {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--',color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()
# AUC is 96%, therefore the rate of prediction of this model is very good.

