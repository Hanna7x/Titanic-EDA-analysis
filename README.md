# Overview
Welcome to my exploratory data analysis (EDA) and survival analysis project on the Titanic dataset. This project aims to uncover patterns and trends in passenger survival, leveraging data cleaning, feature engineering, and visualization techniques. The project also showcases the use of Python and its data analysis libraries to answer key questions about the Titanic tragedy.
This analysis helps me strengthen my data analysis skills, particularly in cleaning data, creating meaningful visualizations, and interpreting results.
# The Questions 
Below are the questions I explored in this project:
1. What is the survival rate across different passenger classes (1st, 2nd, 3rd)?
2. How does gender influence survival rates?
3. How do ticket prefixes reflect socio-economic factors, and what impact do these factors have on survival rates?
4. Does Fare Distribution Vary Across Passenger Classes?
5. What Is the Relationship Between Fare and Survival?
6. hat Are the Survival Trends by Embarkation Port?
These questions allowed me to explore different aspects of the Titanic dataset, from socio-economic disparities to demographic influences on survival.
# Tools and Libraries
For this project, I used the following tools and libraries:
-** Python:
--**Pandas: Data manipulation and cleaning.
--** Matplotlib: Basic data visualizations.
--**Seaborn: Advanced visualizations for trend analysis.
-** Jupyter Notebook: For combining code, analysis, and visualization in a single environment.
# Data Preparation and Feature Engineering
## Data Cleaning
Handled missing values in the dataset:
Imputed missing Age values using the median of the Pclass and Sex group.
Replaced missing Embarked values with the most common port (S).
Dropped irrelevant columns (e.g., Cabin, due to a high percentage of missing values).
```python
# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
# Loading Data
train_data = pd.read_csv('/Users/xiongyihan/Desktop/python_data analyst/Raw_data/train.csv')
test_data = pd.read_csv('/Users/xiongyihan/Desktop/python_data analyst/Raw_data/test.csv')
test_data = pd.read_csv('/Users/xiongyihan/Desktop/python_data analyst/Raw_data/test.csv')
# Data Cleanup
train_data.head()
train_data.isnull().sum()
print(train_data['Embarked'].value_counts())
print(train_data['Cabin'].value_counts())
print(train_data[train_data['Embarked'].isnull()])
train_data['Embarked'] = train_data['Embarked'].fillna('S')
if 'Cabin' in train_data.columns:
    train_data = train_data.drop(columns=['Cabin'])
print(train_data.isnull().sum())
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
print(train_data.isnull().sum())
print(train_data['Age'].isnull().sum())
bins = [0, 12, 18, 60, 100]
labels = ['Children', 'Teenagers', 'Adults', 'Seniors']
train_data['AgeGroup'] = pd.cut(train_data['Age'], bins=bins, labels=labels)
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']
train_data['IsAlone'] = (train_data['FamilySize'] == 0).astype(int)
```

# The analysis and visualizations
Each question was explored with corresponding visualizations. Here’s how I approached each question:

## 1. What is the survival rate across different passenger classes (1st, 2nd, 3rd)?
class_survival = train_data.groupby('Pclass')['Survived'].mean()
print(class_survival)

```python
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(x=class_survival.index, y=class_survival.values)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()
```
### Results

![image](https://github.com/user-attachments/assets/dd9d24a0-3e3d-4a2e-9618-86e5723607d6)

1st class passengers had the highest survival rate, while 3rd class passengers had the lowest, reflecting socio-economic disparities.

## 2. How does gender influence survival rates?
```python
gender_survival = train_data.groupby('Sex')['Survived'].mean()
print(gender_survival)

# Visualization
sns.barplot(x=gender_survival.index, y=gender_survival.values)
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.show()
```
## Results

![image](https://github.com/user-attachments/assets/280d71a5-8ea1-44f4-9e44-d872972dbb03)


Females had a significantly higher survival rate than males, emphasizing the "women and children first" policy.

## 3. How do ticket prefixes reflect socio-economic factors, and what impact do these factors have on survival rates?
```python
# analyze Ticket Prefixes
# Extract ticket prefix
train_data['TicketPrefix'] = train_data['Ticket'].apply(lambda x: x.split()[0] if len(x.split()) > 1 else 'None')

# Group by TicketPrefix to calculate survival rates
ticket_survival = train_data.groupby('TicketPrefix')['Survived'].mean().reset_index()

# Filter prefixes with enough occurrences
ticket_counts = train_data['TicketPrefix'].value_counts()
valid_prefixes = ticket_counts[ticket_counts > 5].index
filtered_ticket_survival = ticket_survival[ticket_survival['TicketPrefix'].isin(valid_prefixes)]

# Visualization
sns.barplot(data=filtered_ticket_survival, x='TicketPrefix', y='Survived', order=filtered_ticket_survival.sort_values('Survived', ascending=False)['TicketPrefix'])
plt.title('Survival Rates by Ticket Prefix')
plt.xlabel('Ticket Prefix')
plt.ylabel('Survival Rate')
plt.xticks(rotation=45)
plt.show()
sns.barplot(data=filtered_ticket_survival, y='TicketPrefix', x='Survived', order=filtered_ticket_survival.sort_values('Survived', ascending=False)['TicketPrefix'])
plt.title('Survival Rates by Ticket Prefix')
plt.xlabel('Survival Rate')
plt.ylabel('Ticket Prefix')
plt.show()
```
## Results


![image](https://github.com/user-attachments/assets/2a5ad7fe-5ca5-44cd-827d-7af69120bb03)

![image](https://github.com/user-attachments/assets/0ae85c8d-1c08-4da3-8c77-0f4425a40fdf)

Certain ticket prefixes (e.g., PC, indicating wealthier passengers) had higher survival rates, while others (e.g., STONO) showed lower survival rates, reflecting socio-economic disparities among passengers.

## 4. Does fare distribution vary across passenger classes?
```python
sns.boxplot(x='Pclass', y='Fare', data=train_data)
plt.title('Fare Distribution by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.show()
```
## Results

![image](https://github.com/user-attachments/assets/57f90091-5a15-4105-8fcc-2341ed5c3204)


1st class passengers paid significantly higher fares, highlighting the wealth disparity between classes.

## 5. What is the relationship between fare and survival?
``` python
# What Is the Relationship Between Fare and Survival?
sns.scatterplot(x='Fare', y='Survived', data=train_data, hue='Pclass', palette='Set1')
plt.title('Relationship Between Fare and Survival')
plt.xlabel('Fare')
plt.ylabel('Survived (0 = No, 1 = Yes)')
plt.show()
```
## Results

![image](https://github.com/user-attachments/assets/aaa015e0-8056-4df4-af44-d8c776bda1f2)


Passengers who paid higher fares generally had better survival chances, reinforcing the connection between socio-economic status and survival.

## 6. What Are the Survival Trends by Embarkation Port?
```python
embarked_survival = train_data.groupby('Embarked')['Survived'].mean()
embarked_labels = embarked_survival.index
embarked_sizes = embarked_survival.values

plt.pie(embarked_sizes, labels=embarked_labels, autopct='%1.1f%%', startangle=140, colors=['#66b3ff','#99ff99','#ffcc99'])
plt.title('Survival Rates by Embarkation Port')
plt.show()
```

## Results

![image](https://github.com/user-attachments/assets/f2100667-e23b-431c-a9a7-882a4fd93152)


Passengers embarking from C had the highest survival rate, suggesting potential socio-economic differences by embarkation point.

# What I learned 
This project helped me develop and apply key skills, including:

- **Data Cleaning**: Addressing missing values and engineering meaningful features.
- **Visualization**: Using libraries like Matplotlib and Seaborn to create impactful charts.
- **Exploratory Data Analysis**: Identifying patterns and deriving insights from raw data.
- **Interpretation**: Drawing connections between data patterns and real-world implications.

# Challenges I Faced 
- **Handling Missing Data**: Addressing missing values in critical columns like Age and Embarked required careful imputation.
- **Feature Engineering**: Creating new features like FamilySize and TicketPrefix to uncover hidden patterns.
- **Complex Visualizations**: Designing visualizations that effectively communicated trends and patterns.

# Conclusion
This project provided a hands-on opportunity to explore the Titanic dataset, analyze survival patterns, and practice data analysis skills. The findings reveal how socio-economic factors, demographics, and travel conditions influenced survival during the Titanic tragedy. This project demonstrates my ability to clean, analyze, and visualize data effectively, making it a valuable addition to my portfolio.
