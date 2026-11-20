# importing libraries for data preprocessing and model training
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('data/dataset.csv') # reading the dataset


#####  data exploration  #####
print(df.describe().round(2).T) # summary statistics


#####  data preprocessing  #####
# turns text column for employment status to numbers
df = pd.get_dummies(df, columns=['employment_status'], dtype=float)

X = df.drop(columns=['default'])
y = df['default']

# splitting data into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=32, stratify=y)

# scaling numerical features so large columns don't drown out small ones
num_cols = ['credit_score', 'annual_income', 'debt_to_income_ratio', 'loan_amount', 'loan_term_months', 'missed_payments']
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])


#### training ####
model = LogisticRegression(class_weight='balanced', max_iter=1000)  # logistic regression model built from sklearn
model.fit(X_train, y_train)


#### evaluation ####
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(f'Precision: {precision_score(y_test, y_pred):.2f}')
print(f'Recall: {recall_score(y_test, y_pred):.2f}')
print(f'F1-score: {f1_score(y_test, y_pred):.2f}')


#### new loan applications ####
new_loans = pd.DataFrame({
    'credit_score': [780, 640, 560],
    'annual_income': [95000, 52000, 28000],
    'employment_status': ['Employed', 'Self-Employed', 'Unemployed'],
    'debt_to_income_ratio': [0.15, 0.30, 0.55],
    'loan_amount': [20000, 25000, 15000],
    'loan_term_months': [36, 48, 60],
    'missed_payments': [0, 1, 3],
})

# same preprocessing as training data
new_loans = pd.get_dummies(new_loans, columns=['employment_status'], dtype=float)
new_loans = new_loans.reindex(columns=X.columns, fill_value=0.0)   # match training column order
new_loans[num_cols] = scaler.transform(new_loans[num_cols])

new_loans['default_probability'] = model.predict_proba(new_loans)[:, 1].round(2)
new_loans['decision'] = new_loans['default_probability'].apply(lambda p: 'DENY' if p >= 0.5 else 'APPROVE')
print(new_loans[['default_probability', 'decision']])
