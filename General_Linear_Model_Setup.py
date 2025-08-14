import pandas as pd
import numpy as np
#Regression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Load dataset (assuming it is a CSV file or DataFrame already loaded)
ORR5 = r'C:\Users\CodyConner\OneDrive - capitalizeconsulting.com\Documents\Projects\OGE\ORR5.xlsx'
#df = pd.read_csv(ORR5)  # Replace with actual data source
df = pd.read_excel(ORR5)

# Create additional prediction data
data9999 = r'C:\Users\CodyConner\OneDrive - capitalizeconsulting.com\Documents\Projects\OGE\9999.xlsx'
#df_additional = pd.read_csv(data9999)  # Replace with actual additional data source
df_additional = pd.read_excel(data9999)
df_additional['__FLAG'] = 0
df['__FLAG'] = 1
df_combined = pd.concat([df, df_additional], ignore_index=True)
df_combined['__DEP'] = df_combined['AvgCustKwh']
df_combined.loc[df_combined['__FLAG'] == 0, 'AvgCustKwh'] = np.nan

# Defining independent variables (features) and dependent variable (target)
categorical_features = ['Month', 'CovidFlags', 'HourEndingCST', 'DayOfWeek', 'OtherA', 'HDAY']
numerical_features = ['HighTempAfternoon', 'HighSquared', 'HighCubed', 'SpecialTemp', 'SpecTSqd', 'SpecTCub']
interaction_terms = [
    'HourEndingCST*DayOfWeek', 'HighTempAfternoon*Month', 'HighSquared*Month', 'HighCubed*Month',
    'HighTempAfternoon*HourEndingCST', 'HighSquared*HourEndingCST', 'HighCubed*HourEndingCST',
    'SpecialTemp*Month', 'SpecTSqd*Month', 'SpecTCub*Month', 'SpecialTemp*HourEndingCST',
    'SpecTSqd*HourEndingCST', 'SpecTCub*HourEndingCST'
]

df_combined['Interaction'] = df_combined[numerical_features].prod(axis=1)

# Encode categorical variables
X = pd.get_dummies(df_combined[categorical_features], drop_first=True)
X = pd.concat([X, df_combined[numerical_features + ['Interaction']]], axis=1)

# Convert all columns to numeric (handle any remaining object types)
X = X.apply(pd.to_numeric, errors='coerce')

df_combined['AvgCustKwh'] = pd.to_numeric(df_combined['AvgCustKwh'], errors='coerce')

# Drop rows with missing values in target
df_combined = df_combined.dropna(subset=['AvgCustKwh'])

# Adding a constant term for intercept
X = sm.add_constant(X)
y = df_combined['AvgCustKwh']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the linear model
model = sm.OLS(y_train, X_train).fit()

# Generate predictions
df_combined['predicted_AvgCustKwh'] = model.predict(X)

# Restore original AvgCustKwh values
df_combined['AvgCustKwh'] = df_combined['__DEP']
df_combined.drop(columns=['__DEP'], inplace=True)

# Display summary and predictions
print(model.summary())
print(df_combined[['predicted_AvgCustKwh', 'AvgCustKwh']].head())

#----------------------------------------------------------------------------
import pandas as pd
import numpy as np
#Regression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

#Load Data
data = Alteryx.read("History")
nine = Alteryx.read("9999")
Dep = Alteryx.read("Dependent")
Quant = Alteryx.read("Quantitative")
Class = Alteryx.read("Classification")
Int = Alteryx.read("Interactions")
df = pd.DataFrame(data)
df_additional = pd.DataFrame(nine)
Depdf = pd.DataFrame(Dep)
Quantdf = pd.DataFrame(Quant)
Classdf = pd.DataFrame(Class)
Intdf = pd.DataFrame(Int)

#print(df_additional)
#print(df)
#print(Depdf)
#print(Quantdf)
#print(Classdf)
print(Intdf)

Dependent = Depdf.at[0,'DependentVariable']
print(Dependent)

qv = Quantdf.at[0,'QuantVariable']
qvar = qv.split(',')
print(qvar)

Classv = Classdf.at[0,'ClassVariable']
classvar = Classv.split(',')
print(classvar)

Intv = Intdf.at[0,'Effects']
Intvar = Intv.split(',')
print(Intvar)

Quantitative = qvar
Classification = classvar
Interactions = Intvar

#print(Dependent)
#print(Quantitative)
#print(Classification)
#print(Interactions)

df_additional['__FLAG'] = 0
df['__FLAG'] = 1
#if second dataframe is empty this section will fail
df_combined = pd.concat([df, df_additional], ignore_index=True)
df_combined['__DEP'] = df_combined['AvgCustKwh']
df_combined.loc[df_combined['__FLAG'] == 0, 'AvgCustKwh'] = np.nan

#print(df.head())
#print(df_additional.head())
#print(df_combined.head())
#print(data.head())

# Defining independent variables (features) and dependent variable (target)
categorical_features = ['Month', 'CovidFlags', 'HourEndingCST', 'DayOfWeek', 'OtherA', 'HDAY']
numerical_features = ['HighTempAfternoon', 'HighSquared', 'HighCubed', 'SpecialTemp', 'SpecTSqd', 'SpecTCub']
#interaction_terms = [Interactions]
interaction_terms = [
    'HourEndingCST*DayOfWeek', 'HighTempAfternoon*Month', 'HighSquared*Month', 'HighCubed*Month',
    'HighTempAfternoon*HourEndingCST', 'HighSquared*HourEndingCST', 'HighCubed*HourEndingCST',
    'SpecialTemp*Month', 'SpecTSqd*Month', 'SpecTCub*Month', 'SpecialTemp*HourEndingCST',
    'SpecTSqd*HourEndingCST', 'SpecTCub*HourEndingCST'
]

# Ensure categorical encoding
X = pd.get_dummies(df_combined[categorical_features], drop_first=True)

#print('X Dummy')
#print(X.dtypes)
#print(X.head(10))

# Include numeric columns
X = pd.concat([X, df_combined[numerical_features]], axis=1)

#print('X Concat')
#print(X.dtypes)
#print(X.head(10))

# Convert all columns to numeric to prevent dtype issues
X = X.apply(pd.to_numeric, errors='coerce')

#print('X numeric')
#print(X.dtypes)
#print(X.head(10))

# Drop any remaining NaN values
X.dropna(inplace=True)
y = df_combined.loc[X.index, 'AvgCustKwh']  # Ensure y aligns with X

# Add a constant for OLS
X = sm.add_constant(X)

# Split data again, ensuring no misalignment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print('X_train')
#print(X_train.dtypes)
#print(X)
#print('y_train')
#print(y_train.dtypes)

# Fit the model
model = sm.OLS(y_train, X_train).fit()

print(model)

# Generate predictions
df_combined['predicted_AvgCustKwh'] = model.predict(X)

#output1 = dfcombined
Alteryx.write(df_combined, 1)

