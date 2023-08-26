#%% Libraries
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
#%% Read Data
raw_data = pd.read_csv(r"C:\Users\DELL\Desktop\Aygaz Bootcamp\insurance.csv")
#%% One-hot encoding
# Sex
le = LabelEncoder()
le.fit(raw_data.sex.drop_duplicates()) 
raw_data.sex = le.transform(raw_data.sex)
# Smoker
le.fit(raw_data.smoker.drop_duplicates()) 
raw_data.smoker = le.transform(raw_data.smoker)
# Region
le.fit(raw_data.region.drop_duplicates()) 
raw_data.region = le.transform(raw_data.region)
#%% Outlier Detection
describe = raw_data.describe()

outlier_list = []
outlier_col = []

for col in describe.columns:
    if col=="charges" or col=="children":
        continue
    mean = describe[col]["mean"]
    std = describe[col]["std"]
    
    min_level = mean-(3*std)
    max_level = mean+(3*std)
    
    for row in range(0,len(raw_data)):
        value = raw_data[col][row]
        if min_level <= value <= max_level:
            continue
        else:
            outlier_list.append(row)
            outlier_col.append(col)
raw_data = raw_data.drop(outlier_list).reset_index(drop=True)     
#%% Feature Selection
x_0 = raw_data[raw_data["sex"]==0].reset_index(drop=True)
x_1 = raw_data[raw_data["sex"]==1].reset_index(drop=True)

x_0_corr = x_0.corr()
x_1_corr = x_1.corr()

# Region anlamsız olduğu için attım.
x_0 = x_0.drop(['region'], axis = 1)

x_1 = x_1.drop(['region'], axis = 1)
x_1 = x_1.drop(['children'], axis = 1)

# Target datası
y_0 = x_0.charges
y_1 = x_1.charges        

x_0 = x_0.drop(['charges'], axis = 1)
x_1 = x_1.drop(['charges'], axis = 1)    
#%% Polynominal Regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

    
# Model oluşturma
poly = PolynomialFeatures(degree=3) # Best params degree=2
X_poly = poly.fit_transform(x_0)
pr = LinearRegression()

# Train, test ayrımı yapma
x_train,x_test,y_train,y_test = train_test_split(X_poly,y_0, random_state = 42, test_size=0.2)

# Model eğitme
pr.fit(x_train, y_train)

# Tahminleme
y_train_pred = pr.predict(x_train)
y_test_pred = pr.predict(x_test)

poly_rmse = round(mean_squared_error(y_test,y_test_pred)**0.5,3)
poly_r2 = round(r2_score(y_test,y_test_pred),3)

print("Polynominal regresyon RMSE:",poly_rmse )
print("Polynominal regresyon R2:",poly_r2 )
#%% Gradient boosting
# Best params n:90,depth:2, rate:0.18
from sklearn.ensemble import GradientBoostingRegressor

# Train, test ayrımı yapma
x_train,x_test,y_train,y_test = train_test_split(x_1,y_1, random_state = 42, test_size=0.2)

# Model oluşturma
gb_regressor = GradientBoostingRegressor(n_estimators=90, learning_rate=0.18, max_depth=2, random_state=42)

# Modeli eğitme
gb_regressor.fit(x_train, y_train)

# Tahminleme
y_train_pred = gb_regressor.predict(x_train)
y_test_pred = gb_regressor.predict(x_test)

gb_rmse = round(mean_squared_error(y_test,y_test_pred)**0.5,3)
gb_r2 = round(r2_score(y_test,y_test_pred),3)
print("Gradient boosting RMSE:",gb_rmse )
print("Gradient boosting R2:",gb_r2 )
#%%
print("Ortalama RMSE: ",round((gb_rmse+poly_rmse)/2,3))
print("Ortalama R2: ", round((poly_r2+gb_r2)/2,3))

