import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

# Step 1: Creating dataset
data =pd.read_csv(r'C:\Users\shrey\Downloads\dataset\train.csv')


# Step 2: Split features and labels
X = data.drop("smoking", axis=1)
y = data["smoking"]

# Step 3: Build a pipeline to handle categorical data + train model
model =  make_pipeline(
    OneHotEncoder(handle_unknown='ignore', sparse_output=False),
    RandomForestClassifier(n_estimators=200,random_state=42, n_jobs=-1)
)


model.fit(X, y)

# Step 4: Predicting new cases
test = pd.read_csv(r'C:\Users\shrey\Downloads\dataset\test.csv')
test=test.drop_duplicates()
test=test.dropna(subset=["id","age"])
data=data.drop_duplicates()
data=data.dropna(subset=["id","age"])
cols_to_numeric = ["height(cm)", "weight(kg)", "systolic", "HDL"]
for col in cols_to_numeric:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data["Urine protein"] = data["Urine protein"].astype("category")

def cap_outliers(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return series.clip(lower, upper)

numeric_cols = data.select_dtypes(include=np.number).columns
for col in numeric_cols:
    data[col] = cap_outliers(data[col])


smoking_prediction = model.predict_proba(test)

probablity=list(model.named_steps["randomforestclassifier"].classes_).index(1)

probablity_values=[]
for prob in smoking_prediction:
    value=prob[probablity]
    probablity_values.append(value)

submission=pd.DataFrame({"id":test["id"],"smoking":probablity_values})

print(submission)

submission.to_csv("submission.csv",index=False)


'''print(data['id'],smoking_prediction[0] )

print(test[["id","smoking_prediction"]])"'''
