from fastapi import FastAPI
from payload_models.Payload import Body
import pickle
import pandas as pd
import numpy as np

app = FastAPI()
model = pickle.load(
    open("/Users/pavanmantha/Pavans/Workshops/GPR_Workshop/RegressionAnalysis/KNeighborsClassifier.pkl", "rb"))


@app.post("/predict")
def health(payload: Body):
    df_2 = pd.DataFrame(
        columns=['Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'Education', 'EducationField',
                 'EmployeeCount', 'Gender', 'JobLevel', 'JobRole', 'MaritalStatus', 'MonthlyIncome',
                 'NumCompaniesWorked',
                 'Over18', 'PercentSalaryHike', 'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
                 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager'],
        # data=np.array([[28,1,1,11,2,3,1,1,2,7,2,58130,2.0,0,20,8,1,5.0,2,0,0,0]])
        data=np.array([payload.data]))
    res = model.predict(df_2)
    print(res)
    if res[0] == 0:
        return {'will_leave_company': 'No'}
    else:
        return {'will_leave_company': 'Yes'}
