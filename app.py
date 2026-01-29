import gradio as gr
import pandas as pd
import pickle

# Load the Model
with open("employee_lr_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

categorical_features_value = {
    'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
    'Department': ['Sales', 'Research & Development', 'Human Resources'],
    'EducationField': ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'],
    'Gender': ['Female', 'Male'],
    'JobRole': ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
                'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'],
    'MaritalStatus': ['Single', 'Married', 'Divorced'],
    'Over18': ['Y'],
    'OverTime': ['Yes', 'No']
    }

default_values = {
    'Age': 30,
    'MonthlyIncome': 5000,
    'YearsAtCompany': 5,
    'JobRole': 'Sales Executive',
    'MaritalStatus': 'Single',
    'DistanceFromHome': 7,
    'WorkLifeBalance': 3,            
    'RelationshipSatisfaction': 3,   
    'EnvironmentSatisfaction': 3,    
    'StockOptionLevel': 1,
    'YearsInCurrentRole': 3,
    'OverTime': 'No',
    'YearsWithCurrManager': 3,
    'Department': 'Research & Development',
    'YearsSinceLastPromotion': 1,
    'JobSatisfaction': 3,             
    'PercentSalaryHike': 14,          
    'TrainingTimesLastYear': 3,
    'DailyRate': 800,
    'NumCompaniesWorked': 2,
    'Education': 3,
    'JobLevel': 2,
    'Gender': 'Male',
    'HourlyRate': 50,
    'BusinessTravel': 'Non-Travel',
    'JobInvolvement': 3,
    'PerformanceRating': 3,
    'TotalWorkingYears': 5,
    'EducationField': 'Technical Degree',
    'MonthlyRate': 15000
}

# top 11 important features (from top 20 feature importance)
top_11_features_list = [
    'JobRole',
    'BusinessTravel',
    'YearsAtCompany',
    'OverTime',
    'MaritalStatus',
    'YearsInCurrentRole',
    'EducationField',
    'Department',
    'TotalWorkingYears',
    'YearsWithCurrManager',
    'NumCompaniesWorked',
]

# prediction
def predict_attrition(
    JobRole,
    BusinessTravel,
    YearsAtCompany,
    OverTime,
    MaritalStatus,
    YearsInCurrentRole,
    EducationField,
    Department,
    TotalWorkingYears,
    YearsWithCurrManager,
    NumCompaniesWorked
):
    data = default_values.copy()
    
    # update data from input value
    data.update({
        'JobRole': str(JobRole),
        'BusinessTravel': str(BusinessTravel),
        'YearsAtCompany': float(YearsAtCompany),
        'OverTime': 1 if OverTime in [1, 'Yes'] else 0,
        'MaritalStatus': str(MaritalStatus),
        'YearsInCurrentRole': float(YearsInCurrentRole),
        'EducationField': str(EducationField),
        'Department': str(Department),
        'TotalWorkingYears': float(TotalWorkingYears),
        'YearsWithCurrManager': float(YearsWithCurrManager),
        'NumCompaniesWorked': float(NumCompaniesWorked),
    })
    
    df = pd.DataFrame([data])

    pred = loaded_model.predict(df)[0]
    
    return "Yes, Employee is likely to leave" if pred == 1 else "No, Employee is likely to stay"


# gradio inputs
inputs = [
    gr.Dropdown(label='Job Role', choices=categorical_features_value['JobRole'], value=default_values['JobRole']),
    gr.Radio(label='Business Travel', choices=categorical_features_value['BusinessTravel'], value=default_values['BusinessTravel']),
    gr.Slider(label='Years at Company', value=default_values['YearsAtCompany'], minimum=0, maximum=40),
    gr.Dropdown(label='OverTime', choices=['Yes','No'], value=default_values['OverTime']),
    gr.Radio(label='Marital Status', choices=categorical_features_value['MaritalStatus'], value=default_values['MaritalStatus']),
    gr.Slider(label='Years In Current Role', value=default_values['YearsInCurrentRole'], minimum=0, maximum=40),
    gr.Dropdown(label='Education Field', choices=categorical_features_value['EducationField'], value=default_values['EducationField']),
    gr.Dropdown(label='Department', choices=categorical_features_value['Department'], value=default_values['Department']),
    gr.Slider(label='Total Working Years', value=default_values['TotalWorkingYears'], minimum=0, maximum=40),
    gr.Slider(label='Years WithCurr Manager', value=default_values['YearsWithCurrManager'], minimum=0, maximum=40),
    gr.Number(label='Number or Year Companies Worked', value=default_values['NumCompaniesWorked'], minimum=0, maximum=40),
]

app = gr.Interface(
    fn=predict_attrition,
    inputs=inputs,
    outputs='text',
    title="Employee Attrition Predictor",
)

app.launch(share=True)