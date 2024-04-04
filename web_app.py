#!/usr/bin/env python
# coding: utf-8

# In[24]:


import streamlit as st
import pandas as pd
import scipy
import joblib
import sklearn
from sklearn.ensemble import GradientBoostingRegressor

# Load your models
model = joblib.load('gradient_boosting_regressor_model.pkl')
preprocessor = joblib.load('preprocessor.joblib')

def main():
    # Inject CSS for RTL support
    st.markdown(
        """
        <style>
        html {
            direction: rtl;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title('2024 Data Analyst Salary Calculator')
    
    col1, col2 = st.columns(2)
    
    with col1:
        is_sql_list = ['כן', 'לא', 'לעיתים נדירות']
        is_sql = st.selectbox("האם התפקיד כולל שימוש באס.קיו.אל?", is_sql_list)
        
    with col2:
        is_ml_list = ['לא', 'כן', 'לעיתים נדירות']
        is_python = st.selectbox("האם התפקיד כולל שימוש בשפת פייתון?", is_ml_list)
        
    col3, col4 = st.columns(2)

    with col3:
        is_viz_tool = ['Tableau', 'Power BI', 'Excel', 'Looker/Qlik/Python/R', 'Other', 'No Tool']
        viz_tool = st.selectbox("מהו כלי הויזואליזציה העיקרי בו אתה משתמש?", is_viz_tool)
    
    with col4:
        is_manager_list = ['לא', 'כן']
        is_manager = st.selectbox("האם תפקיד ניהולי?", is_manager_list)
        
    col5, col6 = st.columns(2)
    
    with col5:
        company_type_list = ['הייטק', 'תעשייה ישראלית', 'אחר']
        company_type = st.selectbox("סוג החברה", company_type_list)
        
    with col6:
        is_analyst_type = ['Business/Data analyst']
        analyst_type = st.selectbox("איזה סוג אנליסט אתה?", is_analyst_type)
    
    exp = st.slider("שנות נסיון", min_value=0, max_value=20, value=0)

    if st.button("Predict"):
        data = {
            'company_type': company_type, 'is_manager': is_manager, 'is_sql': is_sql, 'is_python': is_python,
            'year_of_surv': '2024', 'exp': exp, 'viz_tool': viz_tool, 'analyst_type': analyst_type
        }
        
        new_input_data = pd.DataFrame([list(data.values())], columns=['company_type', 'is_manager', 'is_sql', 'is_python', 'year_of_surv', 'exp', 'viz_tool', 'analyst_type'])
        
        prediction_input = preprocessor.transform(new_input_data)
        prediction = model.predict(prediction_input)
        prediction_formatted = f"{int(round(prediction[0], -2)):,}"
        
        st.markdown(f"<h2 style='text-align: center; color: black;'>Predicted Salary: {prediction_formatted} ₪</h2>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

