#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import pandas as pd
import scipy
import joblib
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np




class YearToCategoryTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Assuming X is a DataFrame for simplicity
        self.feature_names_out_ = X.columns.tolist()  # Capture input feature names
        return self
    
    def transform(self, X, y=None):
        # Transform logic here (your existing transformation code)
        # Example transformation:
        transformed_X = np.digitize(X, bins=[2020, 2021, 2022, 2024])
        return transformed_X

    def get_feature_names_out(self, input_features=None):
        # Return the output feature names
        return self.feature_names_out_

# Load your models (ensure these files exist and are in the correct path)
model = joblib.load('gradient_boosting_regressor_model.pkl')
preprocessor = joblib.load('preprocessor.joblib')

def main():
 
    st.markdown(
        """
        <style>
        /* Ensure the entire app respects RTL */
        body {
            direction: rtl;
        }
        /* Adjust the footer to be a bit lower and ensure it's centered */
        .footer {
            position: fixed;
            right: 0;
            bottom: 10px; /* Adjust this value if you need more space */
            left: 0;
            width: 100%;
            text-align: center;
            padding: 10px; /* Add some padding */
            background-color: transparent; /* Keep the background transparent or set a color */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1 style='text-align: center;'>מחשבון שכר לאנליסטים 2024</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <h2 style='text-align: center; color: gray; font-size: 16px;'>המחשבון מבוסס על תוצאות סקר אנליסטים שנערך בקבוצת <a href="https://www.facebook.com/groups/DataAnalyticsIsrael" target="_blank">Data Analyst</a></h2>
    <br>
    """, unsafe_allow_html=True)

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
    
    exp = st.number_input("שנות נסיון", min_value=0, max_value=20, value=0, format="%d")

    if st.button("Predict"):
        data = {'company_type': company_type, 'is_manager': is_manager, 'is_sql': is_sql, 'is_python': is_python,
                'year_of_surv': 2024, 'exp': exp, 'viz_tool': viz_tool, 'analyst_type': analyst_type}

        new_input_data = pd.DataFrame([list(data.values())], columns=['company_type', 'is_manager', 'is_sql', 'is_python', 'year_of_surv', 'exp', 'viz_tool', 'analyst_type'])

        prediction_input = preprocessor.transform(new_input_data)
        prediction = model.predict(prediction_input)
        prediction_formatted = f"{int(round(prediction[0], -2)):,}"

        st.markdown(f"<h2 style='text-align: center; color: black;'>שכר מוערך צפוי: {prediction_formatted} ₪</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
    פותח על ידי <a href="https://www.linkedin.com/in/daniel-raz-1747b2118/" target="_blank">דניאל רז</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

