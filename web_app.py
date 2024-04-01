#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import scipy
import joblib
import pickle
import sklearn
#import sklearn.ensemble._gb_losses
import sys




# Print Python version and environment details


st.write(sys.version)
st.write(sys.prefix)


#picklefile = open("egradient_boosting_regressor_model.pkl", "rb")
#model = pickle.load(picklefile)

#model = joblib.load('gradient_boosting_regressor_model.pkl')

def main():

    print("Python version:", sys.version)
    print("Python environment:", sys.prefix)
    model = pd.read_pickle('gradient_boosting_regressor_model.pkl')
    preprocessor = joblib.load('preprocessor.joblib')


    st.title('Salary Prediction Web App')
#     company_type = st.selectbox("סוג החברה",["ass","saxaas"]) 

    company_type_list=['תעשייה ישראלית', 'הייטק', 'אחר']
    is_manager_list=['לא', 'כן']
    is_sql_list=['לא', 'כן', 'לעיתים נדירות']
    is_ml_list=['לא', 'כן', 'לעיתים נדירות']
    is_viz_tool=['Other', 'No Tool', 'Tableau', 'Excel', 'Looker/Qlik/Python/R', 'Power BI']
    is_job_location=['מרכז', 'שאר הטארץ', 'אזור תל אביב', 'עבודה מהבית']
    is_analyst_type=['Business analyst','Financial Analyst','Marketing','Data sceintist','other','BI']

    company_type = st.selectbox("סוג החברה", company_type_list) 
    is_manager = st.selectbox("האם תפקיד ניהולי?", is_manager_list) 
    is_sql = st.selectbox("האם התפקיד כולל שימוש באס.קיו.אל?", is_sql_list) 
    is_ml = st.selectbox("האם התפקיד כולל פיתוח מודלי ניבוי?", is_ml_list) 
    viz_tool = st.selectbox("מהו כלי הויזואליזציה העיקרי בו אתה משתמש?", is_viz_tool) 
    job_location = st.selectbox("היכן ממוקמים המשרדים?", is_job_location) 
    analyst_type = st.selectbox("איזה סוג אנליסט אתה?", is_analyst_type) 
    exp = st.text_input("שנות נסיון") 


    
    if st.button("Predict"): 
        features = [[company_type,is_manager,is_sql,is_ml,exp]]
        data = {'company_type': company_type, 'is_manager': is_manager, 'is_sql': is_sql, 'is_ml': is_ml,
                'year_of_surv':'2024','exp': float(exp),'viz_tool':viz_tool,'job_location':job_location,
                'analyst_type':analyst_type, }
        
        print(data)
        new_input_data=pd.DataFrame([list(data.values())], 
                                    columns=['company_type','is_manager','is_sql','is_ml','year_of_surv','exp',
                                             'viz_tool','job_location','analyst_type'])
                
        feature_names = preprocessor.get_feature_names_out()

        # Create a DataFrame template
        template_df = pd.DataFrame(columns=feature_names, dtype=float)
        template_df.loc[0] = 0.0  # Adding a row of zeros
        # Transform new input data
        new_input_transformed = preprocessor.transform(new_input_data)

        # If using a sparse matrix, convert to a dense format
        if scipy.sparse.issparse(new_input_transformed):
            new_input_transformed_dense = new_input_transformed.toarray()
        else:
            new_input_transformed_dense = new_input_transformed

        new_input_df = pd.DataFrame(new_input_transformed_dense, columns=preprocessor.get_feature_names_out())

#         # Align new input data with the template
#         aligned_input = template_df.copy()
#         aligned_input.loc[0] = 0.0  # Reset to zeros
#         for col in new_input_df.columns:
#             if col in aligned_input.columns:
#                 aligned_input[col] = new_input_df[col].values

        # Now, aligned_input has the same shape and order as expected by the model
        prediction_input = new_input_df.values  # Use this for prediction
        prediction = model.predict(new_input_df)
        st.write("### Predictions:")
        st.write(int(prediction[0]))
    

#if __name__ == "__main__":
#    main()


