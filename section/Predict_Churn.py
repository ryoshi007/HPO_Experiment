import streamlit as st
from customtool import smalltools
import pandas as pd
import random
import joblib


def map_income_category(category):
    mapping = {'Unknown': 0, 'Less than $40K': 1, '$40K - $60K': 2, '$60K - $80K': 3, '$80K - $120K': 4, '$120K +': 5}
    return mapping[category]


@st.cache_data
def load_data(csv_file):
    return pd.read_csv(csv_file)


@st.cache_resource
def load_scaler():
    return joblib.load('model/scaler.pkl')


@st.cache_resource
def load_models():
    models = {
        'Logistic Regression': joblib.load('model/logistic_regression_model.pkl'),
        'Support Vector Classifier': joblib.load('model/support_vector_classifier_model.pkl'),
        'Random Forest Classifier': joblib.load('model/random_forest_classifier_model.pkl'),
        'Gradient Boosting Classifier': joblib.load('model/gradient_boosting_classifier_model.pkl')
    }
    return models


def generate_data(data):
    random_index = random.randint(0, len(data) - 1)
    random_record = data.iloc[random_index]

    for key in ['total_trans_ct', 'total_ct_chng_q4_q1', 'total_revolving_bal', 'contacts_count_12_mon',
                'avg_utilization_ratio', 'months_inactive_12_mon', 'gender', 'total_relationship_count',
                'total_amount_chg_q4_q1', 'credit_limit', 'dependent_count', 'total_trans_amt', 'income_category',
                'marital_status_married']:
        st.session_state[key] = random_record[key]


def load_page():
    st.set_page_config(
        page_title="HPO Experiment",
        page_icon=smalltools.page_icon(),
        layout='wide'
    )

    smalltools.hide_unused_pages()
    data = load_data('data/sample_dataset.csv')
    st.title("Predict Credit Card Churn")
    st.write('This section consists of 4 different machine learning models that had been trained in predicting '
             'credit card churn using the dataset.')
    st.divider()
    st.markdown("## Step 1 - Input Data")

    st.write('Click this to generate data')
    if st.button("Generate Customer's Data", on_click=lambda: generate_data(data)):
        pass

    st.write('## ')
    col1, col2 = st.columns(2)

    with st.form(key='user_input_form'):
        with col1:
            total_trans_ct = st.number_input('Total Transaction Count', step=1, key='total_trans_ct')
            total_ct_chng_q4_q1 = st.number_input('Total Count Change Q4 to Q1', step=0.1, key='total_ct_chng_q4_q1', format="%.3f")
            total_revolving_bal = st.number_input('Total Revolving Balance', step=1, key='total_revolving_bal')
            contacts_count_12_mon = st.number_input('Number of Contacts with Bank in Last 12 Months', step=1, key='contacts_count_12_mon')
            avg_utilization_ratio = st.number_input('Average Card Utilization Ratio', step=0.1, key='avg_utilization_ratio', format="%.3f")
            dependent_count = st.number_input('Number of Dependents', step=1, key='dependent_count')
            gender = st.radio('Gender', ['Male', 'Female'], key='gender')

        with col2:
            total_relationship_count = st.number_input('Total Number of Products with Bank', step=1, key='total_relationship_count')
            total_amount_chg_q4_q1 = st.number_input('Change in Transaction from Q4 to Q1', step=0.1, key='total_amount_chg_q4_q1', format="%.3f")
            credit_limit = st.number_input('Credit Limit', step=1, key='credit_limit')
            months_inactive_12_mon = st.number_input('Number of Inactive Months in Last 12 Months', step=1, key='months_inactive_12_mon')
            total_trans_amt = st.number_input('Total Transaction Amount', step=1, key='total_trans_amt')
            income_category = st.selectbox('Income Category',
                                           ['Unknown', 'Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K',
                                            '$120K +'], key='income_category')
            marital_status_married = st.radio('Married Status', ['Yes', 'No'], key='marital_status_married')

        st.write('Click this to confirm input submission')
        submit_button = st.form_submit_button(label='Submit')

    prediction_df = None

    if submit_button:
        gender_f = 1 if gender == 'Female' else 0
        marital_status_married = 1 if marital_status_married == 'Yes' else 0
        income_category = map_income_category(income_category)

        user_input = [
            total_trans_ct, total_ct_chng_q4_q1, total_revolving_bal,
            contacts_count_12_mon, avg_utilization_ratio, total_trans_amt,
            months_inactive_12_mon, total_relationship_count,
            total_amount_chg_q4_q1, credit_limit, gender_f, dependent_count,
            marital_status_married, income_category
        ]

        numerical_features = [
            'total_trans_ct', 'total_ct_chng_q4_q1', 'total_revolving_bal',
            'contacts_count_12_mon', 'avg_utilization_ratio', 'total_trans_amt',
            'months_inactive_12_mon', 'total_relationship_count',
            'total_amount_chg_q4_q1', 'credit_limit', 'dependent_count'
        ]

        user_input_df = pd.DataFrame([user_input], columns=[
            'total_trans_ct', 'total_ct_chng_q4_q1', 'total_revolving_bal',
            'contacts_count_12_mon', 'avg_utilization_ratio', 'total_trans_amt',
            'months_inactive_12_mon', 'total_relationship_count',
            'total_amount_chg_q4_q1', 'credit_limit', 'gender_f',
            'dependent_count', 'marital_status_married', 'income_category'
        ])

        scaler = load_scaler()
        models = load_models()

        numerical_input_scaled = scaler.transform(user_input_df[numerical_features])
        user_input_df[numerical_features] = numerical_input_scaled

        prediction_mapping = {1: 'Attrited Customer', 0: 'Existing Customer'}

        predictions = []
        for model_name, model in models.items():
            pred = model.predict(user_input_df)[0]  # Assuming a single row input
            predictions.append([model_name, prediction_mapping[pred]])

        # Create a DataFrame for displaying the results
        prediction_df = pd.DataFrame(predictions, columns=['Model', 'Prediction'])

    st.divider()
    st.markdown("## Step 2 - Result")
    st.write('View the results predicted from different models')
    st.write('## ')
    if prediction_df is not None:
        st.dataframe(prediction_df)
    else:
        st.write("No prediction made. Please click the 'Submit' button.")


load_page()