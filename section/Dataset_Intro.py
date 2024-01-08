import streamlit as st
from customtool import smalltools
import extra_streamlit_components as stx
import pandas as pd
from streamlit_extras.dataframe_explorer import dataframe_explorer
import warnings
import base64


@st.cache_resource
def load_dataset(file_path):
    return pd.read_csv(file_path)


@st.cache_resource
def load_and_encode_image(image_path):
    with open(image_path, 'rb') as f:
        data = f.read()
    encoded = base64.b64encode(data)
    return "data:image/png;base64," + encoded.decode("utf-8")


def description():
    st.image('static/description design.jpg')
    st.write("## ")
    st.divider()
    st.write("## Credit Card Churning")
    st.write("""
Credit card churning, the phenomenon where customers frequently switch their credit cards for better benefits or offers,
 poses a significant challenge for banks as it leads to a loss of loyal customer base and revenue. Predicting which 
 customers are likely to churn is crucial for banks to proactively offer tailored services or incentives, thereby 
 retaining these customers and maintaining a stable and profitable customer relationship. Since it is one of the popular
  issues in the financial field, the HPO experiment decided to be carried on a public dataset from Kaggle that 
  consisting of bank customers' information.""")

    st.write("### ")
    st.write('Click this button to Kaggle dataset.')
    st.link_button('Kaggle Portal', 'https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers')

    st.divider()
    st.write("## Dataset Description")
    st.markdown('''
    This dataset contains `10,128` records of bank customers' information, utilized for predicting customer churn from 
    credit card services. It comprises `20` features, including age, salary, marital status, credit card limit, and 
    credit card category, among others.
    ''')
    st.write("### ")
    st.write('Table below shows the description for each column in the dataset.')
    data = {
        "Column": ["Attrition_Flag", "Customer_Age", "Gender", "Dependent_count", "Education_Level", "Marital_Status",
                   "Income_Category", "Card_Category", "Months_on_book", "Total_Relationship_Count",
                   "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal",
                   "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct",
                   "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio"],
        "Description": ["Indicates whether the customer's account is active or has churned.", "Age of the customer.",
                        "Gender of the customer.", "Number of dependents for the customer.",
                        "Education level of the customer.", "Marital status of the customer.",
                        "Income category of the customer.", "Type of credit card held by the customer.",
                        "Number of months the customer has been with the bank.",
                        "Total number of products the customer has with the bank.",
                        "Number of months the customer has been inactive in the last 12 months.",
                        "Number of contacts between the customer and the bank in the last 12 months.",
                        "Credit limit of the customer.", "Total revolving balance on the customer's credit card.",
                        "Average open to buy credit line (unused portion of the credit line).",
                        "Change in transaction amount from Q4 to Q1.",
                        "Total transaction amount in the last 12 months.",
                        "Total number of transactions in the last 12 months.",
                        "Change in transaction count from Q4 to Q1.", "Average card utilization ratio."]
    }
    df = pd.DataFrame(data)
    st.table(df)

    st.divider()
    st.write("## Dataset Explorer")
    st.write("1000 records from the original dataset can be explored below.")
    dataset = load_dataset('data/random_sample.csv')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataframe = dataframe_explorer(dataset)
    st.dataframe(dataframe, use_container_width=True)


def eda_categorical():
    col1, col2 = st.columns(spec=2, gap='small')
    with col1:
        categorical_distribution = load_and_encode_image('static/categorical_distribution.png')
        st.image(categorical_distribution)

    with col2:
        st.markdown("""
        1. **Attrition_Flag**: The majority of the customers (83.9%) are existing customers, with a smaller portion (16.1%) having attrited. This indicates that the dataset primarily consists of current customers.

2. **Gender**: The distribution between male and female customers is relatively balanced, with a slightly higher percentage of female customers (52.9%) compared to male customers (47.1%).

3. **Education_Level**: The customers' education levels vary, with the majority holding high school diplomas (30.9%) and graduate degrees (19.9%). A significant portion is uneducated (14.7%) or have unknown education levels (15.0%), and a smaller group have completed college (10.0%), post-graduate (5.1%), or doctorate studies (4.5%).

4. **Marital_Status**: Married customers make up nearly half of the dataset (46.3%), followed by single customers (38.9%). A small percentage of customers have an unknown marital status (7.4%) or are divorced (7.4%).

5. **Income_Category**: The largest income category represented is "Less than \\$40K" (35.2%), followed by "\\$40K - \\$60K" (17.7%) and "\\$60K - \\$80K" (13.8%). Other categories include "\\$80K - \\$120K" (15.2%) and "\\$120K +" (7.2%), with a small portion having an unknown income category (11.0%).

6. **Card_Category**: The overwhelming majority of customers have a blue card (93.2%), indicating that it's the most common card type. A small minority have gold (1.1%), silver (5.5%), or platinum (0.2%) cards.
        """)


def eda_numerical():
    col1, col2 = st.columns(spec=2, gap='small')
    with col1:
        numerical_distribution = load_and_encode_image('static/numerical_distribution.png')
        st.image(numerical_distribution)

    with col2:
        st.markdown("""
        1. **Customer_Age**: The distribution of customer age appears to be fairly normally distributed with a slight right skew. The boxplot indicates that the bulk of customers fall between the mid-40s to mid-50s with some outliers on the higher age range.

2. **Dependent_count**: This variable shows that most customers have between 0 to 3 dependents. There are outliers present with a higher number of dependents.

3. **Months_on_book**: The number of months customers have been with the bank appears to be normally distributed with a peak around 36 months. The boxplot shows a fairly symmetric distribution without many outliers.

4. **Total_Relationship_Count**: It shows a multi-modal distribution, suggesting specific numbers of products are more common, such as 3 or 6.

5. **Months_Inactive_12_mon**: Most customers have been inactive for 1 to 3 months in the last 12 months, with a few outliers who have been inactive for longer.

6. **Contacts_Count_12_mon**: This variable shows that most customers had 2 to 3 contacts with the bank in the last 12 months, with a relatively small number of outliers indicating more frequent contact.

7. **Credit_Limit**: The distribution of credit limits is right-skewed, indicating that most customers have lower credit limits, with a few customers having very high credit limits.

8. **Total_Revolving_Bal**: This feature is also right-skewed, with many customers having lower revolving balances, and a few having higher balances.

9. **Avg_Open_To_Buy**: Similar to credit limit, this metric is right-skewed with most values clustered at the lower end of the scale.

10. **Total_Amt_Chng_Q4_Q1**: This variable shows the change in transaction amount from Q4 to Q1 and appears to be normally distributed with a peak around 0.7 to 0.8.

11. **Total_Trans_Amt**: The total transaction amount in the last 12 months is right-skewed with most customers having lower transaction amounts.

12. **Total_Trans_Ct**: The total number of transactions count over the last 12 months shows a distribution that is slightly left-skewed.

13. **Total_Ct_Chng_Q4_Q1**: The change in transaction count from Q4 to Q1 is normally distributed, similar to the total amount change.

14. **Avg_Utilization_Ratio**: The average card utilization ratio shows a right-skewed distribution, with a peak near zero and a long tail, indicating that while many customers have a low utilization ratio, there are some with higher ratios.
        """)


def show_correlation():
    col1, col2 = st.columns(spec=2, gap='small')
    with col1:
        correlation_pic = load_and_encode_image('static/correlation.png')
        st.image(correlation_pic)

    with col2:
        st.markdown("""There is a strong positive linear relationship between 'Avg Open To Buy' and 'Credit Limit', which makes sense as 'Avg Open To Buy' is typically calculated as the difference between the credit limit and the current balance on the account. Therefore, as the credit limit increases, the 'Avg Open To Buy' would also increase if the balance remains constant or does not increase proportionally.
        """)


def numerical2():
    col1, col2 = st.columns(spec=2, gap='small')
    with col1:
        age_credit_limit = load_and_encode_image('static/age_credit_limit.png')
        st.image(age_credit_limit)
    with col2:
        st.markdown("""
        The scatter plot for Customer_Age vs Credit_Limit doesn't show a distinct pattern, suggesting that there may not be a strong linear relationship between age and credit limit. 
        """)

    col3, col4 = st.columns(spec=2, gap='small')
    with col3:
        months_on_book_total_transaction_amount = load_and_encode_image('static/months_on_book_total_transaction_amount.png')
        st.image(months_on_book_total_transaction_amount)
    with col4:
        st.markdown("""
        The scatter plot suggests there isn't a strong linear relationship between 'Months on book' and 'Total Transaction Amount'.
        """)

    col5, col6 = st.columns(spec=2, gap='small')
    with col5:
        total_revolving_balance_avg_utilization_ratio = load_and_encode_image('static/total_revolving_balance_avg_utilization_ratio.png')
        st.image(total_revolving_balance_avg_utilization_ratio)
    with col6:
        st.markdown("""
        The scatter plot for Total_Revolving_Bal vs Avg_Utilization_Ratio indicates a possible positive correlation, as expected since utilization ratio is partially derived from the revolving balance.
        """)


def categorical_numerical():
    col1, col2 = st.columns(spec=2, gap='small')
    with col1:
        credit_limit_education_level = load_and_encode_image('static/credit_limit_education_level.png')
        st.image(credit_limit_education_level)
    with col2:
        st.markdown("""
- The box plots indicate that there is some variation in credit limits among different education levels, although it does not appear to be a strong distinguishing factor.
- All categories show a wide range of credit limits with a large number of outliers on the higher end, suggesting that individuals with higher education levels do not necessarily have significantly higher credit limits than those with lower education levels.
- However, those with a doctorate show slightly higher median credit limits compared to other education levels.
""")

    col3, col4 = st.columns(spec=2, gap='small')
    with col3:
        average_total_transaction_amount_income_category = load_and_encode_image('static/average_total_transaction_amount_income_category.png')
        st.image(average_total_transaction_amount_income_category)
    with col4:
        st.markdown("""
- The highest average transaction amount is observed in the "60K - 80K" income category, followed closely by the "80K - 120K" and "40K - 60K" categories.
- Interestingly, customers with an income of "Less than $40K" and those with "Unknown" income levels have a lower average transaction amount compared to the middle-income categories.
""")

    col5, col6 = st.columns(spec=2, gap='small')
    with col5:
        average_total_revolving_balance_card_category = load_and_encode_image('static/average_total_revolving_balance_card_category.png')
        st.image(average_total_revolving_balance_card_category)
    with col6:
        st.markdown("""
- Customers with gold cards have the highest average revolving balance at approximately \\$1344.3, indicating that they might be utilizing their credit line more than other cardholders.
- Platinum cardholders follow closely with an average revolving balance of about \\$1268.0. This group also tends to maintain a significant revolving balance, which could suggest a comfort level with utilizing available credit.
- Silver cardholders have an average revolving balance of around \\$1206.1, slightly less than platinum cardholders, but still indicative of substantial credit usage.
- Blue cardholders, which likely represent the entry-level card offering, have the lowest average revolving balance of \\$1157.8 among the card categories.
""")

    col7, col8 = st.columns(spec=2, gap='small')
    with col7:
        average_transaction_amount_card_category = load_and_encode_image('static/average_transaction_amount_card_category.png')
        st.image(average_transaction_amount_card_category)
    with col8:
        st.markdown("""
- Customers with platinum cards have the highest average transaction amount, coming in at approximately \\$8999.8. This suggests that customers who have platinum cards, which are often associated with higher credit limits and more features, tend to spend more.
- The next highest average transaction amount is seen with gold cardholders, averaging around \\$7685.6, followed by silver cardholders with \\$6590.5.
- Customers with blue cards, which are typically the most basic card type offered, have the lowest average transaction amount of \\$4225.4.
- This pattern indicates a clear correlation between the tier of the card category and the average transaction amount. Higher-tier cards (platinum and gold) are associated with higher average transaction amounts, which may reflect the spending power and behavior of the customers who hold these cards.
""")


def categorical2():
    col1, col2 = st.columns(spec=2, gap='small')
    with col1:
        attrition_flag_gender = load_and_encode_image('static/attrition_flag_gender.png')
        st.image(attrition_flag_gender)
    with col2:
        st.markdown("""
For both genders, the majority of customers are existing (non-attrited), with a smaller percentage having attrited.
The exact percentages are annotated on each bar, showing the proportion of attrited and existing customers within each gender category.
""")

    col3, col4 = st.columns(spec=2, gap='small')
    with col3:
        education_level_marital_status = load_and_encode_image('static/education_level_marital_status.png')
        st.image(education_level_marital_status)
    with col4:
        st.markdown("""
- For each marital status, the proportions of education levels are stacked to sum to 100%, allowing for an easy comparison of the educational composition within each category.
- Divorced customers have the highest proportion of individuals with a high school education, while married customers have the highest proportion of college-educated individuals.
- Single and unknown marital status categories have a more even distribution across high school, graduate, and college education levels.
- Post-graduate levels are relatively low across all marital statuses but are most prevalent among married customers.
- The 'Unknown' education level is significant across all marital statuses, indicating a sizable proportion of customers whose education level is not recorded or is ambiguous.
""")

    col5, col6 = st.columns(spec=2, gap='small')
    with col5:
        attrition_flag_others = load_and_encode_image(
            'static/attrition_flag_others.png')
        st.image(attrition_flag_others)
    with col6:
        st.markdown("""
**Attrition Flag by Card Category**:  

- The majority of customers in each card category are existing customers. However, the attrition seems to be relatively higher in the platinum and silver card categories compared to blue and gold.
- The blue card category has the highest number of customers overall, with the vast majority being existing customers and a small number who have attrited.
- Gold cards have a higher number of existing customers compared to attrited customers, but the absolute numbers are much smaller than in the blue card category.
- Platinum and silver card categories have the lowest total counts but show a visible proportion of attrition.


**Attrition Flag by Income Category**:

- Customers in the "Less than \\$40K" income category show the highest count of attrition compared to other income categories.
"40K - \\$60K" and "60K - \\$80K" income categories follow with the second and third highest counts of attrition, respectively.
- The "\\$120K+" category, while having a smaller customer base, shows a relatively low number of attritions.


**Attrition Flag by Education Level**:

- Customers with a graduate education level show the highest count of attrition, followed by those with a high school education.
- Doctorate and post-graduate customers have the lowest counts of attrition, which might suggest a correlation between higher education levels and lower attrition rates.
- The "Unknown" education level category also shows a significant count of attrition, indicating that not having this information might be associated with a higher attrition rate.
""")


def temporal_numerical():
    col1, col2 = st.columns(spec=2, gap='small')
    with col1:
        average_utilization_ratio_months_inactive = load_and_encode_image('static/average_utilization_ratio_months_inactive.png')
        st.image(average_utilization_ratio_months_inactive)
    with col2:
        st.markdown("""
- The average utilization ratio decreases sharply after the first month of inactivity and then levels off.
- It appears that customers who are inactive for one month have a higher average utilization ratio, which might indicate that these customers are temporarily inactive but still maintain their credit usage.
- After one month of inactivity, the utilization ratio remains relatively consistent across the 2-6 month inactivity range, suggesting that longer-term inactive customers tend to use a smaller portion of their credit line.
""")

    col3, col4 = st.columns(spec=2, gap='small')
    with col3:
        total_transaction_amount_months_on_book = load_and_encode_image('static/total_transaction_amount_months_on_book.png')
        st.image(total_transaction_amount_months_on_book)
    with col4:
        st.markdown("""
- The line graph fluctuates significantly, suggesting that the average total transaction amount varies considerably from month to month.
- There are peaks in the transaction amount at various points, most notably in the early months and around the 20-30 month range. This could indicate periods when customers are more actively transacting, possibly due to promotions or lifecycle events.
- The general trend does not show a clear increase or decrease over time, implying that the relationship between the length of the customer's relationship with the bank (months on book) and their transaction amount is not linear.
""")


def temporal_categorical():
    col1, col2 = st.columns(spec=2, gap='small')
    with col1:
        attrition_flag_months_on_book = load_and_encode_image('static/attrition_flag_months_on_book.png')
        st.image(attrition_flag_months_on_book)
    with col2:
        st.markdown("""
- The counts of attrition (both for attrited and existing customers) are relatively stable across the majority of the time span, with a significant spike in attrition counts around 36 months.
- This spike could indicate a common point in the customer lifecycle where a large number of customers decide to leave, possibly due to the end of a promotional period or a common contract term.
- After the spike, the counts for existing customers continue as before, indicating that the event at 36 months does not significantly impact the remaining customer base in the long term.
""")


def eda():
    st.write('Exploratory Data Analysis (EDA) is performed in order to understand the dataset. This can be useful in '
             'determining dataset distribution for identifying outliers, getting insights of the relationships between '
             'different variables and collecting information that might be useful in data processing stage later. Normally, '
             'univariate and bivariate analysis are performed in order to get these informations.')
    st.divider()
    st.write("## Statistical Summary")
    st.write("Explore the most common/mode, median and mean for each variables along with other information.")

    st.write("#### ")
    st.write("### Numerical Variables")
    numerical_statistic = load_dataset('data/numerical_statistic.csv')
    st.dataframe(numerical_statistic)

    st.write("#### ")
    st.write("### Categorical Variables")
    categorical_statistic = load_dataset('data/categorical_statistic.csv')
    st.dataframe(categorical_statistic)

    st.divider()
    st.write("## Univariate Analysis")
    st.write('This section mainly shows the distribution/frequency of a variable. It is useful in identifying outliers, '
             'rare values and other purposes.')
    st.write("### ")
    chosen_option = stx.tab_bar(data=[
        stx.TabBarItemData(id=3, title="Categorical Variables", description=""),
        stx.TabBarItemData(id=4, title="Numerical Variables", description=""),
    ], default=3)

    if int(chosen_option) == 3:
        eda_categorical()
    elif int(chosen_option) == 4:
        eda_numerical()
        
    st.divider()
    st.write("## Bivariate Analysis")
    st.write('This section mainly shows the relationships between two variables. It can shows the patterns in the data '
             'which is helpful in prediction problem later.')
    st.write("### ")
    chosen_option = stx.tab_bar(data=[
        stx.TabBarItemData(id=5, title="Correlation", description=""),
        stx.TabBarItemData(id=6, title="Numerical-Numerical", description=""),
        stx.TabBarItemData(id=7, title="Categorical-Numerical", description=""),
        stx.TabBarItemData(id=8, title="Categorical-Categorical", description=""),
        stx.TabBarItemData(id=9, title="Temporal-Numerical", description=""),
        stx.TabBarItemData(id=10, title="Temporal-Categorical", description="")
    ], default=5)
    
    if int(chosen_option) == 5:
        show_correlation()
    elif int(chosen_option) == 6:
        numerical2()
    elif int(chosen_option) == 7:
        categorical_numerical()
    elif int(chosen_option) == 8:
        categorical2()
    elif int(chosen_option) == 9:
        temporal_numerical()
    elif int(chosen_option) == 10:
        temporal_categorical()


def load_page():
    st.set_page_config(
        page_title="HPO Experiment",
        page_icon=smalltools.page_icon(),
        layout='wide'
    )

    smalltools.hide_unused_pages()
    st.title("Get Insight about Dataset")
    st.write('Please choose the section you are interested in. 👇')
    st.write("### ")

    chosen_option = stx.tab_bar(data=[
        stx.TabBarItemData(id=1, title="Dataset Description", description=""),
        stx.TabBarItemData(id=2, title="Exploratory Data Analysis", description=""),
    ], default=1)

    if int(chosen_option) == 1:
        description()
    elif int(chosen_option) == 2:
        eda()


load_page()