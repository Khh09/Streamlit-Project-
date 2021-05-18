import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objs as go
import streamlit.components.v1 as components



image1= Image.open('undraw_Segment_analysis_re_ocsl.png')
image2= Image.open('undraw_color_palette_yamk.png')
st.set_option('deprecation.showPyplotGlobalUse', False)




st.title('Streamlit Project')
st.header('In this small project, I would like to explore a very underrated subject in the world of business: Customer Churn.')
st.write('In this app, I will attempt to **create a model** that can accurately predict / classify if a customer is likely to churn. I will also **analyze the data** to come up with a possible strategic retention plan. The data set that will be used in this analysis will come from the **Telco company**.')
st.markdown('But before exploring our data, let us become more familiar with what **customer churn** actually is')
st.header('What is Customer Churn?')
st.image(image1, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
st.markdown('Customer churn is the percentage of customers that stopped using your companys product or service during a certain time frame. You can calculate churn rate by dividing the number of customers you lost during that time period -- say a quarter -- by the number of customers you had at the beginning of that time period.')
st.header('What can be a Good Customer Churn Rate?')
st.image(image2, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
st.markdown('There is no easy way of identifying a perfect customer churn rate, especially since it highly depends on the type of industry you are working in. Some sectors have significantly higher rates of customer attrition than others.')

components.html("""<hr style="height:3px;border:none;color:#808080;background-color:#808080;" /> """)


df= pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
if st.checkbox('Show Dataset'):
    number = st.number_input('Number of Rows to View',5,100)
    st.dataframe(df.head(number))

components.html("""<hr style="height:3px;border:none;color:#808080;background-color:#808080;" /> """)

if st.button('Column Names'):
    st.write(df.columns)

components.html("""<hr style="height:3px;border:none;color:#808080;background-color:#808080;" /> """)

if st.checkbox('shape of Dataset'):
    data_dim = st.radio ('Show Dimension By',('Rows', 'Columns'))
    if data_dim == 'Row':
        st.text('Number of Rows')
        st.write(df.shape[0])
    if data_dim == 'Columns':
        st.text('Number of Columns')
        st.write(df.shape[1])
    else:
        st.write(df.shape)

components.html("""<hr style="height:3px;border:none;color:#808080;background-color:#808080;" /> """)

if st.checkbox('Select Columns to Show'):
    all_columns= df.columns.tolist()
    selected_columns = st.multiselect('Select', all_columns)
    new_df= df[selected_columns]
    st.dataframe(new_df)

components.html("""<hr style="height:3px;border:none;color:#808080;background-color:#808080;" /> """)

if st.button('Value Counts'):
    st.text('Value Counts By target/Class')
    st.write(df.iloc[:,-1].value_counts())
    st.info('This is an important observation since we want to know the churn rate')

components.html("""<hr style="height:3px;border:none;color:#808080;background-color:#808080;" /> """)

if st.button('Data Types'):
    st.write(df.dtypes)
    st.info('It seems we need to change the data types for Total Charges and Senior Citizen')

components.html("""<hr style="height:3px;border:none;color:#808080;background-color:#808080;" /> """)

if st.checkbox('fix Data Types'):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['SeniorCitizen'] = df['SeniorCitizen'].apply(str)
    st.write(df.dtypes)
    st.info('Great job, you were able to correctly allocate the right data types to the columns')

components.html("""<hr style="height:3px;border:none;color:#808080;background-color:#808080;" /> """)

if st.button('Check for missing values'):
    st.write(df.isna().sum())
    st.info('There seems to be a few missing values in the TotalCharges Column, we need to deal with them by imputing the mean')

components.html("""<hr style="height:3px;border:none;color:#808080;background-color:#808080;" /> """)

if st.checkbox('Fill in the Missing Values'):
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
    st.write('Total Number of Missing Values', df.isna().sum().sum())
    st.info('Impressive, you were able to fix the missing values with a press of a button!')

components.html("""<hr style="height:3px;border:none;color:#808080;background-color:#808080;" /> """)



st.title('Data Visualization')
st.header('Alright, Now that we have examined our data set and cleaned some parts of it, it is time to **Visualize** it')

components.html("""<hr style="height:3px;border:none;color:#808080;background-color:#808080;" /> """)

st.subheader('Customizable Plot')
all_columns_names= df.columns.tolist()
type_of_plot = st.selectbox('Select Type of Plot',['area','bar','line','hist','box','kde'])
selected_columns_names = st.multiselect('Select Columns To Plot', all_columns_names)

if st.button('Generate Plot'):
    st.success(f'Generate Customizable Plot of {type_of_plot} for {selected_columns_names}')

    if type_of_plot == 'area':
        cust_data = df[selected_columns_names]
        st.area_chart(cust_data)
    elif type_of_plot == 'bar':
        cust_data = df[selected_columns_names]
        st.bar_chart(cust_data)
    elif type_of_plot == 'line':
        cust_data = df[selected_columns_names]
        st.line_chart(cust_data)
    elif type_of_plot:
        cust_plot = df[selected_columns_names].plot(kind= type_of_plot)
        st.write(cust_plot)
        st.pyplot()

components.html("""<hr style="height:3px;border:none;color:#808080;background-color:#808080;" /> """)

st.header('Scatter Plot')
if st.checkbox('Scatter Plot'):
    x_options = [
    'MonthlyCharges', 'TotalCharges']
    x_axis = st.selectbox('Which value do you want to explore?', x_options)
    fig = px.scatter(df, x=x_axis, y='InternetService', hover_name='Churn', title=f'Internet Services ratings vs. {x_axis}')
    st.plotly_chart(fig)
    st.info('Since the data set highly features categorical data, numerical scatter plots will not be able to deliver good insight')

components.html("""<hr style="height:3px;border:none;color:#808080;background-color:#808080;" /> """)

st.header('Churn Pie Plot')
if st.checkbox('Churn Pie Plot'):
    fig2 = px.pie(df, names = 'Churn')
    st.plotly_chart(fig2)
    st.info('It is now clear that the company is losing a quarter of its customers')

components.html("""<hr style="height:3px;border:none;color:#808080;background-color:#808080;" /> """)

st.header('Categorical Pie Plot')
if st.checkbox('Categorical Pie Plot'):
    Names = ['InternetService','Contract','StreamingMovies','DeviceProtection','MultipleLines','Dependents','SeniorCitizen','gender','PaymentMethod']
    y_axis = st.selectbox('Which categories do you want to explore?', Names)
    fig3 = px.pie(df,names=y_axis, title=f'Pie Chart of: {y_axis}')
    st.plotly_chart(fig3)

components.html("""<hr style="height:3px;border:none;color:#808080;background-color:#808080;" /> """)

st.header('Count plot')
if st.checkbox("Count plot"):
                    all_columns_names = df.columns.tolist()
                    s=df[all_columns_names[0]].str.strip().value_counts()
                    col = st.selectbox("Choose Column",df.columns.tolist())
                    if st.button("Generate Count Plot"):
                        s=df[col].str.strip().value_counts()
                        trace  = go.Bar(
                                x=s.index,
                                y=s.values,
                                showlegend = True
                                )

                        layout = go.Layout(
                            title = 'Count of {}'.format(col),
                        )
                        data = [trace]
                        fig = go.Figure(data=data,layout = layout)
                        st.plotly_chart(fig)

components.html("""<hr style="height:3px;border:none;color:#808080;background-color:#808080;" /> """)


st.header('Pair plot')
if st.checkbox('Pair Plot'):
    Names = ['InternetService','Contract','StreamingMovies','DeviceProtection','MultipleLines','Dependents','SeniorCitizen','gender','PaymentMethod']
    y_axis = st.selectbox ('What do you want to plot?', Names)
    fig = px.histogram(df, x=y_axis, color="Churn", barmode='group')
    st.plotly_chart(fig)

components.html("""<hr style="height:3px;border:none;color:#808080;background-color:#808080;" /> """)










st.title('Machine Learning')
st.header('It is time to train our data set in order to predict customer churn')

clean_df= pd.read_csv('clean_df.csv')


def build_model(df):
    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1]

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100)

    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(Y.name)

    rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
        random_state=parameter_random_state,
        max_features=parameter_max_features,
        criterion=parameter_criterion,
        min_samples_split=parameter_min_samples_split,
        min_samples_leaf=parameter_min_samples_leaf,
        bootstrap=parameter_bootstrap,
        oob_score=parameter_oob_score,
        n_jobs=parameter_n_jobs)
    rf.fit(X_train, Y_train)

    st.subheader('2. Model Performance')


    Y_pred_train = rf.predict(X_train)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_train, Y_pred_train) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_train, Y_pred_train) )




    st.subheader('3. Model Parameters')
    st.write(rf.get_params())



#---------------------------------#


#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])


# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader('2.1. Learning Parameters'):
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
    parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
    parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

with st.sidebar.subheader('2.2. General Parameters'):
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
    parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. A Look at the label encoded dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use current Dataset'):

        customer_churn = pd.read_csv('clean_df.csv')
        X = clean_df.drop('Churn', axis = 1)
        Y = clean_df['Churn']
        df = pd.concat( [X,Y], axis=1 )

        st.markdown('The Customer Churn dataset is used as the example.')
        st.write(df.head(5))

        build_model(df)
