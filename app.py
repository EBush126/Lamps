from pyparsing import col
import streamlit as st
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import Lamp
import pickle
from pickle import load
import time
import altair as alt
from Lamp import predict_isRecommended,load_split,train,validate

# Page setting 
st.set_page_config(page_title="MGSC-7650 Gr 2 Streamlit App", layout="wide")
st.markdown('''
<style>
    #MainMenu
    {
        display: none;
    }
    .css-18e3th9, .css-1d391kg
    {
        padding: 1rem 2rem 2rem 2rem;
    }
</style>
''', unsafe_allow_html=True)

# Pages
page_selected = st.sidebar.radio("Menu", ["Home", "Model", "About"])

st.header("TrueReview")
st.markdown('---')

# Application template
background = st.container()
train_dataset = st.container()
model_training = st.container()
product_details = st.container()


# Data
train_data = 'Lamp_imputed.csv'

if page_selected == "Home":
    df=pd.read_csv("Lamp_imputed.csv")
    df=df[["ProductID","Review","Rating","IsRecommended"]]
    interior = Image.open('Interior.jpg')
    st.image(interior, use_column_width=True)
    st.subheader("Description")
    st.write("Target is an iconic brand, a Fortune 50 company and one of America's leading retailers. With the chance to positively impact nearly millions people worldwide, Target takes the customer feedbacks seriously. Our mission is to focus on the Lamp products, and based on the reviews, to predict the recommended levels from customers.")
    st.subheader("Insights")
    co1,col2 = st.columns(2)
    with co1:
        st.write("1. More than 50% of customer will give 3-4 ratings for Target Lamps")
    with col2:
        st.write("🤩🤩🤩🤩🤩")
    co1,col2 = st.columns(2)
    with co1:
        st.write("2. Customers who give ratings under 3.25 are more likely to NOT RECOMMENDED the products, who give ratings from 3.25 to 5 are more likely to RECOMMEND the products​")
    with col2:
        st.write("👌👌👌👌👌")
    co1,col2 = st.columns(2)
    with co1:
        st.write("3. Most of the reviews for the products are multi-dimensional, including video, pictures, comments for price, quality, and user experience")
    with col2:
        st.write("😭😭😭😭😭")
    co1,col2 = st.columns(2)
    with co1:
        st.write("4. The Model to predict recommended level has accuracy score higher than 85%, most of the customers who gave positive reviews are likely to recommend products to others")
    with col2:
        st.write("😀😀😀😀😀")
    co1,col2 = st.columns(2)
    with co1:
        st.write("5. Overall, if we focus on the reviews wrote by customers, to decide whether we should buy the products, it will reduce both time and effort")
    with col2:
        st.write("🤣🤣🤣🤣🤣") 

    st.subheader("Actual Result vs. Predicted Result")
    if st.button('Test Original Data'):
        Prediction = pd.read_csv('Lamp_imputed.csv',index_col = False)
        Prediction = predict_isRecommended(Prediction)

        Prediction['Correct'] = Prediction['IsRecommended'] == Prediction['Prediction']
        col1, col2 = st.columns(2)
        with col1:
            st.write('Actual Result')
            st.write(Prediction[['Review','IsRecommended','Prediction']])
        with col2:
            st.write('Predicted Result')            
            st.write(Prediction['Correct'])
        st.markdown('---')
        st.subheader("Prediction Accuracy")

        ax = pd.crosstab(Prediction.IsRecommended, Prediction.Correct).plot(
            kind="bar",
            figsize=(9,4),
            xlabel = "Actual Recommendation Status",
            color = ['r', 'k'])
        ax.legend(['Incorrect Prediction', 'Correct Prediction'])
        st.pyplot(ax.figure)
        
        st.subheader('Predictions by Rating')
        col1, col2 = st.columns((2,1))
        with col1: 
            ax = pd.crosstab(Prediction.Rating, Prediction.Prediction).plot( 
                kind="bar",  
                figsize=(6,3),  
                xlabel = "Rating",
                color = ['r', 'k']) 
            st.pyplot(ax.figure)
        with col2:
            st.write('This plot shows the total count of predicted recommendation status for different lamps')
        st.markdown('---')
        
        #another chart
        st.subheader('Predictions by Product')
        col1, col2 = st.columns((2,1))
        with col1: 
            ax = pd.crosstab(Prediction.ProductID, Prediction.Prediction).plot( 
                kind="bar",  
                figsize=(6,3),  
                xlabel = "ProductID",
                color = ['r', 'k']) 
            st.pyplot(ax.figure)
        with col2:
            st.write('This chart shows the total count of predicted positive & negative reviews for different lamp products')
        st.markdown('---')

    if st.button('Test New Data'):
        Prediction = pd.read_csv('Lamp_new_and_unseen.csv',index_col = False)
        Prediction = predict_isRecommended(Prediction)

        Prediction['Correct'] = Prediction['IsRecommended'] == Prediction['Prediction']
        col1, col2 = st.columns(2)
        with col1:
            st.write('Actual Result')
            st.write(Prediction[['Review','IsRecommended','Prediction']])
        with col2:
            st.write('Predicted Result')            
            st.write(Prediction['Correct'])
        st.markdown('---')
        st.subheader("Prediction Accuracy")

        ax = pd.crosstab(Prediction.IsRecommended, Prediction.Correct).plot(
            kind="bar",
            figsize=(9,4),
            xlabel = "Actual Recommendation Status",
            color = ['r', 'k'])
        ax.legend(['Incorrect Prediction', 'Correct Prediction'])
        st.pyplot(ax.figure)
        
        st.subheader('Predictions by Rating')
        col1, col2 = st.columns((2,1))
        with col1: 
            ax = pd.crosstab(Prediction.Rating, Prediction.Prediction).plot( 
                kind="bar",  
                figsize=(6,3),  
                xlabel = "Rating",
                color = ['r', 'k']) 
            st.pyplot(ax.figure)
        with col2:
            st.write('This plot shows the total count of predicted recommendation status for different lamps')
        st.markdown('---')
        
        #another chart
        st.subheader('Predictions by Product')
        col1, col2 = st.columns((2,1))
        with col1: 
            ax = pd.crosstab(Prediction.ProductID, Prediction.Prediction).plot( 
                kind="bar",  
                figsize=(6,3),  
                xlabel = "ProductID",
                color = ['r', 'k']) 
            st.pyplot(ax.figure)
        with col2:
            st.write('This chart shows the total count of predicted positive & negative reviews for different lamp products')
        st.markdown('---')
    


elif page_selected == "Model":
    with background:
        st.subheader("About the Training Dataset")
        st.write("To obtain our data for training our model, we used a real-time Target.com API to obtain review information for different lamps currently sold at Target stores.")
        st.image("https://images.unsplash.com/photo-1615557854978-2eac0cd47b0d", caption="Photo credit: Daniel ODonnell")
   
    with train_dataset:
        # Dataset preparation
        df = pd.read_csv(train_data)
        st.subheader("Dataset Preparation")
        col1, col2 = st.columns(2)
        with col1:
            st.write("We created a dataset containing product IDs, text reviews, ratings (1 – 5 stars),and recommendation statuses (whether the reviewer would recommend buying their purchased lamp).The extracted data was already considerably clean, but to aid in model performance and help in balancing the data, we imputed missing recommended values as 'not recommended’ for reviews with ratings 3 or lower.")
        with col2:
            st.write("Reviews that gave 4 or 5 stars with missing or negative recommendation statuses were removed from the dataset.  The final training dataset contained 889 reviews with recommended status and 275 with not recommended status.  The text review was to be the input for our model and the recommendation status would be the target variable.")
        
        # Dataset sample & size
        st.subheader("Dataset Overview")
        st.markdown('Dataset Size:')
        st.write('Total number of records is:', df.shape[0])

        st.subheader("Sample Reviews and Recommendation Status")
        num_of_samples = st.selectbox("Select number of Samples", [1, 2, 5, 10, 15])
        df_sample = df.sample(num_of_samples)
        for index, row in df_sample.iterrows():
            col1, col2 = st.columns((1,5))
            with col1:
                if row['IsRecommended'] == True:
                    st.success("Recommended") 
                else: 
                    st.error("Not Recommended")    
            with col2:
                st.write("ProductID:",row['ProductID'])
                st.write("Rating:",row['Rating'])
                st.write("Review:",row['Review'])
        st.markdown('---')


        st.subheader('Recommendation in Rating Overview:')
        ax = pd.crosstab(df.Rating, df.IsRecommended).plot(
        kind="bar",
        figsize=(6,2),
        xlabel = "Rating",
        color = ['r', 'k'])
        ax.legend(['No', 'Yes'])
        st.pyplot(ax.figure)
        st.markdown('---')
    
    with model_training:
        st.subheader("Model Training and Validation")
        col1, col2 = st.columns(2)
        with col1:
            st.write("We started by considering the different models that might be applicable for this sort of problem. We are doing text analysis, with no classes and only two values for our target variable (recommended or not recommended). This meant a vectorizer was necessary to tokenize the data (in this case, TF-IDF). Our initial ideas for models included:")
            st.write("- Linear SVC")
            st.write("- Multinomial/Gaussian Naïve Bayes")
            st.write("- Boosted Trees")
        with col2:
            st.write("We attempted both Linear SVC and Multinomial NB solutions. While the Multinomial NB model showed improvements in validation, we generally got better results with less tuning via Linear SVC. We modified only a few hyperparameters – c = 3. This regularization parameter was adjusted to reduce overfitting. This value yielded the best test accuracy, whereas higher or lower values of c generally underperformed. We did not perform grid search or any more serious hyperparameter tuning for the time being.")
    
    # Model attempts:
    with st.expander("Model attempts:"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('An attempt at GridSearchCV with a Perceptron. The accuracy was not impressive:')
            failed_model_ex = Image.open("failed_model_ex.png")
            st.image(failed_model_ex, use_column_width=True)
        with col2:
            failed_model_accu = Image.open("failed_model_acc.png")
            st.markdown('See here the validation accuracy of only 0.8 - far lower than our final model:')
            st.image(failed_model_accu, use_column_width=True)

    # Uploading the dataset
    with st.expander("Ingest form (please provide a compatible csv)!", expanded = False):
        input_file = st.file_uploader("Upload here", type=['csv'])
        if (input_file is not None) and input_file.name.endswith(".csv"):
            df = pd.read_csv(input_file)
            st.dataframe(df)
            if st.button('Run sentiment analysis?'):
                X_train, X_test, y_train, y_test = Lamp.load_split(df)
                mean_cross_val_accuracy, pipeline = Lamp.train(X_train, y_train)
                test_accuracy, confusion_matrix = Lamp.validate(X_test, y_test, pipeline)
                col1, col2, col3 = st.columns(3)
                col1.metric("cross_validation_accuracy", mean_cross_val_accuracy)
                col2.metric("test_accuracy", test_accuracy)
                with col3:
                        st.write("confusion_matrix")
                        st.dataframe(confusion_matrix)
                st.markdown('---')


    X_train, X_test, y_train, y_test =  Lamp.load_split(train_data)
    mean_cross_val_accuracy, pipeline = Lamp.train(X_train, y_train)
    test_accuracy, cm = Lamp.validate(X_test, y_test, pipeline)
    # Model performance
    st.subheader('Model Performance ')
    st.write('- This model was trained with 80% of the dataset, while 20% was used for validation')
    # model method and accuracy score
    col1, col2, col3 = st.columns(3)
    with col1 :
        st.write('Training model method:')
        st.write('Passive Aggressive Classifier')
    with col2 :
        st.write('Training accuracy score is:')
        st.write(mean_cross_val_accuracy.round(2))
    with col3:
        st.write('Confusion Matrix:')
        st.write(cm)

    # Word Cloud
    st.subheader("Word Frequency in Reviews")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('Word Cloud for Recommended')
        st.image('wc_rec.png')
    with col2:
        st.markdown('Word Cloud for Not Recommended')
        st.image('wc_norec.png')
    st.markdown('---')
    
    
    

elif page_selected == "About":
    st.subheader("About Our Company")
    st.write('Our project uses only customer reviews to perform machine learning to know their true recommendation level – their TrueReview®️.​ We use machine learning to process the reviews, so that rather than aggregating from stars, points, ratings, or any other single metric, we look at what consumers actually said to generate a Consumers have access to product information, reviews, and ratings – but it can be hard to aggregate all these different inputs. ​Our model reads reviews to try and gauge on a simple binary scale whether the item is "recommended" or not.​')
    st.markdown('---')

    st.subheader("About Our Team")
    from PIL import Image
    Lackuna = Image.open('Lackuna.jpg')
    Erica = Image.open('Erica.jpg')
    Spencer = Image.open('Spencer.jfif')
    Mona = Image.open('Mona.jpg')
    Sean = Image.open('Sean.jpg')

    col1, col2, col3, col4, col5 = st.columns((1,1,1, 1, 1))
    with col1:
        st.image(Lackuna)
        st.subheader('Lackuna Alounthong')
        st.write("Lackuna's Bio")
        st.write("Lackuna is originally from Laos. She was working in a research development area for government of Lao PDR and was a Management Trainee for Carlsberg Group (Laos) for about 3+ years, where her interests about data analytics started to begin. Then, Lackuna decided to apply for a Fulbright program to study Master of Business Analytics at Tulane.")
    with col3:
        st.image(Erica)
        st.subheader('Erica Bush')
        st.write("Erica's Bio")
        st.write("Erica grew up on Long Island, New York before attending Tulane University where she recieved her BS in mathematics.  After teaching middle and high school math in the New Orleans area for 3 years, Erica came back to Tulane to work towards receiving her Master's in Business Analytics.")
    with col5:
        st.image(Spencer)
        st.subheader('Spencer Davis')
        st.write("Spencer's Bio")
        st.write("https://www.spencer4hire.xyz/about-me/")
        st.write(""" 
                        ░░░░░░░░░░░░░░░░░░░░░░░░░
                        ░░░░░░░░▄▀▀▄░░░░░░░░░░░░░
                        ░░░░░░▄▀▒▒▒▒▀▄░░░░░░░░░░░
                        ░░░░░░░▀▌▒▒▐▀░░░░░░░░░░░░
                        ░▄███▀░◐░░░▌░░░░░░░░░░░░░░
                        ░░░▐▀▌░░░░░▐░░░░░░░░░░▄▀▀▀▄▄
                        ░░▐░░▐░░░░░▐░░░░░░░░░█░▄█▀
                        ░░▐▄▄▌░░░░░▐▄▄░░░░░░█░░▄▄▀▀▀▀▄
                        ░░░░░░▌░░░░▄▀▒▒▀▀▀▀▄▀░▄▀░▄▄▄▀▀
                        ░░░░▐░░░░▐▒▒▒▒▒▒▒▒▀▀▄░░▀▄▄▄░▄
                        ░░░░▐░░░░▐▄▒▒▒▒▒▒▒▒▒▒▀▄░▄▄▀▀
                        ░░░░░▀▄░░░░▀▄▒▒▒▒▒▒▒▒▒▒▀▄
                        ░░░░░░░▀▄▄░░░█▄▄▄▄▄▄▄▄▄▄▄▀▄
                        ░░░░░░░░░░▀▀▄▄▄▄▄▄▄▄▀▀░
                        ░░░░░░░░░░░░▌▌░▌▌░░░
                        ░░░░░░░░░░▄▄▌▌▄▌▌""")
    
    
    col1, col2, col3, col4, col5 = st.columns((1,1, 1, 1, 1))
    with col2:
        st.image(Mona)
        st.subheader('Jingwen Sang')
        st.write("Mona's Bio")
        st.write("Mona is from China, with Korean ethnicity, and grew up in a multicultural family background. Her undergraduate studied International Business in China, attending a lot of extracurricular activities like Art Troupe, Dance Club. After graduated in 2021, Mona came to Tulane University to continuing her study in business analytics")
    with col4:
        st.image(Sean)
        st.subheader('Chengao Yuan')
        st.write("Sean's Bio")
        st.write("Sean (Chengao Yuan), has been exposed to international trade since college, and as a member of the university next to the FTA, has been working hard to be a part of the future business development. After graduating from college with a degree in Network Engineering, he came to Tulane University to pursue his Master's degree in Business Analytics. And is ready to start his ambition.")



