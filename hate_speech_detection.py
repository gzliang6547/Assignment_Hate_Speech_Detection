# import required library
import streamlit as st
import nltk
from nltk import NaiveBayesClassifier
from nltk.classify import apply_features
from joblib import load
from joblib import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load the TF-IDF vectorizer and all hate speech detection model
tfidf_loaded = load('tfidf_vectorizer.joblib')
linear_r_loaded = load('linear_regression_model.joblib')
logistic_r_loaded = load('logistic_regression_model.joblib')
knn_loaded = load('KNN_model.joblib')
svm_loaded = load('SVC_model.joblib')

# Main Func or start of the Web Application
def main():
    # Set Title of the Web
    st.title("Hate Speech Detection App")

    # Sidebar for navigation
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose how to input data", ["Enter text", "Upload file"])

    # Option to manually enter text
    if option == "Enter text":
        # Text box for user input
        user_input = st.text_input("Enter a sentence to check if it's hate speech or not:")

        # Predict button
        if st.button('Predict'):
            if user_input:  # Check if the input is not empty
                predict_and_display([user_input])  # Single sentence prediction
            else:
                st.error("Please enter a sentence for prediction.")
    else:  # Option to upload file
        uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file is not None:
            if uploaded_file.type == "text/csv" or uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:  # Assume text file
                data = pd.read_table(uploaded_file, header=None, names=['text'])

            # Check if the file has content
            if not data.empty:
                sentences = data['text'].tolist()
                predict_and_display(sentences)  # File-based prediction

def predict_and_display(sentences):
    # Transform the sentences
    transformed_sentences = tfidf_loaded.transform(sentences)

    # Make predictions
    score_results = linear_r_loaded.predict(transformed_sentences)
    logistic_r_target_results = linear_r_loaded.predict(transformed_sentences)
    knn_target_results = knn_loaded.predict(transformed_sentences)
    svm_target_results = svm_loaded.predict(transformed_sentences)

    # Combine the inputs and predictions into a DataFrame
    score_results_df = pd.DataFrame({
        'Input': sentences,
        'Predicted Hate Speech Score': score_results
    })

    # Tabulate and display the results
    with st.expander("Show/Hide Prediction Table"):
        st.table(score_results_df)

    logisitic_r_target_results_df = pd.DataFrame({
        'Target Race': [logistic_r_target_results[:,0]],
        'Target Religion': logistic_r_target_results[:,1],
        'Target Origin': logistic_r_target_results[:,2],
        'Target Gender': logistic_r_target_results[:,3],
        'Target Sexuality': logistic_r_target_results[:,4],
        'Target Age': logistic_r_target_results[:,5],
        'Target Disability': logistic_r_target_results[:,6]
    })

    with st.expander("Show/Hide Prediction Table"):
        st.table(logisitic_r_target_results_df)

    knn_target_results_df = pd.DataFrame({
        'Target Race': logistic_r_target_results[:,0],
        'Target Religion': logistic_r_target_results[:,1],
        'Target Origin': logistic_r_target_results[:,2],
        'Target Gender': logistic_r_target_results[:,3],
        'Target Sexuality': logistic_r_target_results[:,4],
        'Target Age': logistic_r_target_results[:,5],
        'Target Disability': logistic_r_target_results[:,6]
    })

    with st.expander("Show/Hide Prediction Table"):
        st.table(knn_target_results_df)

    svm_target_results_df = pd.DataFrame({
        'Target Race': logistic_r_target_results[:,0],
        'Target Religion': logistic_r_target_results[:,1],
        'Target Origin': logistic_r_target_results[:,2],
        'Target Gender': logistic_r_target_results[:,3],
        'Target Sexuality': logistic_r_target_results[:,4],
        'Target Age': logistic_r_target_results[:,5],
        'Target Disability': logistic_r_target_results[:,6]
    })

    with st.expander("Show/Hide Prediction Table"):
        st.table(svm_target_results_df)
    

    # # Display histogram of predictions
    # st.write("Bar Chart Of Distribution Of Prediction:")
    # fig, ax = plt.subplots()
    # prediction_counts = pd.Series(results).value_counts().sort_index()
    # prediction_counts.plot(kind='bar', ax=ax)
    # ax.set_title("Number of Hate Speech Predictions")
    # ax.set_xlabel("Category")
    # ax.set_ylabel("Count")
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Ensure y-axis has integer ticks
    # st.pyplot(fig)
if __name__ == '__main__':
    main()
