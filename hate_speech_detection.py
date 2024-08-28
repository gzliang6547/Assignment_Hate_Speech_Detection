# import required library
import re 
import string
import contractions
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.classify import apply_features
from joblib import load
from textblob import TextBlob, Word
from scipy.sparse import hstack
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
nltk.download('stopwords')
nltk.download('punkt') 
nltk.download('wordnet') 

# Load the TF-IDF vectorizer and all hate speech detection model
tfidf_loaded = load('tfidf_vectorizer.joblib')
linear_r_no_polarity_loaded = load('linear_regression_model_no_polarity.joblib')
linear_r_with_polarity_loaded = load('linear_regression_model_with_polarity.joblib')
logistic_r_loaded = load('logistic_regression_model.joblib')
knn_loaded = load('KNN_model.joblib')
svm_loaded = load('SVC_model.joblib')

# Main Func or start of the Web Application
def main():
    # Set Title of the Web
    st.title(":rainbow[Hate Speech Detection Web App]")
    # Sidebar for navigation
    st.sidebar.title("Input Options")
    option = st.sidebar.selectbox("Choose Method To Input Text Data/Comments", ["Manually Enter Text", "Upload File"])

    hate_speech_score_type = pd.DataFrame({
        'Range of Hate Speech Score ' : ['hate speech score > 0.5','-1 <= hate speech score <= 0.5','hate speech score < -1'],
        'Type of Text/Comment ' : ['hate speech','neutral speech or ambiguous','non-hate speech or supportive speech']
    })  

    st.table(hate_speech_score_type)
    
    # Option to manually enter text
    if option == "Manually Enter Text":       
        # Text box for user input
        st.subheader(":orange[Enter a sentence to check it's hate speech score and determine if it's hate speech or not]\n(Higher Hate Speech Score = More Hateful)")    
        user_input = st.text_input("Your Sentence:")

        # Predict button
        if st.button('Predict'):
            if user_input:  # Check if the input is not empty
                processed_user_input = preprocess_and_clean([user_input])
                predict_and_display([user_input],processed_user_input)  # Single sentence prediction
            else:
                st.error("Please enter a sentence for prediction.")
    else:  # Option to upload file
        st.subheader(":green[Please select a text(.txt) or a csv(.csv) file to upload and check the hate speech score]")
        uploaded_file = st.file_uploader("Choose a file to upload", type=['txt', 'csv'])
        if uploaded_file is not None:
            if uploaded_file.type == "text/csv" or uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:  # Assume text file
                data = pd.read_table(uploaded_file, header=None, names=['text'])

            # Check if the file has content
            if not data.empty:
                sentences = data['text'].tolist()
                processed_sentences = preprocess_and_clean(sentences)
                predict_and_display(sentences,processed_sentences)  # File-based prediction

def preprocess_and_clean(sentences):
    #remove any links or url (e.g. https://123abc.com]
    sentences_df = pd.DataFrame(sentences,columns=["Sentences"])
    sentences_df['Sentences'] = sentences_df['Sentences'].apply(lambda x: re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|''[!*,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', x))
    
    #remove punctuation(except apostrophes[']) and change all text to lowercase 
    my_punctuation = string.punctuation.replace("'", "")
    sentences_df['Sentences'] = sentences_df['Sentences'].apply(lambda x: re.sub('[%s]' % re.escape(my_punctuation), ' ', x.lower())) 
    
    #remove contractions (e.g remove We're and change to We are)
    sentences_df['Sentences'] = sentences_df['Sentences'].apply(lambda x: ' '.join([contractions.fix(word) for word in x.split()]))
    
    #remove apostrophe that are still remained after removing contractions
    sentences_df['Sentences'] = sentences_df['Sentences'].apply(lambda x: x.replace("'","")) 
    
    #remove alphanumeric
    sentences_df['Sentences'] = sentences_df['Sentences'].apply(lambda x: re.sub(r"""\w*\d\w*""", ' ', x)) 
    
    #change multiple space characters between words into one space character only
    sentences_df['Sentences'] = sentences_df['Sentences'].apply(lambda x: re.sub(r'\s+', ' ', x))
    
    #remove leading and trailing whitespace character
    sentences_df['Sentences'] = sentences_df['Sentences'].apply(lambda x: re.sub(r'^\s+|\s+?$','', x))

    #create stopword object
    stop = stopwords.words('english')
    #remove stopwords
    sentences_df['Sentences'] = sentences_df['Sentences'].apply(lambda x : ' '.join([word for word in x.split() if word not in (stop)]))
    
    # #create lemmatizer object
    # lemmatizer = nltk.WordNetLemmatizer()
    # #lemmatize each word
    # sentences_df['Sentences'] = sentences_df['Sentences'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    sentences_df['Sentences'] = sentences_df['Sentences'].apply(lambda x: ' '.join([w.lemmatize() for word in TextBlob(x).words]))
    
    # create stemming object
    stemmer = LancasterStemmer()
    # perform stemming on each word
    sentences_df['Sentences'] = sentences_df['Sentences'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()])) 

    return sentences_df["Sentences"].tolist()

def predict_and_display(unprocessed_sentences,sentences):
    # Transform the sentences
    transformed_sentences = tfidf_loaded.transform(sentences)

    # use textblob library to determine polarity score and transform the sentence
    sentences_df = pd.DataFrame(sentences)
    polarity_score = sentences_df.iloc[:, 0].apply(lambda x: TextBlob(x).sentiment.polarity)
    transformed_sentences_with_polarity = hstack([transformed_sentences, polarity_score.values.reshape(-1, 1)]) 
    
    # Make predictions
    score_results_no_polarity = linear_r_no_polarity_loaded.predict(transformed_sentences)
    text_type_no_polarity = []
    for x in score_results_no_polarity:
        if x > 1 :
            text_type_no_polarity.append("hate speech")
        elif x > -1 :
            text_type_no_polarity.append("neutral speech or ambiguous")
        else :
            text_type_no_polarity.append("non-hate speech or supportive speech")
    
    score_results_with_polarity = linear_r_with_polarity_loaded.predict(transformed_sentences_with_polarity)
    text_type_with_polarity = []
    for x in score_results_with_polarity:
        if x > 1 :
            text_type_with_polarity.append("hate speech")
        elif x > -1 :
            text_type_with_polarity.append("neutral speech or ambiguous")
        else :
            text_type_with_polarity.append("non-hate speech or supportive speech")
            
    logistic_r_target_results = logistic_r_loaded.predict(transformed_sentences)
    knn_target_results = knn_loaded.predict(transformed_sentences)
    svm_target_results = svm_loaded.predict(transformed_sentences)

    # Combine the inputs and predictions into a DataFrame
    score_results_no_polarity_df = pd.DataFrame({
        'Original Input': unprocessed_sentences,
        'Processed Input': sentences,
        'Predicted Hate Speech Score': score_results_no_polarity,
        'Type Or Category Of Input Text' : text_type_no_polarity
    })

    # Tabulate and display the results
    with st.expander("Show/Hide Prediction Table (Result With Hate Speech Score Only)"):
        st.table(score_results_no_polarity_df)

    # Combine the inputs and predictions into a DataFrame
    score_results_with_polarity_df = pd.DataFrame({
        'Original Input': unprocessed_sentences,
        'Processed Input': sentences,
        'Polarity Score' : polarity_score,
        'Predicted Hate Speech Score': score_results_with_polarity,
        'Type Or Category Of Input Text' : text_type_with_polarity
    })

    # Tabulate and display the results
    with st.expander("Show/Hide Prediction Table (Result With Polarity Score And Hate Speech Score)"):
        st.table(score_results_with_polarity_df)

    logisitic_r_target_results_df = pd.DataFrame({
        'Target Race': logistic_r_target_results[:,0],
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
        'Target Race': knn_target_results[:,0],
        'Target Religion': knn_target_results[:,1],
        'Target Origin': knn_target_results[:,2],
        'Target Gender': knn_target_results[:,3],
        'Target Sexuality': knn_target_results[:,4],
        'Target Age': knn_target_results[:,5],
        'Target Disability': knn_target_results[:,6]
    })

    with st.expander("Show/Hide Prediction Table"):
        st.table(knn_target_results_df)

    svm_target_results_df = pd.DataFrame({
        'Target Race': svm_target_results[:,0],
        'Target Religion': svm_target_results[:,1],
        'Target Origin': svm_target_results[:,2],
        'Target Gender': svm_target_results[:,3],
        'Target Sexuality': svm_target_results[:,4],
        'Target Age': svm_target_results[:,5],
        'Target Disability': svm_target_results[:,6]
    })

    with st.expander("Show/Hide Prediction Table"):
        st.table(svm_target_results_df)

    # Label for x-axis of bar chart
    x = np.array(["Race", "Religion", "Origin", "Gender", "Sexuality", "Age", "Disability"])

    #------------ Logistic Regression Model ------------
    # Convert result to dataframe
    logistic_r_result_df = pd.DataFrame(logistic_r_target_results)
    logistic_r_result_y = np.array([len(logistic_r_result_df[logistic_r_result_df[0]==True]),len(logistic_r_result_df[logistic_r_result_df[1]==True]),len(logistic_r_result_df[logistic_r_result_df[2]==True]),len(logistic_r_result_df[logistic_r_result_df[3]==True]),len(logistic_r_result_df[logistic_r_result_df[4]==True]),len(logistic_r_result_df[logistic_r_result_df[5]==True]),len(logistic_r_result_df[logistic_r_result_df[6]==True])])
    
    # Display histogram of predictions
    st.write("Bar Chart Of Distribution Of Prediction:")
    fig, ax = plt.subplots()
    ax.bar(x,logistic_r_result_y)
    ax.set_title("Number of Hate Speech Predictions")
    ax.set_xlabel("Target Type")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Ensure y-axis has integer ticks
    st.pyplot(fig)

    #------------ KNN Model ------------
    # Convert result to dataframe
    knn_target_results_df = pd.DataFrame(knn_target_results)
    knn_target_results_y = np.array([len(knn_target_results_df[knn_target_results_df[0]==True]),len(knn_target_results_df[knn_target_results_df[1]==True]),len(knn_target_results_df[knn_target_results_df[2]==True]),len(knn_target_results_df[knn_target_results_df[3]==True]),len(knn_target_results_df[knn_target_results_df[4]==True]),len(knn_target_results_df[knn_target_results_df[5]==True]),len(knn_target_results_df[knn_target_results_df[6]==True])])
    
    # Display histogram of predictions
    st.write("Bar Chart Of Distribution Of Prediction:")
    fig, ax = plt.subplots()
    ax.bar(x,knn_target_results_y)
    ax.set_title("Number of Hate Speech Predictions")
    ax.set_xlabel("Target Type")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Ensure y-axis has integer ticks
    st.pyplot(fig)

    #------------ SVC Model ------------
    # Convert result to dataframe
    svm_target_results_df = pd.DataFrame(svm_target_results)
    svm_target_results_y = np.array([len(svm_target_results_df[svm_target_results_df[0]==True]),len(svm_target_results_df[svm_target_results_df[1]==True]),len(svm_target_results_df[svm_target_results_df[2]==True]),len(svm_target_results_df[svm_target_results_df[3]==True]),len(svm_target_results_df[svm_target_results_df[4]==True]),len(svm_target_results_df[svm_target_results_df[5]==True]),len(svm_target_results_df[svm_target_results_df[6]==True])])
    
    # Display histogram of predictions
    st.write("Bar Chart Of Distribution Of Prediction:")
    fig, ax = plt.subplots()
    ax.bar(x,svm_target_results_y)
    ax.set_title("Number of Hate Speech Predictions")
    ax.set_xlabel("Target Type")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Ensure y-axis has integer ticks
    st.pyplot(fig)


if __name__ == '__main__':
    main()
