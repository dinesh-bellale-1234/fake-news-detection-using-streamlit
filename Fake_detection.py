import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

@st.cache_data
def load_data():
    data = pd.read_csv(r"C:\Users\deepa\python & ML documents\Machine learning Notes\ML Assignments\Multiple CSV\archive\fakenews.csv")
    return data

def about_dataset():
    st.title("About the Dataset")
    st.write("""
    The dataset is a collection of news articles labeled as either real or fake. It's commonly used for research and development of fake news detection algorithms and models. The dataset typically includes features such as the title, text, and metadata of the news articles, along with the corresponding labels indicating whether the news is real or fake.
    
    Analyzing this dataset involves natural language processing (NLP) techniques to extract features from the text data and machine learning algorithms to classify the news articles as real or fake. Researchers and data scientists often use this dataset to train and evaluate the performance of their fake news detection models.
    
    It's important to note that while the dataset provides a valuable resource for studying fake news, the definition of "fake news" can be subjective, and the labels assigned to the articles may not always be definitive. Therefore, careful preprocessing and analysis are necessary to ensure the validity and reliability of the findings.
    """)
    st.subheader("Problem Statement:")
    st.write("Develop a machine learning model to classify news articles as either real or fake based on their content and data.")

def check(data, column):
    lower = " ".join(data[column]).islower()
    html = data[column].apply(lambda x: True if re.search("<.*?>", x) else False).sum()
    url = data[column].apply(lambda x: True if re.search("http[s]?://.+?\S+", x) else False).sum()
    tags = data[column].apply(lambda x: True if re.search("#\S+", x) else False).sum()
    mention = data[column].apply(lambda x: True if re.search("@\S+", x) else False).sum()
    un_w = data[column].apply(lambda x: True if re.search("[\.\*'\-#$%^&)(0-9]", x) else False).sum()
    emojis = data[column].apply(lambda x: True if emoji.emoji_count(x) > 0 else False).sum()

    if not lower:
        st.write("Your data contains both cases.")
    if html > 0:
        st.write("Your data contains HTML tags:", html)
    if url > 0:
        st.write("Your data contains URLs:", url)
    if tags > 0:
        st.write("Your data contains tags:", tags)
    if mention > 0:
        st.write("Your data contains mentions:", mention)
    if un_w > 0:
        st.write("Your data contains unwanted characters:", un_w)
    if emojis > 0:
        st.write("Your data contains emojis:", emojis)

def basic_eda(data):
    st.title("Basic EDA")

    st.write(data.head())
    st.write(data.iloc[:,1].value_counts()) 

    st.write("Text analysis:")
    check(data,"text") 

def preprocessing(data):
    st.title("Preprocessing")

    stop_words = set(stopwords.words('english'))

    def preprocess_text(x, emoj="F"):
        x = x.lower()  # Convert to lowercase
        x = re.sub("<.*?>", " ", x)  # Remove HTML tags
        x = re.sub("http[s]?://.+?\S+", " ", x)  # Remove URLs
        x = re.sub("#\S+", " ", x)  # Remove hashtags
        x = re.sub("@\S+", " ", x)  # Remove mentions
        if emoj == "T":
            x = emoji.demojize(x)  # Convert emojis to text
        x = re.sub(r"(?<= )[\[\]:]|[,;”“/~!.'’\"\\*$%^&)(0-9?]|—", " ", x)
        x = re.sub(r"\s+-\s+", " ", x)
        x = re.sub(r"-", "", x) 
        x = re.sub(r"[\[\]:]", "", x) # Remove all unwanted characters
        
        tokens = word_tokenize(x)
        filtered_tokens = [word for word in tokens if word not in stop_words]
        x = ' '.join(filtered_tokens)
        return x

    preprocessed_text= data["text"].apply(preprocess_text, args=("T",)) 
    preprocessed_data = pd.concat([preprocessed_text, data.iloc[:, 1]], axis=1) 
    st.session_state["Preprocessed_Data"] = preprocessed_data
    st.write(preprocessed_data)

def advanced_eda(data):
    st.title("Advanced EDA")

    # Word Frequency Analysis
    st.subheader("Word Frequency Analysis")
    tokens = word_tokenize(' '.join(data["text"]))
    freq_dist = nltk.FreqDist(tokens)
    plt.figure(figsize=(6, 3))
    freq_dist.plot(20, cumulative=False)
    plt.title("Word Frequency Analysis")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    st.pyplot(plt)
    
    # Word Clouds
    st.subheader("Word Clouds")
    
    # Word Cloud for Real News
    real_wordcloud = WordCloud(width=400, height=400, background_color='black').generate(' '.join(data[data['label'] == 0]['text']))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(real_wordcloud, interpolation='bilinear')
    plt.title("Real News Word Cloud")
    plt.axis('off')

    # Word Cloud for Fake News
    fake_wordcloud = WordCloud(width=400, height=400, background_color='black').generate(' '.join(data[data['label'] == 1]['text']))
    plt.subplot(1, 2, 2)
    plt.imshow(fake_wordcloud, interpolation='bilinear')
    plt.title("Fake News Word Cloud")
    plt.axis('off')
    st.pyplot(plt)



def model_evaluation(data):
    st.title("Model Evaluation")
    option = st.sidebar.radio("Select an option:", ["Bernoulli Naive Bayes", "Multinomial Naive Bayes", "K-Nearest Neighbors"])
    
    if option == "Bernoulli Naive Bayes":
        # Bernoulli Naive Bayes
        fv = data.iloc[:, 0]
        cv = data.iloc[:, 1]
        x_train, x_test, y_train, y_test = train_test_split(fv, cv, test_size=0.2, random_state=10, stratify=cv)

        cv1 = CountVectorizer(binary=True)
        x_trainf_1 = cv1.fit_transform(x_train)
        x_testf_1 = cv1.transform(x_test)

        alpha = []
        final_f1 = []
        for i in np.arange(0.1, 5.0, 0.1):
            final_f1.append(np.mean(cross_val_score(BernoulliNB(alpha=i), x_trainf_1, y_train, scoring="f1", cv=4)))
            alpha.append(i)

        st.write("### Alpha vs F1_Score")
        chart_data = pd.DataFrame({"Alpha": alpha, "Final F1 Score": final_f1})
        st.line_chart(chart_data.set_index("Alpha"))

        bnb = BernoulliNB(alpha=0.1)
        model = bnb.fit(x_trainf_1, y_train)
        predicted = np.argmax(model.predict_log_proba(x_testf_1), axis=1)
        report = classification_report(y_test, predicted)
        st.text_area(label="Classification Report:", value=report, height=300)

    elif option == "Multinomial Naive Bayes":
        # Multinomial Naive Bayes
        fv = data.iloc[:, 0]
        cv = data.iloc[:, 1]
        x_train, x_test, y_train, y_test = train_test_split(fv, cv, test_size=0.2, random_state=10, stratify=cv)

        cv = CountVectorizer()
        x_trainf = cv.fit_transform(x_train)
        x_testf = cv.transform(x_test)

        alpha = []
        final_f1 = []
        for i in np.arange(0.1, 5.0, 0.1):
            final_f1.append(np.mean(cross_val_score(MultinomialNB(alpha=i), x_trainf, y_train, scoring="f1", cv=4)))
            alpha.append(i)

        st.write("### Alpha vs F1_Score")
        chart_data = pd.DataFrame({"Alpha": alpha, "Final F1 Score": final_f1})
        st.line_chart(chart_data.set_index("Alpha"))

        mnb = MultinomialNB(alpha=0.8)
        model = mnb.fit(x_trainf, y_train)
        predicted = np.argmax(model.predict_log_proba(x_testf), axis=1)
        report = classification_report(y_test, predicted)
        st.text_area(label="Classification Report:", value=report, height=300)

    elif option == "K-Nearest Neighbors":
        sub_option = st.sidebar.radio("Select an option:", ["Bag of Words", "Binary Bag of Words"])
        
        fv = data.iloc[:, 0]
        cv = data.iloc[:, 1]
        x_train, x_test, y_train, y_test = train_test_split(fv, cv, test_size=0.2, random_state=10, stratify=cv)  

        if sub_option == "Bag of Words":
            # Bag of Words
            cv = CountVectorizer()
            x_trainf_1 = cv.fit_transform(x_train)
            x_testf_1 = cv.transform(x_test)

            k = []
            final_f1 = []
            for i in range(1, 65, 2):
                final_f1.append(np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=i), x_trainf_1, y_train, scoring="f1", cv=4)))
                k.append(i)
            
            st.write("### K vs F1_Score for Bag of Words")
            chart_data = pd.DataFrame({"K": k, "Final F1 Score": final_f1})
            st.line_chart(chart_data.set_index("K"))

            knn = KNeighborsClassifier(n_neighbors=21)
            model = knn.fit(x_trainf_1, y_train)
            predicted = model.predict(x_testf_1)
            report = classification_report(y_test, predicted)
            st.text_area(label="Classification Report for Bag of Words:", value=report, height=300)

        elif sub_option == "Binary Bag of Words":
            # Binary Bag of Words
            cv1 = CountVectorizer(binary=True)
            x_trainf_1 = cv1.fit_transform(x_train)
            x_testf_1 = cv1.transform(x_test)

            k = []
            final_f1 = []
            for i in range(1, 65, 2):
                final_f1.append(np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=i), x_trainf_1, y_train, scoring="f1", cv=4)))
                k.append(i)
            
            st.write("### K vs F1_Score for Binary Bag of Words")
            chart_data = pd.DataFrame({"K": k, "Final F1 Score": final_f1})
            st.line_chart(chart_data.set_index("K"))

            knn = KNeighborsClassifier(n_neighbors=21)
            model = knn.fit(x_trainf_1, y_train)
            predicted = model.predict(x_testf_1)
            report = classification_report(y_test, predicted)
            st.text_area(label="Classification Report for Binary Bag of Words:", value=report, height=300)

def predict_text(text):
    preprocessed_data = st.session_state["Preprocessed_Data"]
    fv = preprocessed_data.iloc[:, 0]
    cv = preprocessed_data.iloc[:, 1]
    cv1 = CountVectorizer(binary=True)
    x_trainf_1 = cv1.fit_transform(fv)
    x_testf_1 = cv1.transform([text])
    bnb = BernoulliNB(alpha=0.1)
    model = bnb.fit(x_trainf_1, cv)
    prediction = np.argmax(model.predict_log_proba(x_testf_1), axis=1)
    return "Real" if prediction[0] == 0 else "Fake"

selected_option = st.sidebar.radio("Select an option:", ["About the Dataset", "Basic EDA", "Preprocessing", "Advanced EDA","Model Evaluation", "Predict"])

data = load_data()

if selected_option == "About the Dataset":
    about_dataset()
elif selected_option == "Basic EDA":
    basic_eda(data)
elif selected_option == "Preprocessing":
    preprocessing(data) 
elif selected_option == "Advanced EDA":
    if "Preprocessed_Data" not in st.session_state:
        st.warning("Please perform preprocessing first.")
    else:
        preprocessed_data = st.session_state["Preprocessed_Data"]
        advanced_eda(preprocessed_data)
elif selected_option == "Model Evaluation":
    if "Preprocessed_Data" not in st.session_state:
        st.warning("Please perform preprocessing first.")
    else:
        preprocessed_data = st.session_state["Preprocessed_Data"]
        model_evaluation(preprocessed_data)
elif selected_option == "Predict":
    st.title("Predict")
    user_input = st.text_input("Enter a text:", "")
    if st.button("Predict"):
        prediction = predict_text(user_input)
        st.write("Prediction:", prediction)