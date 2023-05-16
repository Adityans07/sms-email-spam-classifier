import streamlit as st
import pickle
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
Ps = PorterStemmer()


def transform_text(Text):
    Text = Text.lower()
    Text = word_tokenize(Text)
    carry = []
    for i in Text:
        if i.isalnum():
            carry.append(i)
    y = []
    for i in carry:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
    carry.clear()
    carry = []
    for i in y:
        carry.append(Ps.stem(i))
    y.clear()
    return " ".join(carry)

tfidf = pickle.load(open('vectorize.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS spam classifier")

input_sms = st.text_input("Enter the message")
if st.button("Predict"):
    transform_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transform_sms])
    result = model.predict(vector_input)[0]
    if result==1:
        st.header("Spam")
    else:
        st.header("Not spam")
