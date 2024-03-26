import streamlit as st
import pickle
from PIL import Image

def load_model_and_vectorizer():
    model = pickle.load(open('model_tweet.sav', 'rb'))
    vectorizer = pickle.load(open('vector_tweet.sav', 'rb'))
    return model, vectorizer


def predict_tweet(text, model, vectorizer):
    feature = [text]
    vectorized_text = vectorizer.transform(feature)
    prediction = model.predict(vectorized_text)
    return prediction


def main():
    st.markdown("<h1 style='text-align: center;'><span style='vertical-align: middle;'>📩</span> Tweet Analysis App</h1>", unsafe_allow_html=True)
    image=Image.open('Twitter-analytics.jpg')
    st.image(image,width=800)
    # Text input field for user input
    text = st.text_input('Text', placeholder='Type here')

    # Button to trigger prediction
    pred_button = st.button('PREDICT')

    if pred_button:
        # Load model and vectorizer
        model, vectorizer = load_model_and_vectorizer()

        # Perform prediction
        if text.strip() == "":
            st.write("Please enter some text.")
        else:
            prediction = predict_tweet(text, model, vectorizer)
            if prediction == -1:
                st.markdown("<p style='text-align: center; color: red; font-size: 24px;'>Negative 😞</p>", unsafe_allow_html=True)
            elif prediction == 0:
                st.markdown("<p style='text-align: center; color: orange; font-size: 24px;'>Neutral 😐</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='text-align: center; color: green; font-size: 24px;'>Positive 😊</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
