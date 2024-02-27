import streamlit as st
import pickle
from PIL import Image

def main():
    st.title(":rainbow[BANK CUSTOMER CHURN PREDICTION] :bank:")
    image=Image.open('bankimg.jpg')
    st.image(image, width=800)
    cs=st.slider('Credit Score',min_value=350,max_value=850,step=1)
    geo=st.radio('Geography',['France','Germany','Spain'])
    if geo == 'France':
        geography=0
    elif geo == 'Germany':
        geography=1
    elif geo == 'Spain':
        geography=2
    sex=st.radio('Gender',['Male','Female'])
    if sex == 'Male':
        gender=1
    else:
        gender=0
    age=st.text_input('Age',placeholder='Type here')
    tenure=st.text_input('Tenure',placeholder='Type here')
    balance=st.text_input('Balance',placeholder='Type here')
    nop=st.selectbox('Number of Products',[1,2,3,4])
    cc=st.radio('Has Credit Card?',['Yes','No'])
    if cc == 'Yes':
        hcc=1
    else:
        hcc=0
    am = st.radio('Is active member?', ['Yes', 'No'])
    if am == 'Yes':
        iam = 1
    else:
        iam = 0
    sal=st.text_input('Estimated Salary',placeholder='Type here')
    features=[cs,geography,gender,age,tenure,balance,nop,hcc,iam,sal]
    model=pickle.load(open('model_bank.sav','rb'))
    scaler=pickle.load(open('scaler_bank.sav','rb'))
    pred = st.button("PREDICT")
    if pred:
        prediction = model.predict(scaler.transform([features]))
        if prediction == 0:
            st.write('The customer will not exit the bank')
        else:
            st.write('The customer will exit the bank')
main()
