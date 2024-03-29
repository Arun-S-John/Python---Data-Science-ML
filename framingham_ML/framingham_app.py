import pickle
from PIL import Image

def main():
    st.title("FRAMINGHAM CHRONIC HEART DISEASE PREDICTION")
    image=Image.open("framinghamimg.jpeg")
    st.image(image,width=800)
    gender=st.radio("Gender",["Male","Female"])
    if gender == "Male":
        male=1
    else:
        male=0
    age=st.text_input("Age","Type Here")
    cs=st.radio("Current Smoker",["Yes","No"])
    if cs == "Yes":
        currentSmoker=1
    else:
        currentSmoker=0
    cpd = st.text_input("Cigs per Day", "Type Here")
    bpm = st.radio("BPMeds", ["Yes", "No"])
    if bpm == "Yes":
        bpmeds = 1
    else:
        bpmeds = 0
    ps = st.radio("Prevalent Stroke", ["Yes", "No"])
    if ps == "Yes":
        prevalentStroke = 1
    else:
        prevalentStroke = 0
    ph = st.radio("Prevalent Hypertension", ["Yes", "No"])
    if ph == "Yes":
        prevalentHyp = 1
    else:
        prevalentHyp = 0
    db = st.radio("Diabetes",["Yes","No"])
    if db == "Yes":
        diabetes=1
    else:
        diabetes=0
    tc = st.text_input("Total Cholestrol", "Type Here")
    sbp = st.text_input("Systolic BP", "Type Here")
    dbp = st.text_input("Diastolic BP", "Type Here")
    bmi = st.text_input("BMI", "Type Here")
    hr = st.text_input("Heart Rate", "Type Here")
    gc = st.text_input("Glucose", "Type Here")
    features=[male,age,currentSmoker,cpd,bpmeds,prevalentStroke,prevalentHyp,diabetes,tc,sbp,dbp,bmi,hr,gc]
    model = pickle.load(open('model_framingham_1.sav', 'rb'))
    scaler = pickle.load(open('scaler_framingham_1.sav', 'rb'))
    pred = st.button('PREDICT')
    if pred:
        prediction = model.predict(scaler.transform([features]))
        if prediction == 0:
            st.write("The person will not suffer from Chronic Heart Disease in ten years")
        else:
            st.write("The person will suffer from Chronic Heart Disease in ten years")
main()
