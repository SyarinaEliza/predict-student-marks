import streamlit as st
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Marks Prediction",page_icon="📜",layout="centered")

st.title(" Students Marks Prediction")
st.write("Enter The Number Of Hours Studied (1-10) And **Click Predict** To See The Predicted Marks")

# Load The Model 

def load_model(model):
    with open(model,"rb") as f:
        slr = pickle.load(f)
    return slr

try:
    model = load_model("slr.pkl")
except Exception as e:
    st.error("your pickle file not found.....")
    st.exception("Failed To Load The Model :",e)
    st.stop()

hours = st.number_input("Hours_Studied",
                        min_value=1.0,
                        max_value=10.0,
                        value=4.0,
                        step=0.1,
                        format="%.f")

if st.button("Predict"):
    try:
        X = np.array([[hours]])
        predictions = model.predict(X)
        predictions = predictions[0]

        st.success(f"Predicted Marks : {predictions:.1f}")
        st.write("Note : This Is Ml Model Prediction **Result May Vary**")
    except Exception as e:
        st.error(f"Prediction Failed : {e}")


