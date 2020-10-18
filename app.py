import streamlit as st
import pickle
import numpy as np
model=pickle.load(open('model.pkl','rb'))


def predict_forest(chlorides,alcohol):
    input=np.array([[chlorides,alcohol]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0],2)
    return float(pred)

def main():
    st.title("Streamlit Tutorial")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">wine quality Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    
    chlorides = st.text_input("chlorides","Type Here")
    alcohol = st.text_input("alcohol","Type Here")
    safe_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> Your wine is good</h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> Your wine is in bad</h2>
       </div>
    """

    if st.button("Predict"):
        output=predict_forest(chlorides,alcohol)
        st.success('The probability of wine being bad is {}'.format(output))

        if output > 0.5:
            st.markdown(safe_html,unsafe_allow_html=True)
        else:
            st.markdown(danger_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()