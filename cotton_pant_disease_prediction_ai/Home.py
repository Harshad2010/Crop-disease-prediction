import streamlit as st 

st.set_page_config(page_title="Home", 
                   layout="wide",
                   page_icon="./images/home.png")
st.title("Get Cotton Plant Disease Prediction")
st.caption("This web application demostrate Object Detection")

#contents
st.markdown("""
### This App detects objects from images
- Automatically detects 4 predictions from the images
- [Click here for the app](./1_PRED_image/)

Below given are the objects that our model will detect
1. diseased cotton leaf         
2. diseased cotton plant            
3. fresh cotton leaf          
4. fresh cotton plant                       
 
            """)




















