import streamlit as st
from PIL import Image
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2

model = load_model('model2/v4_1_pred_cott_dis.h5')

CLASS_NAMES = ['diseased cotton leaf','diseased cotton plant','fresh cotton leaf','fresh cotton plant']

st.set_page_config(page_title="Cotton Plant Disease Prediction",
                   layout='wide',
                   page_icon='./icons/object.png')

st.header('Get Cotton Plant Disease Prediction for any Image')
st.write('Please Upload Image to get Prediction')

        
#st.balloons()

def upload_image():
    # Upload Image
    image_file = st.file_uploader(label='Upload Image')
    if image_file is not None:
        size_mb = image_file.size/(1024**2)
        file_details = {"filename":image_file.name,
                        "filetype":image_file.type,
                        "filesize": "{:,.2f} MB".format(size_mb)}
        #st.json(file_details)
        # validate file
        if file_details['filetype'] in ('image/png','image/jpeg'):
            st.success('VALID IMAGE file type (png or jpeg')
            return {"file":image_file,
                    "details":file_details}
        
        else:
            st.error('INVALID Image file type')
            st.error('Upload only png,jpg, jpeg')
            return None
        
def main():
    object = upload_image()
    
    if object:
        #prediction = False
        image_obj = Image.open(object['file'])       
        
        col1 , col2 = st.columns(2)
        
        with col1:
            st.info('Preview of Image')
            st.image(image_obj) #Displays the images
            
        with col2:
            st.subheader('Check below for file details')
            st.json(object['details'])
            button = st.button('Get prediction from APP')
            if button:
                with st.spinner("""
                Checking Cotton Plant. please wait
                                """):
                    # below command will convert
                    # obj to array
                    #image_array = np.array(image_obj)
                    #pred_img = yolo.predictions(image_array)
                    #pred_img_obj = Image.fromarray(pred_img)
                    #prediction = True
                    
                    image_obj = image_obj.resize((150, 150))  # Resize the image
                    test_image = np.array(image_obj) / 255.0  # Convert the image to a numpy array and normalize
                    test_image = np.expand_dims(test_image, axis=0)  # Add an extra dimension to represent the batch size
                    result = model.predict(test_image).round(3)  # Make a prediction on the image

                    pred_img_obj = Image.fromarray(result)
                    
                    pred = np.argmax(result) # get the index of max value
            
                    if pred == 0:
                        pred = CLASS_NAMES[0]
                    elif pred == 1:
                        pred = CLASS_NAMES[1]
                    elif pred == 2:
                        pred = CLASS_NAMES[2]
                    else:
                        pred = CLASS_NAMES[3]
                    
                    st.success("Result")
                    #st.json(pred)
                    st.header(pred)
                    
                
        
    
    
    
if __name__ == "__main__":
    main()