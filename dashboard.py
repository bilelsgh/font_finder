from xml.etree.ElementPath import prepare_predicate
import streamlit as st
from dotenv import load_dotenv
import os 
from model import Model
from PIL import Image
import numpy as np
from utils.generate_dataset import text_to_image


# Init
load_dotenv(".env")
DATASET = os.getenv("dataset")

# Init Model
model = Model(f"{DATASET}/split_dataset")
model.load_model()

st.title("Font finder")
pred, gen = st.tabs(["Predict", "Generate"])


# Options
with st.sidebar:
    if st.button("Get model accuracy"):
        st.write( model.eval() )


# Predict
with pred:
    st.header("What's the font ?")
    uploaded_image = st.file_uploader("Choose a file")
    if uploaded_image:
        image = Image.open(uploaded_image).convert('L')
        img_array = np.array( [np.array(image)] )
        font = model.predict(img_array)
        st.success("I'm already done ! B-)")
        
        st.markdown("""---""")

        st.markdown(f"### Mmmh.. it looks like _{font}_")
        st.image( text_to_image("Your", f"data/fonts/{font.replace('_',' ')}.ttf", 1, False) )
        st.image( text_to_image("font", f"data/fonts/{font.replace('_',' ')}.ttf", 1, False) )
        st.image( text_to_image("1234", f"data/fonts/{font.replace('_',' ')}.ttf", 1, False) )

# Generate #
with gen:
    st.header("Generate a font")
    fonts = os.listdir(f"{DATASET}/raw_dataset")

    font = st.selectbox( 'Which font do you want to use ?', fonts)
    text = st.text_input("Your text", max_chars=10)

    if st.button("Generate !"):
        image = text_to_image(text, f"data/fonts/{font.replace('_',' ')}.ttf", 1)
        st.image(image)
        st.success("Done ! Download the image now !")
