## Welcome to What's the font?
_What's the font?__ is an application allowing you to find a font from a photo you took !

## Architecture
- _dashboard.py_: A Streamlit dashboard allowing: 
  - to generate an image to predict with the fonts known by the model. 
  - to predict the font of a well formated image (for prediction from natural image: _branch crop_)
- _generate_dataset.py_: Create well formated image containing a word. It uses the fonts in _data/fonts_
- _model.py_: Contains the Model class 

## Technical stack
`pip install -r requirements.txt`
- Python
- Tensorflow
- OpenCV
- PIL
- easyocr
- Streamlit (interface)

## Run
`streamlit run dashboard.py`

## Sources
- English words : https://gist.github.com/deekayen/4148741
