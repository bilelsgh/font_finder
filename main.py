import cv2
import numpy as np
from dotenv import load_dotenv
import os
from PIL import ImageFont, ImageDraw, Image

load_dotenv(".env")
WIDTH = int( os.getenv("width") )
HEIGHT = int( WIDTH//7 )

def get_correct_font_size(word):
    global HEIGHT

    font_size = 12
    font = ImageFont.truetype("arial.ttf", font_size)

    while ( font.getsize(word)[1] < HEIGHT ):
        font_size += 1
        font = ImageFont.truetype("arial.ttf", font_size)

    return font_size, font.getsize(word)


def create_image(word):
    font_size, text_size = get_correct_font_size(word)

    # Create an image containing <word>
    text = Image.new('RGB', (text_size[0], text_size[1]), (0, 0, 0))
    draw = ImageDraw.Draw(text)
    font = ImageFont.truetype("arial.ttf", font_size)
    draw.text((0, 0), word, (255,255,255), font=font)

    # Paste the text on a bigger image
    white = Image.new('RGB', (WIDTH,HEIGHT), (0, 0, 0))
    white.paste( text, ( (WIDTH - text_size[0])//2 , 0) )

    white.show()

def create_image_opencv(word):
    MAX_HEIGHT = WIDTH//7
    textSize = cv2.getTextSize(word, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=3, thickness=5)

    blank_text = np.zeros( [textSize[0][1],textSize[0][0],1], dtype=np.uint8 )
    blank_text.fill(255)
    text_image = cv2.putText(img=blank_text, text =word, org=(0, textSize[0][1]), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=3 , color=(0, 0, 0),thickness=5)
    text_image = cv2.resize( text_image, ( int(textSize[0][0]*MAX_HEIGHT/textSize[0][1]), MAX_HEIGHT) )
    h_txt, w_txt = text_image.shape
    


    bg = np.zeros( [WIDTH, WIDTH], dtype=np.uint8 )
    bg.fill(255)

    print(f"Bg shape: {bg.shape}\nText shape: {text_image.shape}\nOffset: {int( (WIDTH-h_txt)/2 )}")

    bg[ int( (WIDTH-h_txt)/2 ):int( (WIDTH-h_txt)/2 ) + h_txt, int( (WIDTH-w_txt)/2 ):int( (WIDTH-w_txt)/2 ) + w_txt] = text_image 


    cv2.imshow('',bg)
    cv2.waitKey(0)
    
    
    

if __name__ == "__main__":
    create_image("Bilel")