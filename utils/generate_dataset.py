import shutil
import random
import cv2
import numpy as np
from dotenv import load_dotenv
import os
import sys
from PIL import ImageFont, ImageDraw, Image

load_dotenv(".env")
WIDTH = int( os.getenv("width") )
HEIGHT = int( os.getenv("height") )

def get_correct_font_size(word, font):
    """
    Return the font size with a height closest to HEIGHT. 
    :param word: Word to write (string)
    :param font: Font to use (string)
    :return 1: Font size (int)
    :return 2: Size of 'word' written with 'font' and 'font_size' (tuple)
    """
    global HEIGHT

    font_size = 12
    font_obj = ImageFont.truetype(font, font_size)

    while ( font_obj.getsize(word)[1] < HEIGHT ):
        font_size += 1
        font_obj = ImageFont.truetype(font, font_size)

    return font_size, font_obj.getsize(word)


def text_to_image(word, font, index, negative=True, path=-1):
    """
    Create an image from a text
    :param word: Word to write (string)
    :param font: Font to use (string)
    :param negative: white text and black background if True, otherwise the opposite 
    :param path: Saving path
    :param index: the index-th word generated with the font 'font'
    """
    font_size, text_size = get_correct_font_size(word,font)
    bg_color =  (0,0,0) if negative else (255,255,255)
    text_color =  (0,0,0) if not negative else (255,255,255)

    # Create an image containing <word>
    text = Image.new('RGB', (text_size[0], text_size[1]), bg_color)
    draw = ImageDraw.Draw(text)
    font_obj = ImageFont.truetype(font, font_size)
    draw.text((0, 0), word, text_color, font=font_obj)

    # Paste the text on a bigger image
    bg = Image.new('RGB', (WIDTH,HEIGHT), bg_color)
    bg.paste( text, ( (WIDTH - text_size[0])//2 , 0) )

    # Save
    if path != -1 :
        font_folder = font.split("/")[-1].split(".")[0].replace(" ","_")
        path = path.replace("\\","/")
        try:
            bg.save(f"{path}/{font_folder}/{index}.jpg")
        except:
            os.mkdir(f"{path}/{font_folder}")
            bg.save(f"{path}/{font_folder}/{index}.jpg")
    
    return bg


def create_images(path, nb_font=-1):
    # Get the english words (1000)
    with open("data/words_en.txt", "r") as f:
        words = f.readlines()

    words = list( map( lambda x : x.replace("\n",""), words) )
    print(f".About to generate {len(words)} images for {nb_font} fonts..")

    # Generate images
    fonts_dir = os.listdir('data/fonts')
    random.shuffle(fonts_dir)
    fonts = fonts_dir if nb_font == -1 else fonts_dir[:nb_font]

    for font in fonts :
        index = 1
        for word in words :
            text_to_image(word, f"data/fonts/{font}", index, path)
            index += 1
        print(f"- {font}: all images have been generated")


def split_dataset(train_prop, src_dataset_path, dest_dataset_path):
    """
    Split a dataset intro a train and test dataset
    :param train_prop: Proportion that represents the train dataset over the whole dataset
    :param x_dataset_path: U know
    """

    for class_ in os.listdir(src_dataset_path):
        size_training_dataset = int( len( os.listdir(f"{src_dataset_path}/{class_}") ) * train_prop ) 

        # Split among test and train
        for idx,image in enumerate( os.listdir(f"{src_dataset_path}/{class_}") ) :
            step = "train" if idx <= size_training_dataset else "test"

            if step not in os.listdir(f"{dest_dataset_path}"):
                os.mkdir(f"{dest_dataset_path}/{step}")
            
            if class_ not in os.listdir(f"{dest_dataset_path}/{step}"):
                os.mkdir(f"{dest_dataset_path}/{step}/{class_}")
            
            shutil.copyfile(f"{src_dataset_path}/{class_}/{image}", f"{dest_dataset_path}/{step}/{class_}/{image}")

           
def get_images_labels(dataset_path):
    """
    From a dataset get an array containing images and another containing labels
    :param dataset_path: Path pointing to the dataset containing every class
    :return: Two np arrays images and labels + dict get the class name from an integer
    """

    images = []
    labels = []
    int_to_class = {}
    int_label = 0

    for idx,class_ in enumerate(os.listdir(dataset_path)):
        int_to_class[int(idx)] = class_
        for img in os.listdir(f"{dataset_path}/{class_}"):
            images.append( cv2.imread(f"{dataset_path}/{class_}/{img}", cv2.IMREAD_GRAYSCALE) )
            labels.append(int_label)
        int_label += 1

    return np.array( images,  dtype="float"), np.array( labels,  dtype="float"), int_to_class

if __name__ == "__main__":
    
    # Get the function to run
    try:
        function = sys.argv[1]
    except:
        exit("Please use the following format: `pythonX generate_dataset.py <function> <train_prop> <src_path> <dest_path>` and use one of the following functions: \n- create_images\n- split_dataset")

    
    # Run it
    if function == "create_images":
        try:
            nb_font = int(sys.argv[2])
            path = sys.argv[3]
        except IndexError:
            exit("Please use the following format: pythonX generate_dataset.py create_images <nb_font> <save_path>")
        except ValueError:
            exit("Please provide an integer for the parameter 'nb_font'")
        create_images(path, nb_font)
    
    elif function == "split_dataset":
        try:
            train_prop = float(sys.argv[2])
            src_path = sys.argv[3].replace("\\","/")
            dest_path = sys.argv[4].replace("\\","/")
        except IndexError:
            exit("Please use the following format: pythonX generate_dataset.py split_dataset <train_prop> <src_path> <dest_path>")
        except ValueError:
            exit("Please provide a float for the parameter 'train_prop'")
        split_dataset(train_prop,src_path,dest_path)
        
    else:
        exit("Please use the following format: `pythonX generate_dataset.py <function> <train_prop> <src_path> <dest_path>` and use one of the following functions: \n- create_images\n- split_dataset")
        