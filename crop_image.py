from data_argument import *

import os

if __name__ == "__main__":
    imgDirectory = os.getcwd()+"\image"
    cropDirectory = os.getcwd()+"\crop_image"
    load_dir(imgDirectory, cropDirectory, 256)
