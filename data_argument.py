from PIL import Image
from numpy import asarray
import os

def load_dir(directory,crop_directory,crop_size):
    no_file = 0
    for img_file in os.listdir(directory):
        no_file += 1
        for img in os.listdir(directory + "\\" + img_file):
            image = Image.open(directory + "\\" + img_file + "\\" + img)
            imgwidth, imgheight = image.size
            if imgwidth%crop_size == 0 and imgheight%crop_size == 0:
                for i in range(0,imgheight,crop_size):
                    for j in range(0,imgwidth,crop_size):
                        box = (j, i, j+crop_size, i+crop_size)
                        a = image.crop(box)
                        img_name = crop_directory
                        print(img.lower())
                        if img.lower().endswith("rough_ao_1k.jpg"):
                            img_name += "\\" + "rough_ao" + "\\" + "img_" + str(no_file) + "_" + str(i) + "_" + str(j) + "_ao.jpg"
                        elif img.lower().endswith("diff_1k.jpg"):
                            img_name += "\\" + "diff" + "\\" + "img_" + str(no_file) + "_" + str(i) + "_" + str(j) + "_diff.jpg"
                        elif img.lower().endswith("disp_1k.jpg"):
                            img_name += "\\" + "disp" + "\\" + "img_" + str(no_file) + "_" + str(i) + "_" + str(j) + "_disp.jpg"
                        elif img.lower().endswith("nor_1k.jpg"):
                            img_name += "\\" + "nor" + "\\" + "img_" + str(no_file) + "_" + str(i) + "_" + str(j) + "_nor.jpg"
                        elif img.lower().endswith("rough_1k.jpg"):
                            img_name += "\\" + "rough" + "\\" + "img_" + str(no_file) + "_" + str(i) + "_" + str(j) + "_rough.jpg"
                        elif img.lower().endswith("ao_1k.jpg"):
                            img_name += "\\" + "ao" + "\\" + "img_" + str(no_file) + "_" + str(i) + "_" + str(j) + "_rough_ao.jpg"
                        a.save(img_name)
            else:
                print("cannot crop due to wrong input size")
                            
def JPGtoNumpy(image_path):
    image = Image.open(image_path)
    image.load()
    data = asarray(image)
    return data

