"""
usage : python partitions.py [-H] [-i IMAEDIR] [-o OUTPUTDIR] [-r RATIO] [-x]
partion dataset into train and test
optional arguments:
    -h, --help            show this help message and exit
    -i IMAGEDIR, --imagedir IMAGEDIR
                            image directory path , if not specified the cwd will be used
    -o OUTPUTDIR, --outputdir OUTPUTDIR
                            output directory path , if not specified  defaults to the same directory as IMAGEDIR.
    -r RATIO, --ratio RATIO
                            ratio of train/test , default is 0.1
    -x, --xml             if specified , the xml files will be used to generate the train/test

execute : python partitions.py -x -i /home/hassen/project/models/models/images -o /home/hassen/project/models/models/images -r 0.2

"""
import math
import os
import re
from shutil import copyfile
import argparse
import random

def iterate_dir(source, dest, ratio, xml):
    '''
    iterate over the source directory and copy the files to the destination directory
    :param source: source directory
    :param dest: destination directory
    :param ratio: ratio of train/test
    :param xml: if true , the xml files will be used to generate the train/test
    :return:
    '''
    train_dir = os.path.join(dest, 'train') # path to train dir
    test_dir = os.path.join(dest, 'test') # path to test dir

    if not os.path.exists(train_dir): # create train dir if not exists
        os.umask(0)
        os.makedirs(train_dir, 0o777)
    if not os.path.exists(test_dir): # create test dir if not exists
        os.umask(0)
        os.makedirs(test_dir, 0o777)

    images = [f for f in os.listdir(source) if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.jpg|.jpeg|.png)$', f)] # get all images
    num_images = len(images) # number of images
    num_test_images = math.ceil(ratio * num_images) # number of test images
    for i in range(num_test_images): # iterate over test images
        idx = random.randint(0, len(images) - 1) # get random index
        filename = images[idx] # get random image
        copyfile(os.path.join(source, filename), os.path.join(test_dir, filename)) # copy image to test dir
        if xml: # if xml flag is set
            xml_filename = os.path.splitext(filename)[0] + '.xml' # get xml filename
            copyfile(os.path.join(source, xml_filename), os.path.join(test_dir, xml_filename)) # copy xml to test dir
        images.pop(idx) # remove image from images list

        for filename in images: # iterate over train images
            copyfile(os.path.join(source, filename), os.path.join(train_dir, filename)) # copy image to train dir
            if xml: # if xml flag is set
                xml_filename = os.path.splitext(filename)[0] + '.xml' # get xml filename
                copyfile(os.path.join(source, xml_filename), os.path.join(train_dir, xml_filename)) # copy xml to train dir

def main():
    parser = argparse.ArgumentParser(description='partion dataset into train and test' , formatter_class=argparse.RawTextHelpFormatter) # create parser
    parser.add_argument('-i', '--imagedir', help='image directory path , if not specified the cwd will be used' , type=str , default=os.getcwd()) # add image dir argument
    parser.add_argument('-o', '--outputdir', help='output directory path ,if not specified defaults to the same directory as IMAGEDIR.' , type=str , default=os.getcwd()) # add output dir argument
    parser.add_argument('-r', '--ratio', help='ratio of train/test , default is 0.1' , type=float , default=0.1) # add ratio argument
    parser.add_argument('-x', '--xml' , help='if specified , the xml files will be used to generate the train/test' , action='store_true') # add xml flag
    args = parser.parse_args() # parse arguments

    if args.outputdir is None:
        args.outputdir = args.imagedir

    iterate_dir(args.imagedir, args.outputdir, args.ratio, args.xml) # call iterate_dir function

if __name__ == '__main__':
    main()
