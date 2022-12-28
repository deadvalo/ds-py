"""
Sample TensorFlow code to convert xml annotations to TFRecord format.
usage : python TFRecord.py [-h] [-x XML_DIR] [-l LABELS_PATH] [-o OUTPUT_PATH] [-i IMAGE_DIR] [-c CSV_PATH]
optional arguments:
    -h, --help            show this help message and exit
    -x XML_DIR, --xml_dir XML_DIR  xml directory path
    -l LABELS_PATH, --labels_path LABELS_PATH labels path (.pbtxt) file
    -o OUTPUT_PATH, --output_path OUTPUT_PATH output path (.record) file
    -i IMAGE_DIR, --image_dir IMAGE_DIR image directory path , default is the same directory as XML_DIR
    -c CSV_PATH, --csv_path CSV_PATH csv path (.csv) file , if not provided , the xml files will be used to generate the train/test

creaye train data:
    python TFRecord.py -x /home/hassen/project/models/models/images/train -l /home/hassen/project/models/models/annotations/label_map.pbtxt -o /home/hassen/project/models/models/annotations/train.record -c /home/hassen/project/models/models/annotations/train_labels.csv
create test data:
    python TFRecord.py -x /home/hassen/project/models/models/images/test -l /home/hassen/project/models/models/annotations/label_map.pbtxt -o /home/hassen/project/models/models/annotations/test.record -c /home/hassen/project/models/models/annotations/test_labels.csv
"""

import os
import io
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET
import glob
from PIL import Image
import argparse
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple, OrderedDict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # supress warnings

parser = argparse.ArgumentParser(description='Sample TensorFlow code to convert xml annotations to TFRecord format.' , formatter_class=argparse.RawTextHelpFormatter) # create parser
parser.add_argument('-x', '--xml_dir', help='xml directory path' , type=str) # add xml dir argument
parser.add_argument('-l', '--labels_path', help='labels path (.pbtxt) file' , type=str) # add labels path argument
parser.add_argument('-o', '--output_path', help='output path (.record) file' , type=str) # add output path argument
parser.add_argument('-i', '--image_dir', help='image directory path , default is the same directory as XML_DIR' , type=str, default=None) # add image dir argument
parser.add_argument('-c', '--csv_path', help='csv path (.csv) file , if not provided , the xml files will be used to generate the train/test' , type=str, default=None) # add csv path argument
args = parser.parse_args() # parse the arguments

if args.image_dir is None: # if image dir is not provided
    args.image_dir = args.xml_dir

label_map = label_map_util.load_labelmap(args.labels_path) # load label map
label_map_dict = label_map_util.get_label_map_dict(label_map) # get label map dict


def class_text_to_int(row_label): # convert class text to int
    return label_map_dict[row_label]

def split(df, group): # split data into train/test
    data = namedtuple('data', ['filename', 'object']) # create namedtuple
    gb = df.groupby(group) # group by group
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)] # return data

def create_tf_example(group, path): # create tf example
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid: # open image
        encoded_jpg = fid.read() # read image
    encoded_jpg_io = io.BytesIO(encoded_jpg) # create io
    image = Image.open(encoded_jpg_io) # open image
    width, height = image.size # get width and height

    filename = group.filename.encode('utf8') # encode filename
    image_format = b'jpg' # set image format
    x_min = [] # create empty list for x_min
    x_max = [] # create empty list for x_max
    y_min = [] # create empty list for y_min
    y_max = [] # create empty list for y_max
    classes_text = [] # create empty list for classes_text
    classes = [] # create empty list for classes

    for index, row in group.object.iterrows(): # iterate over each object
        x_min.append(row['xmin'] / width) # append x_min
        x_max.append(row['xmax'] / width) # append x_max
        y_min.append(row['ymin'] / height) # append y_min
        y_max.append(row['ymax'] / height) # append y_max
        classes_text.append(row['class'].encode('utf8')) # append classes_text
        classes.append(class_text_to_int(row['class'])) # append classes

    tf_example = tf.train.Example(features=tf.train.Features(feature={ # create tf example

        'image/height': dataset_util.int64_feature(height), # add height
        'image/width': dataset_util.int64_feature(width), # add width
        'image/filename': dataset_util.bytes_feature(filename), # add filename
        'image/source_id': dataset_util.bytes_feature(filename), # add source_id
        'image/encoded': dataset_util.bytes_feature(encoded_jpg), # add encoded
        'image/format': dataset_util.bytes_feature(image_format), # add format
        'image/object/bbox/xmin': dataset_util.float_list_feature(x_min), # add x_min
        'image/object/bbox/xmax': dataset_util.float_list_feature(x_max), # add x_max
        'image/object/bbox/ymin': dataset_util.float_list_feature(y_min), # add y_min
        'image/object/bbox/ymax': dataset_util.float_list_feature(y_max), # add y_max
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text), # add classes_text
        'image/object/class/label': dataset_util.int64_list_feature(classes), # add classes
    }))
    return tf_example # return tf example

def main(_): # main function
    writer = tf.io.TFRecordWriter(args.output_path) # create tf record writer
    path = os.path.join(args.image_dir) # get image path
    examples = pd.read_csv(args.csv_path) # read csv
    print(examples.head(10))# print examples
    grouped = split(examples, 'filename') # split data into train/test
    for group in grouped: # iterate over each group
        tf_example = create_tf_example(group, path) # create tf example
        writer.write(tf_example.SerializeToString()) # write tf example
    writer.close() # close tf record writer
    print('Successfully created the TFRecords: {}'.format(args.output_path)) # print success message
    if args.csv_path is None:
        print('No CSV file provided , the XML files will be used to generate the train/test')

if __name__ == '__main__': # if main
    tf.compat.v1.app.run() # run main