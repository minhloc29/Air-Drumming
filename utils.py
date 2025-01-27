import matplotlib.pyplot as plt
import random
from matplotlib.patches import Rectangle
import json
import cv2
import os 
import tensorflow as tf

def get_image_with_box(dir = "data/mouth_box"): 
    i = random.randint(1, 200)
    
    mouth = os.listdir(dir)
    with open(os.path.join(dir, mouth[i]), 'r') as f:
        label = json.load(f) 
    image = os.path.join('data', label['imagePath'][3:])
    image = cv2.imread(image)[:,:, ::-1]
    x_min = label['shapes'][0]['points'][0][0]
    y_min = label['shapes'][0]['points'][0][1]
    x_max = label['shapes'][0]['points'][1][0]
    y_max = label['shapes'][0]['points'][1][1]
    
    width = x_max - x_min
    height = y_max - y_min
    
    # Display the image
    plt.imshow(image)
    ax = plt.gca()
    
    # Create a Rectangle patch
    rect = Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    
    # Show the plot
    plt.axis('off')  # Hide axes
    plt.show()
    
def get_infor_from_json(json_path):
    with open(json_path, 'r') as f:
        infor = json.load(f)
    
    coords = [0, 0, 0, 0]
    coords[0] = infor['shapes'][0]['points'][0][0]
    coords[1] = infor['shapes'][0]['points'][0][1]
    coords[2] = infor['shapes'][0]['points'][1][0]
    coords[3] = infor['shapes'][0]['points'][1][1]
    
    url = infor['imagePath']
    image_path = os.path.join("data/mouth", '/'.join(url.split('/')[2:]))
    label = int(infor['shapes'][0]['label'] == 'open')
    
    return coords, image_path, label


def load_image(x): 
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)
        
    return [label['class']], label['bbox']

def load_tensor_data(folder_path, batch_size = 16):
    images = tf.data.Dataset.list_files(f"{folder_path}/images/*.jpg", shuffle = False)
    images = images.map(load_image)
    images = images.map(lambda x: tf.image.resize(x, (224,224)))
    images = images.map(lambda x: x/255)
    
    labels = tf.data.Dataset.list_files(f"{folder_path}/labels/*.json", shuffle = False)
    labels = labels.map(lambda x : tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

    data = tf.data.Dataset.zip((images, labels))
    data = data.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    return data