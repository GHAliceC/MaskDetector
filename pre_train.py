#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import shutil
import random
from pathlib import Path
from shutil import copyfile
import glob
from tqdm import tqdm

# Image Processing
from PIL import Image, ImageFile
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Classification Model
import tensorflow as tf

# Data Manipulation
import pandas as pd

# Download Data
import urllib.request

# Face recognition model to find facial landmarks
import face_recognition


# In[ ]:


get_ipython().system('pip install cmake')


# In[ ]:


get_ipython().system('pip install face_recognition')


# In[2]:


ROOT_DIR = os.path.dirname(os.path.abspath(os.curdir))

# Set seed to sample same set of images each time
random.seed(61)

# Sample 10 images per indentity (person)
N_IMAGES_PER_IDENTITY = 15

# We need nose bridge and chin to fit mask on a face
KEY_FACIAL_FEATURES = {'nose_bridge', 'chin'}
MODEL = 'hog cnn' # cnn or hog cnn is slower than hog but more accurate in terms of face detection

# Disable TF warnings (Disabling the warnings is not a good practice but we disable it to make this notebook prettier)
tf.get_logger().setLevel('ERROR')

DATA_DIR = Path(ROOT_DIR) / 'data'
VGGFACE2_DIR = Path(ROOT_DIR) / 'vggface2'

# Raw images will be used for validation and test set. Note that VGGFace2 images will be used for training. 
RAW_IMAGES_DIR = DATA_DIR / 'raw_images'


# In[ ]:

# download VGGFace2
# import urllib.request
# url = 'http://zeus.robots.ox.ac.uk/vgg_face2/get_file?fname=vggface2_test.tar.gz'
# filename = '/home/jupyter/vggface2'
# urllib.request.urlretrieve(url, filename)


# In[ ]:


# import wget
# wget.download('http://zeus.robots.ox.ac.uk/vgg_face2/get_file?fname=vggface2_test.tar.gz')


# In[ ]:


# !pip install wget


# In[28]:


import requests
import getpass
import sys

LOGIN_URL = "http://zeus.robots.ox.ac.uk/vgg_face2/login/"
FILE_URL = "http://zeus.robots.ox.ac.uk/vgg_face2/get_file?fname=vggface2_test.tar.gz"

print('Please enter your VGG Face 2 credentials:')
user_string = input('    User: ')
password_string = getpass.getpass(prompt='    Password: ')

payload = {
    'username': user_string,
    'password': password_string
}

session = requests.session()
r = session.get(LOGIN_URL)

if 'csrftoken' in session.cookies:
    csrftoken = session.cookies['csrftoken']
elif 'csrf' in session.cookies:
    csrftoken = session.cookies['csrf']
else:
    raise ValueError("Unable to locate CSRF token.")

payload['csrfmiddlewaretoken'] = csrftoken

r = session.post(LOGIN_URL, data=payload)

filename = FILE_URL.split('=')[-1]

with open(filename, "wb") as f:
    print(f"Downloading file: `{filename}`")
    r = session.get(FILE_URL, data=payload, stream=True)
    bytes_written = 0
    for data in r.iter_content(chunk_size=4096):
        f.write(data)
        bytes_written += len(data)
        MiB = bytes_written / (1024 * 1024)
        sys.stdout.write(f"\r{MiB:0.2f} MiB downloaded...")
        sys.stdout.flush()

print("\nDone.")


# In[ ]:


import tarfile
my_tar = tarfile.open('vggface2_test.tar.gz')
my_tar.extractall('/home/jupyter/vggface2') # specify which folder to extract to
my_tar.close()


# In[3]:


# Find identity directories in test folder
identity_dirs = [x for x in VGGFACE2_DIR.iterdir() if x.is_dir()]

# Create DATA directory if it does not exist
os.makedirs(DATA_DIR, exist_ok=True)

# Create a directory to save sampled images if not created
sampled_image_dir = os.path.join(ROOT_DIR, 'data', 'sampled_face_images')

# Create sampled image directory if not created to store sample images from VGG Dataset
os.makedirs(sampled_image_dir, exist_ok=True)

# Delete all images from sampled_face_images folder if already exists
images = glob.glob(os.path.join(sampled_image_dir, '*'))
for f in images:
    os.remove(f)
    
# Get N_IMAGES_PER_IDENTITY sample images from each identity and save it to sample_faces directory
for identity_dir in identity_dirs:
    indentity_face_images = glob.glob(os.path.join(identity_dir, "*.jpg"))
    if len(indentity_face_images) < 10:
        continue
    sampled_image_dirs = random.sample(indentity_face_images, N_IMAGES_PER_IDENTITY)
    identity_name = os.path.basename(identity_dir)
    for src_dir in sampled_image_dirs:
        face_image_fname = os.path.basename(src_dir)
        dst_dir = os.path.join(sampled_image_dir, identity_name + '_' + face_image_fname)
        copyfile(src_dir, dst_dir)


# In[4]:


def create_masked_face(image_path, mask_path, crop_face=True):
    # Convert image into a format that face_recognition library understands
    face_image_np = face_recognition.load_image_file(image_path)
    # Recognize face boundaries from an image 
    face_locations = face_recognition.face_locations(face_image_np, model=MODEL)
    # Find facial landmarks from the recognized face to fit mask
    face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)
    has_key_face_landmarks = check_face_landmarks(face_landmarks)
    
    if has_key_face_landmarks:
        face_img = Image.fromarray(face_image_np)
        mask_img = Image.open(mask_path)
        face_mask_img = mask_face(face_img, mask_img, face_landmarks[0])
        if crop_face:
            return crop_image(face_mask_img, face_locations[0])
        else: 
            return face_mask_img
    else:
        return None

def check_face_landmarks(face_landmarks):
    # Check if the face have enough information for use
    # May root out those cartoon/unreal faces
    if len(face_landmarks) > 0:
        # Check face_landmarks include all key facial features to fit mask
        if face_landmarks[0].keys() >= KEY_FACIAL_FEATURES:
            return True
        else:
            return False
    else:
        return False

def mask_face(face_img, mask_img, face_landmark):
    nose_bridge = face_landmark['nose_bridge']
    nose_point = nose_bridge[len(nose_bridge) * 1 // 4]
    nose_v = np.array(nose_point)

    chin = face_landmark['chin']
    chin_len = len(chin)
    chin_bottom_point = chin[chin_len // 2]
    chin_bottom_v = np.array(chin_bottom_point)
    chin_left_point = chin[chin_len // 8]
    chin_right_point = chin[chin_len * 7 // 8]

    width = mask_img.width
    height = mask_img.height
    width_ratio = 1.1
    new_height = int(np.linalg.norm(nose_v - chin_bottom_v))

    # left
    mask_left_img = mask_img.crop((0, 0, width // 2, height))
    mask_left_width = get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
    mask_left_width = int(mask_left_width * width_ratio)
    mask_left_img = mask_left_img.resize((mask_left_width, new_height))

    # right
    mask_right_img = mask_img.crop((width // 2, 0, width, height))
    mask_right_width = get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
    mask_right_width = int(mask_right_width * width_ratio)
    mask_right_img = mask_right_img.resize((mask_right_width, new_height))

    # merge mask
    size = (mask_left_img.width + mask_right_img.width, new_height)
    mask_img = Image.new('RGBA', size)
    mask_img.paste(mask_left_img, (0, 0), mask_left_img)
    mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

    # rotate mask
    angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
    rotated_mask_img = mask_img.rotate(angle, expand=True)

    # calculate mask location
    center_x = (nose_point[0] + chin_bottom_point[0]) // 2
    center_y = (nose_point[1] + chin_bottom_point[1]) // 2

    offset = mask_img.width // 2 - mask_left_img.width
    radian = angle * np.pi / 180
    box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
    box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

    # add mask
    face_img.paste(mask_img, (box_x, box_y), mask_img)
    return face_img

def get_distance_from_point_to_line(point, line_point1, line_point2):
    distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                      (line_point1[0] - line_point2[0]) * point[1] +
                      (line_point2[0] - line_point1[0]) * line_point1[1] +
                      (line_point1[1] - line_point2[1]) * line_point1[0]) / \
               np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                       (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
    return int(distance)

def save(save_dir, fname, face_img):
    dest_path = os.path.join(save_dir, fname)
    face_img.save(dest_path)
    
def crop_image(img, face_location):
    top, right, bottom, left = face_location
    return img.crop((left, top, right, bottom))


# In[5]:


# Get available masks in mask-templates dir
available_masks = glob.glob(os.path.join(ROOT_DIR, 'data', 'mask-templates', "*.png"))
available_sample_face_imgs = glob.glob(os.path.join(ROOT_DIR, 'data', 'sampled_face_images', "*.jpg"))
random.shuffle(available_sample_face_imgs)

# Half of the images will be masked and the other half will be not masked
target_n_masked_face_imgs = len(available_sample_face_imgs) / 2
n_masked_images = 0

# Create a directory to save masked / not masked images
train_dir = Path(DATA_DIR) / 'train'
masked_img_dir = train_dir / 'masked'
not_masked_img_dir = train_dir / 'not_masked'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(masked_img_dir, exist_ok=True)
os.makedirs(not_masked_img_dir, exist_ok=True)

# Delete all train data images if it already exists
images = glob.glob(os.path.join(masked_img_dir, '*.jpg'))
images += glob.glob(os.path.join(not_masked_img_dir, '*.jpg'))
for f in images:
    os.remove(f)
    
pbar = tqdm(total=target_n_masked_face_imgs)

# Add artificial mask to the detected faces until we reach half of the images in sampled_face_images directory
while n_masked_images < target_n_masked_face_imgs:
    image_path = available_sample_face_imgs.pop()
    fname  = os.path.basename(image_path)
    
    random_mask_path = random.choice(available_masks)
    try:
        masked_face = create_masked_face(image_path, random_mask_path)
    except:
        print('{} file is passed'.format(fname))
    if masked_face is not None:
        save(masked_img_dir, fname, masked_face)
        n_masked_images += 1
        pbar.update(1)
pbar.close()

# Plot some examples
ex_artif_masked_face_dirs = random.sample(glob.glob(os.path.join(masked_img_dir, '*.jpg')), 9)
fig = plt.figure(figsize=(10, 10))
for idx, artif_masked_face_dir in enumerate(ex_artif_masked_face_dirs):
    pil_im = Image.open(artif_masked_face_dir)
    im_array = np.asarray(pil_im)
    fig.add_subplot(3, 3, idx + 1)
    plt.imshow(im_array)
plt.show()


# In[38]:


ls = glob.glob(os.path.join(real_masked_val_dir, "*.jpg"))
len(ls)
# len(available_sample_face_imgs)


# In[16]:


import gc
gc.collect()


# In[17]:


# get unmasked face and write to disk
for remaining_img_path in tqdm(available_sample_face_imgs):
    # Convert image into format that face_recognition library understands 
    face_image_np = face_recognition.load_image_file(remaining_img_path)
    
    # Recognize face boundaries from an image 
    face_locations = face_recognition.face_locations(face_image_np, model=MODEL)
    face_img = Image.fromarray(face_image_np)
    if len(face_locations) > 0:
        cropped_img = crop_image(face_img, face_locations[0])
        fname = os.path.basename(remaining_img_path)
        save(not_masked_img_dir, fname, cropped_img)
        
# # Plot some examples
# ex_not_masked_face_dirs = random.sample(glob.glob(os.path.join(not_masked_img_dir, '*.jpg')), 9)
# fig = plt.figure(figsize=(10, 10))
# for idx, not_masked_face_dir in enumerate(ex_not_masked_face_dirs):
#     pil_im = Image.open(not_masked_face_dir)
#     im_array = np.asarray(pil_im)
#     fig.add_subplot(3, 3, idx + 1)
#     plt.imshow(im_array)
# plt.show()


# In[21]:


artif_masked_val_dir = DATA_DIR / 'validation' / 'artificial' / 'masked'
artif_not_masked_val_dir = DATA_DIR / 'validation' / 'artificial' / 'not_masked'

# Create DATA directory if it does not exist
os.makedirs(artif_masked_val_dir, exist_ok=True)
os.makedirs(artif_not_masked_val_dir, exist_ok=True)

sample_ratio = 0.1

# Move 20% of the random masked faces data from train to validation
train_masked_imgs = glob.glob(str(DATA_DIR / 'train' / 'masked' / '*.jpg'))
sample_train_masked_imgs = random.sample(train_masked_imgs, int(sample_ratio * len(train_masked_imgs)))
for src_img_dir in sample_train_masked_imgs:
    shutil.move(src_img_dir, artif_masked_val_dir / os.path.basename(src_img_dir))
    
# Move 20% of the random not masked faces data from train to validation
train_not_masked_imgs = glob.glob(str(DATA_DIR / 'train' / 'not_masked' / '*.jpg'))
sample_train_not_masked_imgs = random.sample(train_not_masked_imgs, int(sample_ratio * len(train_not_masked_imgs)))
for src_img_dir in sample_train_not_masked_imgs:
    shutil.move(src_img_dir, artif_not_masked_val_dir / os.path.basename(src_img_dir))


# In[35]:


real_masked_val_dir = DATA_DIR / 'validation' / 'real' / 'masked'
real_not_masked_val_dir = DATA_DIR / 'validation' / 'real' / 'not_masked'

# Create directories if cropped directory does not exist 
os.makedirs(str(real_masked_val_dir), exist_ok=True)
os.makedirs(str(real_not_masked_val_dir), exist_ok=True)

# Delete all real validation set images if it already exists
images = glob.glob(os.path.join(str(real_masked_val_dir), '*.jpg'))
images += glob.glob(os.path.join(str(real_not_masked_val_dir), '*.jpg'))
for f in images:
    os.remove(f)


# In[36]:

# PSA dataset head portrait intrepretation
import xml.etree.ElementTree as ET
from PIL import Image
import cv2
import csv
import glob

files = glob.glob("/home/jupyter/face-mask-detection/annotations/*.xml")
output_csv = open("/home/jupyter/face-mask-detection/metadata.csv", "w")

# # Create a directory to save validation masked / not masked images
# valid_dir = Path(DATA_DIR) / 'validation'
# real_masked_img_dir = valid_dir / 'real' / 'masked'
# real_not_masked_img_dir = valid_dir / 'real' / 'not_masked'
# os.makedirs(valid_dir, exist_ok=True)
# os.makedirs(real_masked_img_dir, exist_ok=True)
# os.makedirs(real_not_masked_img_dir, exist_ok=True)

csvwriter = csv.writer(output_csv)
csv_head = ["class", "xmin", "xmax", "ymin", "ymax", "image_dir"]
csvwriter.writerow(csv_head)

for f in files:
    tree = ET.parse(f)
    root = tree.getroot()
    img_name = root.find('filename').text
    img = cv2.imread(os.path.join(ROOT_DIR, "face-mask-detection", "images", img_name))
    count = -1
    for member in root.findall("object"):
        count+=1
        row = []
        c = member.find("name").text
        bndbox = member.find("bndbox")
        box_xmin = int(bndbox.find("xmin").text)
        box_xmax = int(bndbox.find("xmax").text)
        box_ymin = int(bndbox.find("ymin").text)
        box_ymax = int(bndbox.find("ymax").text)
        row.append(c)
        row.append(box_xmin)
        row.append(box_xmax)
        row.append(box_ymin)
        row.append(box_ymax)
        row.append(img_name)
        csvwriter.writerow(row)
        
#         Tricks 1: keep size over 50*50 pixels' face image
        if box_ymax - box_ymin < 50:
            continue
        
        pic = img[box_ymin:box_ymax, box_xmin:box_xmax,::-1]
        pic = Image.fromarray(pic)
        img_crop_name = img_name[:-4] + "_" + str(count) + ".jpg"
        if c == 'with_mask': 
            pic.save(os.path.join(real_masked_val_dir, img_crop_name))
#             pic.show()
#             print(os.path.join(real_masked_img_dir, img_crop_name))
        else:
            pic.save(os.path.join(real_not_masked_val_dir, img_crop_name))
#             pic.show()
#             print(os.path.join(real_not_masked_img_dir, img_crop_name))  
output_csv.close()


# In[39]:


real_masked_test_dir = DATA_DIR / 'test' / 'masked'
real_not_masked_test_dir = DATA_DIR / 'test' / 'not_masked'

# Create DATA directory if it does not exist
os.makedirs(real_masked_test_dir, exist_ok=True)
os.makedirs(real_not_masked_test_dir, exist_ok=True)

sample_ratio = 0.1

# Move 20% of the random masked faces real data from validation to test
validate_masked_imgs = glob.glob(str(DATA_DIR / 'validation' / 'real' / 'masked' / '*.jpg'))
sample_validate_masked_imgs = random.sample(validate_masked_imgs, int(sample_ratio * len(validate_masked_imgs)))
for src_img_dir in sample_validate_masked_imgs:
    shutil.move(src_img_dir, real_masked_test_dir / os.path.basename(src_img_dir))
    
# Move 20% of the random not masked faces real data from validation to test
validate_not_masked_imgs = glob.glob(str(DATA_DIR / 'validation' / 'real' / 'not_masked' / '*.jpg'))
sample_validate_not_masked_imgs = random.sample(validate_not_masked_imgs, int(sample_ratio * len(validate_not_masked_imgs)))
for src_img_dir in sample_validate_not_masked_imgs:
    shutil.move(src_img_dir, real_not_masked_test_dir / os.path.basename(src_img_dir))


# In[32]:


# Alternative way to crop images
from PIL import Image
img = Image.open("/home/jupyter/face-mask-detection/images/maksssksksss110.png")
# Setting the points for cropped image 
left = 6
top = 43
right = 111
bottom = 148
# 111:148, 6:43
  
# Cropped image of above dimension 
# (It will not change orginal image) 
im1 = img.crop((left, top, right, bottom)) 
  
# Shows the image in image viewer 
im1.save("/home/jupyter/try.png")


# In[ ]:





# In[ ]:


ls /home/jupyter/vggface2/.ipynb_checkpoints


# In[4]:


# !kaggle datasets download -d ashishjangra27/face-mask-12k-images-dataset


# In[6]:


# !ls


# In[34]:

# HPI dataset download 
get_ipython().system('unzip face-mask-12k-images-dataset.zip')


# In[35]:


get_ipython().system('mv /home/jupyter/faceRecognition/Face\\ Mask\\ Dataset/* /home/jupyter/face-mask-12k')


# In[19]:


import matplotlib.pyplot as plt
img = plt.imread("/home/jupyter/face-mask-12k/Train/WithMask/10.png")
plt.imshow(img)
img.shape


# In[66]:


ls = glob.glob(os.path.join(masked_img_dir, "*.jpg"))
len(ls)


# In[64]:


import cv2
from PIL import Image
for img_path in ls:
    img = cv2.imread(img_path)
    if img.shape[0] < 50 or img.shape[1] < 50:
        continue
    image = face_recognition.load_image_file(img_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_landmarks = face_recognition.face_landmarks(img, face_locations, "large")
    if "left_eye" not in face_landmarks[3].keys() and "right_eye" not in face_landmarks[3].keys():
        continue
    filename = os.path.basename(img_path)[:-4] +".jpg"
#     img.save(os.path.join(masked_img_dir, filename))
    cv2.imwrite(os.path.join(real_not_masked_test_dir, filename), img)


# In[61]:


train_dir = Path(DATA_DIR) / 'train'
masked_img_dir = train_dir / 'masked'
not_masked_img_dir = train_dir / 'not_masked'
real_masked_val_dir = DATA_DIR / 'validation' / 'real' / 'masked'
real_not_masked_val_dir = DATA_DIR / 'validation' / 'real' / 'not_masked'
real_masked_test_dir = DATA_DIR / 'test' / 'masked'
real_not_masked_test_dir = DATA_DIR / 'test' / 'not_masked'


# In[31]:


type(os.path.basename(ls[0]))


# In[49]:


os.path.basename(img_path)[:-4] +".jpg"


# In[46]:


img_path = "/home/jupyter/face-mask-12k/Train/WithMask/Augmented_62_20563.png"


# In[ ]:




