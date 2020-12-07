#!/usr/bin/env python
# coding: utf-8

# In[10]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNorm
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import preprocess_input

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
import os
import cv2

from sklearn import metrics 
from pathlib import Path


# In[2]:


target_size = (100,100) # （112，112）
batch_size = 24 # 32
lr = 0.01 # [0.005, 0.008, 0.015, 0.02]
n_epochs = 10

root_dir = os.path.dirname(os.path.abspath(os.curdir))
data_dir = Path(root_dir) / 'data'

model_dir = data_dir / 'classifier_model_weights'
model_dir.mkdir(exist_ok=True)

performance_plots_dir = Path('img')
performance_plots_dir.mkdir(exist_ok=True)


# In[3]:


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(str(data_dir / 'train'),
                                                    target_size=target_size,
                                                    batch_size=batch_size,
                                                    class_mode='binary',
                                                    classes=['not_masked', 'masked'],
                                                    shuffle=True)


# In[4]:


val_datagen_artificial = ImageDataGenerator(preprocessing_function=preprocess_input)
val_generator_artificial = val_datagen_artificial.flow_from_directory(str(data_dir / 'validation' / 'artificial'),
                                                    target_size=target_size,
                                                    batch_size=batch_size,
                                                    class_mode='binary',
                                                    classes=['not_masked', 'masked'],
                                                    shuffle=False)


# In[5]:


val_datagen_real = ImageDataGenerator(preprocessing_function=preprocess_input)
val_generator_real = val_datagen_real.flow_from_directory(str(data_dir / 'validation' / 'real'),
                                                    target_size=target_size,
                                                    batch_size=batch_size,
                                                    class_mode='binary',
                                                    classes=['not_masked', 'masked'],
                                                    shuffle=False)


# In[6]:

test_datagen_real = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator_real = test_datagen_real.flow_from_directory(str(data_dir / 'test'),
                                                    target_size=target_size,
                                                    batch_size=batch_size,
                                                    class_mode='binary',
                                                    classes=['not_masked', 'masked'],
                                                    shuffle=False)

# In[7]


base_model = MobileNet(weights='imagenet',include_top=False, input_shape=(target_size[0],target_size[1],3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(0.2)(x) # [0.1, 0.15, 0.25]
preds = Dense(1,activation='sigmoid')(x)

model = Model(inputs=base_model.input,outputs=preds)
print(model.layers[])

for layer in model.layers[:-4]:
    layer.trainable = False


# base_model = MobileNetV2(weights='imagenet',include_top=False, input_shape=(target_size[0],target_size[1],3))

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(128,activation='relu')(x)
# preds = Dense(1,activation='sigmoid')(x)

# model = Model(inputs=base_model.input,outputs=preds)

# for layer in model.layers[:-4]:
#     layer.trainable = False

# In[8]:


class SubsetAccuracy(tf.keras.callbacks.Callback):
    """
    We want to monitor accuracy in the validation set separately for real and artificial face masks.
    This callback will print this to the output, and store the values for each epoch.
    
    It also stores the best model (according to validation accuracy on the masked faces) to disk.
    """
    
    def __init__(self, real_val_gen=None):
        self.real_val_gen = real_val_gen
        self.cur_best_acc = 0
        
    def on_epoch_end(self, batch, logs={}):
        pred = self.model.predict(self.real_val_gen)
        bin_pred = [x > 0.5 for x in pred]
        real_acc = metrics.accuracy_score(self.real_val_gen.classes, bin_pred)
        
        print(f"Accuracy on the real validation set: {real_acc:.2f}")
        
        if real_acc > self.cur_best_acc:
            self.model.save(model_dir / 'best.h5')
            self.cur_best_acc = real_acc
          
subset_acc_val = SubsetAccuracy(val_generator_real)
subset_acc_test = SubsetAccuracy(test_generator_real)


# In[9]:


opt = tf.keras.optimizers.(learning_rate=lr)
model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
step_size_train = train_generator.n//train_generator.batch_size
step_size_val = val_generator_artificial.n//val_generator_artificial.batch_size


# In[10]:


model.fit_generator(generator=train_generator, steps_per_epoch=step_size_train, 
                    epochs=n_epochs, validation_data=val_generator_artificial, 
                    validation_steps=step_size_val, callbacks=[subset_acc_val])


# In[11]:


best_model = tf.keras.models.load_model(model_dir / 'best.h5')


# In[12]:


# val_pred = best_model.predict(val_generator_real)
test_pred = best_model.predict(test_generator_test)


# In[13]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    From: https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[14]:


# val_pred_bin = [x[0] > 0.5 for x in val_pred]

# acc = metrics.accuracy_score(val_generator_real.classes, val_pred_bin)
# print(f"Accuracy = {acc:.3f}")
# test
test_pred_bin = [x[0] > 0.5 for x in test_pred]

acc = metrics.accuracy_score(test_generator_real.classes, test_pred_bin)
print(f"Accuracy_test = {acc:.3f}")


# In[15]:


# cm = metrics.confusion_matrix(val_generator_real.classes, val_pred_bin)
# plt.figure()
# plot_confusion_matrix(cm, ['not masked', 'masked'])
# plt.savefig(performance_plots_dir / 'confusion.png', bbox_inches='tight', pad_inches=0)

cm = metrics.confusion_matrix(test_generator_real.classes, test_pred_bin)
plt.figure()
plot_confusion_matrix(cm, ['not masked', 'masked'])
plt.savefig(performance_plots_dir / 'confusion_test.png', bbox_inches='tight', pad_inches=0)

# In[16]:


# fpr, tpr, thr = metrics.roc_curve(val_generator_real.classes, val_pred)
# auc = metrics.auc(fpr, tpr) 

# plt.figure()
# plt.plot(fpr, tpr, color='darkorange')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([-0.01, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate', size=15)
# plt.ylabel('True Positive Rate', size=15)
# plt.title(f'ROC for mask/no_mask classification\nAUC = {auc:.3f}', size=15)
# plt.savefig(performance_plots_dir / 'roc_classification.png', bbox_inches='tight', pad_inches=0)

fpr, tpr, thr = metrics.roc_curve(test_generator_real.classes, test_pred)
auc = metrics.auc(fpr, tpr) 

plt.figure()
plt.plot(fpr, tpr, color='darkorange')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', size=15)
plt.ylabel('True Positive Rate', size=15)
plt.title(f'ROC for mask/no_mask classification\nAUC = {auc:.3f}', size=15)
plt.savefig(performance_plots_dir / 'roc_classification.png', bbox_inches='tight', pad_inches=0)


# In[17]:


n_to_plot = 16

mistakes = []
correct = []
for fn, true_label, pred_label in zip(val_generator_real.filenames, val_generator_real.classes, val_pred_bin):
    if true_label != pred_label:
        mistakes.append(data_dir / 'validation' / 'real'/ fn)
    else: 
        correct.append(data_dir / 'validation' / 'real'/ fn)
        
nrow = np.ceil(np.sqrt(n_to_plot))
ncol = nrow

to_plot = random.sample(correct,n_to_plot)
plt.figure(figsize=(10,10))
for idx, x in enumerate(to_plot):
    plt.subplot(nrow, ncol, idx+1)
    img = cv2.imread(str(x))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
plt.suptitle('Sample of images classified correctly', size=20, y=0.92)
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.show()

to_plot = random.sample(correct,n_to_plot)
plt.figure(figsize=(10,10))
for idx, x in enumerate(to_plot):
    plt.subplot(nrow, ncol, idx+1)
    img = cv2.imread(str(x))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
plt.suptitle('Sample of images classified wrongly', size=20, y=0.92)
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.show()


# In[ ]:

Epoch = [1,2,3,4,5,6,7,8,9,10]
Train_Loss_v1 = [0.0561,0.0175,0.0102,0.0212,0.0077,0.0055, 0.0127,0.0094,0.0019,0.0017]
Valid_Loss_v1 = [0.0747,0.0079,0.0148,0.0223,0.0174,0.0174,0.0465,0.0333,0.0383,0.0294]
Train_v1 = []
Valid_v1 = []
for i in Train_Loss_v1:
    Train_v1.append(1 - i)
print(Train_v1)
for i in Valid_Loss_v1:
    Valid_v1.append(1 - i)
print(Valid_v1)
# Train_v1 = [0.9818,0.9880,0.9869,0.9940,0.9890,0.9875,0.9897,0.9881,0.9894,0.9895]
# Valid_v1 = [0.9620,0.9781,0.9965,0.9961,0.9957,0.9977,0.9910,0.9949,0.9959,0.9949]

import matplotlib.pyplot as plt
import pandas as pd
# df = pd.DataFrame({"Epoch": Epoch; "Train_Loss_v1": Train_Loss_v1, "Valid_Loss_v1": Valid_Loss_v1,
#                   "Train_acc_v1": Train_v1, "Valid_acc_v1": Valid_v1})
# plt.plot( "Epoch", 'Train_acc_v1', data=df, marker='-', markerfacecolor='blue', color='skyblue', linewidth=2)
# plt.plot( 'Epoch', 'Valid_acc_v1', data=df, marker='', color='olive', linewidth=2)
# plt.plot( 'Epoch', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
# plt.legend()


acc = Train_v1
val_acc = Valid_v1

loss = Train_Loss_v1
val_loss = Valid_Loss_v1

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(Epoch, acc, label='Training Accuracy')
plt.plot(Epoch, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([0.85,1.05])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(Epoch, loss, label='Training Loss')
plt.plot(Epoch, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([-0.1,0.5])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


