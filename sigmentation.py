import tensorflow as tf

import numpy as np
import cv2
import keras
from keras import layers

import random

import tqdm

import os

import matplotlib as plt

img_w = 256
img_h = 256

image_sets = ['1.png','2.png','3.png','4.png','5.png']

def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb,yb,angle):
    M_rotate = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb,yb

def blur(img):
    img = cv2.blur(img, (3, 3));
    return img

def add_noise(img):
    for i in range(200): # Добавить точку шума
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    return img


def data_augment(xb,yb):
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,90)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,180)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode> 0: перевернуть вдоль оси y
        yb = cv2.flip(yb, 1)

    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb,1.0)

    if np.random.random() < 0.25:
        xb = blur(xb)

    if np.random.random() < 0.2:
        xb = add_noise(xb)

    return xb,yb

def creat_dataset(image_num = 100000, mode = 'original'):
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 0
    for i in tqdm(range(len(image_sets))):
        count = 0
        src_img = cv2.imread('./data/src/' + image_sets[i])  # 3 channels
        label_img = cv2.imread('./data/label/' + image_sets[i],cv2.IMREAD_GRAYSCALE)  # single channel
        X_height,X_width,_ = src_img.shape
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w,:]
            label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
            if mode == 'augment':
                src_roi,label_roi = data_augment(src_roi,label_roi)

            visualize = np.zeros((256,256)).astype(np.uint8)
            visualize = label_roi *50

            cv2.imwrite(('./aug/train/visualize/%d.png' % g_count),visualize)
            cv2.imwrite(('./aug/train/src/%d.png' % g_count),src_roi)
            cv2.imwrite(('./aug/train/label/%d.png' % g_count),label_roi)
            count += 1
            g_count += 1

def SegNet():
    model = keras.Sequential()
    #encoder
    model.add(layers.Conv2D(64,(3,3),strides=(1,1),input_shape=(3,img_w,img_h),padding='same',activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    #(128,128)
    model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    #(64,64)
    model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    #(32,32)
    model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    #(16,16)
    model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    #(8,8)
    #decoder
    model.add(layers.UpSampling2D(size=(2,2)))
    #(16,16)
    model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.UpSampling2D(size=(2, 2)))
    #(32,32)
    model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.UpSampling2D(size=(2, 2)))
    #(64,64)
    model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.UpSampling2D(size=(2, 2)))
    #(128,128)
    model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.UpSampling2D(size=(2, 2)))
    #(256,256)
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), input_shape=(3,img_w, img_h), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same'))
    model.add(layers.Reshape((64, img_w*img_h)))
    # axis = 1 и axis = 2 позиции обмена, эквивалентные np.swapaxes (layer, 1,2)
    model.add(layers.Permute((2,1)))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    model.summary()
    return model

def get_train_val(val_rate = 0.25):
    train_url = []
    train_set = []
    val_set  = []
    for pic in os.listdir(filepath + ‘src‘):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i])
        else:
            train_set.append(train_url[i])
    return train_set,val_set

def predict(args):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = keras.models.load_model(args["model"])
    stride = args['stride']
    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
        #load the image
        image = cv2.imread('./test/' + path)
        # pre-process the image for classification
        #image = image.astype("float") / 255.0
        #image = img_to_array(image)
        h,w,_ = image.shape
        padding_h = (h//stride + 1) * stride
        padding_w = (w//stride + 1) * stride
        padding_img = np.zeros((padding_h,padding_w,3),dtype=np.uint8)
        padding_img[0:h,0:w,:] = image[:,:,:]
        padding_img = padding_img.astype("float") / 255.0
        padding_img = img_to_array(padding_img)
        print 'src:',padding_img.shape
        mask_whole = np.zeros((padding_h,padding_w),dtype=np.uint8)
        for i in range(padding_h//stride):
            for j in range(padding_w//stride):
                crop = padding_img[:3,i*stride:i*stride+image_size,j*stride:j*stride+image_size]
                _,ch,cw = crop.shape
                if ch != 256 or cw != 256:
                    print 'invalid size!'
                    continue

                crop = np.expand_dims(crop, axis=0)
                #print 'crop:',crop.shape
                pred = model.predict_classes(crop,verbose=2)
                pred = labelencoder.inverse_transform(pred[0])
                #print (np.unique(pred))
                pred = pred.reshape((256,256)).astype(np.uint8)
                #print 'pred:',pred.shape
                mask_whole[i*stride:i*stride+image_size,j*stride:j*stride+image_size] = pred[:,:]


        cv2.imwrite('./predict/pre'+str(n+1)+'.png',mask_whole[0:h,0:w])

def train(args):
    EPOCHS = 30
    BS = 16
    model = SegNet()
    modelcheck = ModelCheckpoint(args['model'],monitor='val_acc',save_best_only=True,mode='max')
    callable = [modelcheck]
    train_set,val_set = get_train_val()
    train_numb = len(train_set)
    valid_numb = len(val_set)
    print ("the number of train data is",train_numb)
    print ("the number of val data is",valid_numb)
    H = model.fit_generator(generator=generateData(BS,train_set),steps_per_epoch=train_numb//BS,epochs=EPOCHS,verbose=1,
                    validation_data=generateValidData(BS,val_set),validation_steps=valid_numb//BS,callbacks=callable,max_q_size=1)

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on SegNet Satellite Seg")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])