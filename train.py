from __future__ import absolute_import, division, print_function
import tensorflow as tf
import config
from prepare_data import generate_datasets
import math
import os
import time
import numpy as np
import pandas as pd
from tensorflow import keras

# from statistics import mean
# GPU settings
devices = tf.config.experimental.list_physical_devices('GPU')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(devices[0],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(
                                                                    memory_limit=5000)])

if not os.path.exists('acc_loss.txt'):
    with open('acc_loss.txt', 'w') as file:
        pass
if not os.path.exists('mean_acc_loss.txt'):
    with open('mean_acc_loss.txt', 'w') as file:
        pass

# get the original_dataset
train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()
# model = keras.Sequential([
#     tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation='relu', padding='same'),
#     # 32为filter 卷积核个数
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#     tf.keras.layers.BatchNormalization(),
#     # CNN网络输入的是三维数据，chanel会变为32
#     tf.keras.layers.MaxPooling2D(),  # 经过此层后缩小一半 由28*28*32 变为 14*14*32
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dense(1)
# ])
model=keras.Sequential()
model.add(keras.layers.Conv2D(64,(3,3),input_shape=(256,256,3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Conv2D(128,(3,3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Conv2D(256,(3,3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Conv2D(512,(3,3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Conv2D(1024,(3,3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.GlobalAveragePooling2D())
# model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1024))
# model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dense(256))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dense(1))

model.summary()

optimizer = tf.keras.optimizers.Adam()

#%%

epoch_loss_avg = tf.keras.metrics.Mean('train_loss')
train_accuracy = tf.keras.metrics.Accuracy()

epoch_loss_avg_test = tf.keras.metrics.Mean('test_loss')
test_accuracy = tf.keras.metrics.Accuracy()

def train_step(model, images, labels):
    with tf.GradientTape() as t:
        pred = model(images,training=True) #用BatchNormalization要标记trianing

        loss_step = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels, pred)
        #----------------------
        loss_regularization = []
        for p in model.trainable_variables:
            loss_regularization.append(tf.nn.l2_loss(p))
        loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))
        loss_step = loss_step + 0.0001 * loss_regularization  # l1正则化公式？ 可以起到稀释w参数的作用 保留重要特征 不重要特征w为0
        #---------------------------------
    grads = t.gradient(loss_step, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    epoch_loss_avg(loss_step)
    train_accuracy(labels, tf.cast(pred>0, tf.int32))

    # print('lossavg',a)



def test_step(model, images, labels):
    pred = model(images, training=False)

    loss_step = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels, pred)
    epoch_loss_avg_test(loss_step)
    test_accuracy(labels, tf.cast(pred>0, tf.int32))

if not os.path.exists('acc_loss.txt'):
    with open('acc_loss.txt','w') as file:
        pass

try:
    al=pd.read_table('acc_loss.txt',sep=',',header=None)
    for v in al.iloc[len(al)-1:len(al),:].values[0]:
        if v.split(':')[0].strip()=='Epoch':
            epoc=int(v.split(':')[-1])
        if v.split(':')[0].strip() == 'valid loss':
            val_loss = float(v.split(':')[-1])
        if v.split(':')[0].strip() == 'valid accuracy':
            val_acc = float(v.split(':')[-1])
    print(epoc,val_loss,val_acc)

    best_acc=val_acc
    lowest_loss=val_loss
except:
    # best_acc=0
    # lowest_loss=100
    epoc=0
resume=False
if resume:
    model.load_weights(filepath=config.save_model_dir)
    print('导入模型继续训练')
train_loss_results = []
train_acc_results = []
test_loss_results = []
test_acc_results = []
num_epochs=2
print(train_count)
i=0
for epoch in range(epoc,config.EPOCHS):
    step=0
    for imgs_, labels_ in train_dataset:
        step+=1
        train_step(model, imgs_, labels_)

        train_loss_results.append(epoch_loss_avg.result())
        train_acc_results.append(train_accuracy.result().numpy())
        print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1,
                                                                                 config.EPOCHS,
                                                                                 step,
                                                                                 math.ceil(train_count / config.BATCH_SIZE),
                                                                                 epoch_loss_avg.result(),
                                                                                 train_accuracy.result()))

    # print(train_acc_results)
    # mean_train_loss=tf.reduce_mean(train_loss_results)
    # mean_train_acc=tf.reduce_mean(train_acc_results)
    # print(mean_train_acc)


    for imgs_, labels_ in valid_dataset:
        test_step(model, imgs_, labels_)

        test_loss_results.append(epoch_loss_avg_test.result())
        test_acc_results.append(test_accuracy.result())
    # mean_test_loss=tf.reduce_mean(test_loss_results)
    # mean_test_acc=tf.reduce_mean(test_acc_results)

    print('Epoch:{}: test_loss: {:.3f}, test_accuracy: {:.3f}'.format(
        epoch + 1,
        epoch_loss_avg_test.result(),
        test_accuracy.result()
    ))
    with open('acc_loss.txt', 'a') as acc_file:
        acc_file.write(
            'Epoch: {},train loss: {:.5f}, train accuracy: {:.5f},valid loss: {:.5f}, valid accuracy: {:.5f}\n'.format(
                epoch + 1, epoch_loss_avg.result(),
                train_accuracy.result(),
                epoch_loss_avg_test.result(),
                test_accuracy.result()))
    # with open('mean_acc_loss.txt', 'a') as acc_file:
    #     acc_file.write(
    #         'Epoch: {},mean_train loss: {:.5f}, mean_train accuracy: {:.5f},mean_valid loss: {:.5f}, mean_valid accuracy: {:.5f}\n'.format(
    #             epoch + 1, mean_train_loss,
    #             mean_train_acc,
    #             mean_test_loss,
    #             mean_test_acc))

    epoch_loss_avg.reset_states()
    train_accuracy.reset_states()

    epoch_loss_avg_test.reset_states()
    test_accuracy.reset_states()
model.save_weights(filepath=config.save_model_dir, save_format='tf')