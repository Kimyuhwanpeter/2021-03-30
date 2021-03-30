# -*- coding:utf-8 -*-
from random import random, shuffle

import tensorflow as tf
import numpy as np
import os
import datetime
import easydict

FLAGS = easydict.EasyDict({"img_height": 128,
                           
                           "img_width": 88,
                           
                           "tr_txt_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/train.txt",
                           
                           "tr_txt_name": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI_IDList_train.txt",
                           
                           "tr_img_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI/",
                           
                           "te_txt_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/test.txt",
                           
                           "te_txt_name": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI_IDList_test_fix.txt",
                           
                           "te_img_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI/",
                           
                           "batch_size": 64,
                           
                           "epochs": 500,
                           
                           "num_classes": 86,
                           
                           "lr": 0.001,
                           
                           "save_checkpoint": "",
                           
                           "graphs": "", 
                           
                           "train": True,
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": ""})

optim = tf.keras.optimizers.Adam(FLAGS.lr)

def tr_func(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_png(img, 3)
    img = tf.image.resize(img, [FLAGS.img_height, FLAGS.img_width])
    img = tf.image.convert_image_dtype(img, dtype=tf.float32) / 255.

    #lab = lab_list - 2
    n_age = 10
    generation = 0.
    lab = 0.
    if lab_list >= 2 and lab_list < 10:
        generation = 0
        generation = tf.one_hot(generation, 9)
        lab = n_age - (10 - lab_list)
        lab = tf.one_hot(lab, n_age)
    if lab_list >= 10 and lab_list < 20:
        generation = 1
        generation = tf.one_hot(generation, 9)
        lab = n_age - (20 - lab_list)
        lab = tf.one_hot(lab, n_age)
    if lab_list >= 20 and lab_list < 30:
        generation = 2
        generation = tf.one_hot(generation, 9)
        lab = n_age - (30 - lab_list)
        lab = tf.one_hot(lab, n_age)
    if lab_list >= 30 and lab_list < 40:
        generation = 3
        generation = tf.one_hot(generation, 9)
        lab = n_age - (40 - lab_list)
        lab = tf.one_hot(lab, n_age)
    if lab_list >= 40 and lab_list < 50:
        generation = 4
        generation = tf.one_hot(generation, 9)
        lab = n_age - (50 - lab_list)
        lab = tf.one_hot(lab, n_age)
    if lab_list >= 50 and lab_list < 60:
        generation = 5
        generation = tf.one_hot(generation, 9)
        lab = n_age - (60 - lab_list)
        lab = tf.one_hot(lab, n_age)
    if lab_list >= 60 and lab_list < 70:
        generation = 6
        generation = tf.one_hot(generation, 9)
        lab = n_age - (70 - lab_list)
        lab = tf.one_hot(lab, n_age)
    if lab_list >= 70 and lab_list < 80:
        generation = 7
        generation = tf.one_hot(generation, 9)
        lab = n_age - (80 - lab_list)
        lab = tf.one_hot(lab, n_age)
    if lab_list >= 80:
        generation = 8
        generation = tf.one_hot(generation, 9)
        lab = n_age - (90 - lab_list)
        lab = tf.one_hot(lab, n_age)



    return img, lab, generation

def te_func(img, lab):

    img = tf.io.read_file(img)
    img = tf.image.decode_png(img, 3)
    img = tf.image.resize(img, [FLAGS.img_height, FLAGS.img_width])
    img = tf.image.convert_image_dtype(img, dtype=tf.float32) / 255.

    #lab = lab - 2

    return img, lab

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def modified_fea(logits, label, repeat, i):

    logit = logits.numpy()
    logit[label[i]] = logit[label[i]] * (1. - logit[label[i]])
    for j in range(repeat):
        if j != label[i]:
            logit[j] = logit[j] * logit[label[i]]

    return logit

def cal_loss(model, images, age_labels, gener_labels):
    first_loss = 0.
    final_loss = 0.

    with tf.GradientTape(persistent=True) as tape:
        first_logits, final_logits = run_model(model, images, True)
        first_logits, final_logits = tf.nn.softmax(first_logits, 1), tf.nn.softmax(final_logits, 1)

        arg_generation = tf.argmax(gener_labels, 1, tf.int32)
        arg_generation = arg_generation.numpy()

        spec_age = tf.argmax(age_labels, 1, tf.int32)
        spec_age = spec_age.numpy()

        for i in range(FLAGS.batch_size):

            first_logit = first_logits[i]
            ############################################################################################################    이 부분을 추가하면 grad가 None이 됨
            #first_logit = first_logit.numpy()
            #first_logit[arg_generation[i]] = first_logit[arg_generation[i]] * (1. - first_logit[arg_generation[i]])
            #for j in range(9):
            #    if j != arg_generation[i]:
            #        first_logit[j] = first_logit[j] * first_logit[arg_generation[i]]
            first_logit = modified_fea(first_logit, arg_generation, 9, i)
            ############################################################################################################

            first_logit_ = tf.convert_to_tensor(first_logit)
            first_loss += tf.keras.losses.categorical_crossentropy(gener_labels[i], first_logit_) / FLAGS.batch_size

            final_logit = final_logits[i]
            ############################################################################################################
            #final_logit = final_logit.numpy()
            #final_logit[spec_age[i]] = final_logit[spec_age[i]] * (1. - final_logit[spec_age[i]])
            #for j in range(10):
            #    if j != spec_age[i]:
            #        final_logit[j] = final_logit[j] * final_logit[spec_age[i]]
            first_logit = modified_fea(final_logit, spec_age, 10, i)
            ############################################################################################################

            final_logit_ = tf.convert_to_tensor(final_logit)
            final_loss += tf.keras.losses.categorical_crossentropy(age_labels[i], final_logit_) / FLAGS.batch_size

        total_loss = first_loss + final_loss

    #grads += tape.gradient(total_loss, model.trainable_variables)
    grads = tape.gradient(total_loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))
    

    #    first_logits, final_logits = run_model(model, images, True)
    #    total_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(age_labels, final_logits)
    #grads = tape.gradient(total_loss, model.trainable_variables)
    #optim.apply_gradients(zip(grads, model.trainable_variables))        

    return total_loss

def cal_mae(model, images, labels):

    first_logits, final_logits = run_model(model, images, False)

    # 학습할 때 softmax를 계산했던것처럼 똑같이 해주자
    first_logits, final_logits = tf.nn.softmax(first_logits, 1), tf.nn.softmax(final_logits, 1)
    first_arg = tf.argmax(first_logits, 1, tf.int32)
    first_arg = first_arg.numpy()
    final_arg = tf.argmax(final_logits, 1, tf.int32)
    final_arg = final_arg.numpy()

    ae = 0
    for i in range(137):
        first_logit = first_logits[i]
        first_logit = first_logit.numpy()
        first_logit[first_arg[i]] = first_logit[first_arg[i]] * (1. - first_logit[first_arg[i]])
        for j in range(9):
            if j != first_arg[i]:
                first_logit[j] = first_logit[j] * first_logit[first_arg[i]]

        final_logit = final_logits[i]
        final_logit = final_logit.numpy()
        final_logit[final_arg[i]] = final_logit[final_arg[i]] * (1. - final_logit[final_arg[i]])
        for j in range(10):
            if j != final_arg[i]:
                final_logit[j] = final_logit[j] * final_logit[final_arg[i]]

        first_logit = tf.convert_to_tensor(first_logit)
        final_logit = tf.convert_to_tensor(final_logit)

        first_predict = tf.argmax(first_logit, -1, tf.int32) * 10 
        final_predict = tf.argmax(final_logit, -1, tf.int32)
        predict_age = first_predict + final_predict
        
        ae += tf.abs(predict_age - labels.numpy()[i])

    return ae

def main():
    model = tf.keras.applications.ResNet50V2(include_top=False, input_shape=(FLAGS.img_height, FLAGS.img_width, 3), pooling="avg")
    regularizer = tf.keras.regularizers.l2(0.000005)

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
              setattr(layer, attr, regularizer)

    #for layer in model.layers:  # fi = 48
    #    if layer.name == "input_1":
    #        layer = inputs = tf.keras.Input([FLAGS.img_height, FLAGS.img_width, 1])

    #    if layer.name == "conv1_pad":
    #        zero_pad = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(inputs)

    #    if layer.name == "conv1_conv":
    #        #initializer = tf.keras.initializers.GlorotUniform()
    #        #weights = initializer(shape=(3, 3, 1, 48))
    #        #layer.set_weights(np.array(weights, dtype=np.float32))
    #        layer = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1_conv')(zero_pad)

    #model = tf.keras.Model(inputs=inputs, outputs=model.output)
    h = model.output
    first_out = tf.keras.layers.Dense(9)(h) # age generation
    final_out = tf.keras.layers.Dense(10)(first_out)    # spec age
    model = tf.keras.Model(inputs=model.input, outputs=[first_out, final_out])
    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored the latest checkpoint!!")


    if FLAGS.train:
        count = 0;

        tr_img = np.loadtxt(FLAGS.tr_txt_name, dtype="<U100", skiprows=0, usecols=0)
        tr_img = [FLAGS.tr_img_path + img + ".png"for img in tr_img]
        tr_lab = np.loadtxt(FLAGS.tr_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        te_img = np.loadtxt(FLAGS.te_txt_name, dtype="<U100", skiprows=0, usecols=0)
        te_img = [FLAGS.te_img_path + img + ".png" for img in te_img]
        te_lab = np.loadtxt(FLAGS.te_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        te_gener = tf.data.Dataset.from_tensor_slices((te_img, te_lab))
        te_gener = te_gener.map(te_func)
        te_gener = te_gener.batch(137)
        te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

        #############################
        # Define the graphs
        #current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #train_log_dir = FLAGS.graphs + current_time + '/train'
        #train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        #val_log_dir = FLAGS.graphs + current_time + '/val'
        #val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        #############################

        for epoch in range(FLAGS.epochs):
            TR = list(zip(tr_img, tr_lab))
            shuffle(TR)
            tr_img, tr_lab = zip(*TR)
            tr_img, tr_lab = np.array(tr_img), np.array(tr_lab, dtype=np.int32)

            tr_gener = tf.data.Dataset.from_tensor_slices((tr_img, tr_lab))
            tr_gener = tr_gener.shuffle(len(tr_img))
            tr_gener = tr_gener.map(tr_func)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(tr_gener)
            tr_idx = len(tr_img) // FLAGS.batch_size
            for step in range(tr_idx):
                batch_images, batch_labels, batch_age_gener = next(tr_iter)

                loss = cal_loss(model, batch_images, batch_labels, batch_age_gener)

                if count % 10 == 0:
                    print(loss)

                if count % 100 == 0:
                    te_iter = iter(te_gener)
                    te_idx = len(te_img) // 137
                    ae = 0
                    for i in range(te_idx):
                        imgs, labs = next(te_iter)

                        ae += cal_mae(model, imgs, labs)
                        if i % 100 == 0:
                            print("{} mae = {}".format(i + 1, ae / ((i + 1) * 137)))

                    MAE = ae / len(te_img)
                    print("================================")
                    print("step = {}, MAE = {}".format(count, MAE))
                    print("================================")


                count += 1

if __name__ == "__main__":
    main()