import os
import glob
import numpy as np
import time
from FER_model.ResNet import ResNet34
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout



def get_model(key='FER-Tuned', preTrained=True,
              weight_path=os.path.join(os.getcwd(), 'FER_model', 'ResNeXt34_Parallel_add',
                                       'checkpoint_4_300000-320739.ckpt'),
              num_seq_image=9, input_size=(224, 224), dropout_rate=0.2):
    if key == 'FER-Tuned':
        # Model load
        model = ResNet34(cardinality=32, se='parallel_add')

        if preTrained:
            # load pre-trained weights
            assert len(glob.glob(weight_path + '*')) > 1, 'There is no weight file | {}'.format(weight_path)
            model.load_weights(weight_path)
            print("[I] The model weights has been load", weight_path)

        return model


    elif key == 'CAPNet':
        # Base model load
        base_model = ResNet34(cardinality=32, se='parallel_add')

        base_weights = os.path.join(os.getcwd(), 'weights', 'FER-Tuned', 'best_weights')
        assert len(glob.glob(base_weights + '*')) > 1, 'There is no weight file | {}'.format(base_weights)
        base_model.load_weights(base_weights)
        print("[I] The cnn architecture weights has been load", base_weights)

        base_model.build(input_shape=(None, input_size[0], input_size[1], 3))
        sub_model = tf.keras.Sequential()
        sub_model.add(tf.keras.Input(shape=(input_size[0], input_size[1], 3)))
        for i in range(6):
            sub_model.add(base_model.layers[i])

        input_ = tf.keras.Input(shape=(num_seq_image, input_size[0], input_size[1], 3))
        for i in range(num_seq_image):
            out_ = sub_model(input_[:, i, :, :, :])

            if i == 0:
                out_0 = tf.expand_dims(out_, axis=1)
            elif i == 1:
                out_1 = tf.expand_dims(out_, axis=1)
                output_ = tf.concat([out_0, out_1], axis=1)
            else:
                out_3 = tf.expand_dims(out_, axis=1)
                output_ = tf.concat([output_, out_3], axis=1)

        lstm = LSTM(256, input_shape=(num_seq_image, 512), dropout=dropout_rate)(output_)

        do1 = Dropout(rate=dropout_rate)(lstm)
        fo1 = Dense(256, activation='tanh')(do1)
        fo2 = Dense(2, activation='tanh')(fo1)

        model = Model(inputs=input_, outputs=fo2)

        if preTrained:
            assert len(glob.glob(weight_path + '*')) > 1, 'There is no weight file | {}'.format(weight_path)
            model.load_weights(weight_path)
            print("[I] The model weights has been load", weight_path)

        for layer in model.layers:
            layer.trainable = False
        model.layers[-1].trainable = True
        model.layers[-2].trainable = True

        return model