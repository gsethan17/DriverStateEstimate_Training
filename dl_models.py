import dlib
import os
import time
from FER_model.ResNet import ResNet34
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications.mobilenet import MobileNet

def get_mobilenet(num_seq_image, dropout_rate=0.001) :
    base_model = MobileNet(include_top=False, pooling='avg')

    input_ = Input(shape=(num_seq_image, 224, 224, 3))

    for i in range(num_seq_image):
        out_ = base_model(input_[:, i, :, :, :])

        if i == 0:
            out_0 = tf.expand_dims(out_, axis=1)
        elif i == 1:
            out_1 = tf.expand_dims(out_, axis=1)
            output_ = tf.concat([out_0, out_1], axis=1)
        else:
            out_3 = tf.expand_dims(out_, axis=1)
            output_ = tf.concat([output_, out_3], axis=1)

    lstm1 = LSTM(256, input_shape=(num_seq_image, output_.shape[-1]), return_sequences=True,
                 dropout=dropout_rate)(output_)
    lstm2 = LSTM(256, dropout=dropout_rate)(lstm1)

    fo2 = Dense(4, activation='softmax')(lstm2)

    model = Model(inputs=input_, outputs=fo2)

    return model



def get_capnet(num_seq_image, dropout_rate) :
    model = ResNet34(cardinality=32, se='parallel_add')
    fer_weight_path = os.path.join(os.getcwd(), 'weights', 'FER-Tuned', 'best_weights')
    model.load_weights(fer_weight_path)

    model.build(input_shape=(None, 224, 224, 3))
    # print(model.summary())
    input_ = Input(shape=(224, 224, 3))

    for i in range(6) :

        if i == 0 :
            fe = model.layers[i](input_)
        else :
            fe = model.layers[i](fe)

    fe_model = Model(inputs=input_, outputs=fe)

    # print(fe_model.summary())
    '''
    # V-A prediction model
    input_ = Input(shape=(num_seq_image, 224, 224, 3))
    for i in range(num_seq_image):
        out_ = fe_model(input_[:, i, :, :, :])

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

    weight_path = os.path.join(os.getcwd(), 'weights', 'CAPNet_2', 'best_weights')
    model.load_weights(weight_path)

    # print(model.summary())


    input_ = Input(shape=(num_seq_image, 512))
    for i in range(-4, 0):

        if i == -4:
            fe = model.layers[i](input_)
        else:
            fe = model.layers[i](fe)

    ce_model = Model(inputs=input_, outputs=fe)
    '''
    # print(ce_model.summary())


    input_ = Input(shape=(num_seq_image, 512))
    bn = BatchNormalization()(input_)
    lstm = LSTM(256, input_shape=(num_seq_image, 512), dropout=dropout_rate)(bn)

    do1 = Dropout(rate=dropout_rate)(lstm)
    fo1 = Dense(256, activation='tanh')(do1)
    fo2 = Dense(4, activation='softmax')(fo1)

    cf_model = Model(inputs=input_, outputs=fo2)
    print(cf_model.summary())


    return fe_model, cf_model


class face_detector():
    def __init__(self, mode, THRESHOLD=0.0):
        self.mode = mode
        self.threshold = THRESHOLD
        self.detector = self.load_detector()
        self.st_time = 0
        self.duration = 0
        self.rectangles = {}
        self.rectangles['count'] = 0
        self.rectangles['lefttop'] = []
        self.rectangles['rightbottom'] = []
        self.rectangles['confidence'] = []

    def reset(self):
        self.rectangles = {}
        self.rectangles['count'] = 0
        self.rectangles['lefttop'] = []
        self.rectangles['rightbottom'] = []
        self.rectangles['confidence'] = []
        self.st_time = 0
        self.duration = 0

    def load_detector(self):
        if self.mode == 'mmod':
            config_path = './src/mmod/mmod_human_face_detector.dat'
            detector = dlib.cnn_face_detection_model_v1(config_path)

        else:
            print('mode is not valid.')
            return -1

        return detector

    def get_detection(self, img):
        self.reset()
        self.st_time = time.time()

        if self.mode == 'mmod':
            detections = self.detector(img, 0)

            for detection in detections:
                confidence = detection.confidence
                if confidence > self.threshold:
                    self.rectangles['count'] += 1
                    self.rectangles['lefttop'].append((detection.rect.left(), detection.rect.top()))
                    self.rectangles['rightbottom'].append((detection.rect.right(), detection.rect.bottom()))
                    self.rectangles['confidence'].append(confidence)

        duration = time.time() - self.st_time
        return self.rectangles, duration


def crop_detection(img, rectangles):
    if rectangles['confidence']:
        max_idx = rectangles['confidence'].index(max(rectangles['confidence']))
        left, top = rectangles['lefttop'][max_idx]
        right, bottom = rectangles['rightbottom'][max_idx]
        conf = rectangles['confidence'][max_idx]

        img_crop = img[top:bottom, left:right].copy()

        return True, img_crop, conf

    else:
        return False, None, None



if __name__ == '__main__' :
    num_seq_image = 6
    dropout_rate = 0.2

    model = get_mobilenet((num_seq_image))
    print(model.summary())

    # fe_model, ce_model = get_capnet(num_seq_image, dropout_rate)
    #
    # print(fe_model.summary())
    # print(ce_model.summary())