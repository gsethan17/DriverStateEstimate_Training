import cv2
import numpy as np
from dl_models import face_detector, crop_detection
import tensorflow as tf

def resize_img(img, h, w) :
    batch_size = img.shape[0]
    n_timewindow = img.shape[1]
    output_ = np.zeros((batch_size, n_timewindow, h, w, 3))

    for i in range(batch_size) :
        for j in range(n_timewindow) :
            output_[i, j, :, :, :] = cv2.resize(img[i, j, :, :, :], (h, w), interpolation=cv2.INTER_AREA)

    return output_

def get_input(detector, train_x, train_y, num_seq_img) :

    batch_size = train_x.shape[0]
    img_length = train_x.shape[1]

    # make train_input
    input_ = np.zeros((batch_size, num_seq_img, 224, 224, 3))

    drop_count = 0

    for b in range(batch_size):

        crop_list = {}
        crop_list['exist'] = []
        crop_list['img'] = []
        crop_list['conf'] = []

        for l in range(img_length):
            rec, _ = detector.get_detection(train_x[b, l, :, :, :])
            exist, img, conf = crop_detection(train_x[b, l, :, :, :], rec)
            # img = np.resize(img, (224, 224, 3))
            if exist:
                img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
                # normalize
                img = img / 255.
                # cv2.imwrite('./images/{}_{}_{}.jpeg'.format(i, b, l), img)

            crop_list['exist'].append(exist)
            crop_list['img'].append(img)
            crop_list['conf'].append(conf)

        b = b - drop_count

        for j in range(num_seq_img):
            ss_images = crop_list['img'][(j * 10):(j * 10) + 10]
            ss_scores = crop_list['conf'][(j * 10):(j * 10) + 10]
            ss_exist = crop_list['exist'][(j * 10):(j * 10) + 10]

            ss_image = []
            ss_score = []

            for t in range(len(ss_exist)):
                if ss_exist[t]:
                    ss_score.append(ss_scores[t])
                    ss_image.append(ss_images[t])

            # print(b, i, sum(ss_exist))
            if sum(ss_exist) == 0:
                # if sum(ss_scores) == 0 :
                # train_input = np.delete(train_input, [b], 0)
                # print(crop_list['exist'])
                input_ = np.delete(input_, [b], 0)
                train_y = np.delete(train_y, [b], 0)
                drop_count += 1
                break

            else:
                idx = ss_score.index(max(ss_score))
                input_img = ss_image[idx]
                # cv2.imwrite('./images/{}.jpeg'.format(num), input_img)
                # train_input[b, i, :, :, :] = input_img
                input_img = np.expand_dims(input_img, axis=0)
                input_[b, j, :] = input_img
        '''
        print('####################################')
        print('{}_{}_{}'.format(i, b, l))
        print(features.shape)
        print(train_y.shape)
        print(drop_count)
        print('####################################')
        '''

        return input_, train_y


def get_feature(i, detector, fe_model, train_x, train_y, num_seq_img) :
    # print(train_x.shape, train_y.shape)

    batch_size = train_x.shape[0]
    img_length = train_x.shape[1]

    # make train_input
    features = np.zeros((batch_size, num_seq_img, 512))

    drop_count = 0

    for b in range(batch_size):

        crop_list = {}
        crop_list['exist'] = []
        crop_list['img'] = []
        crop_list['conf'] = []

        for l in range(img_length):
            rec, _ = detector.get_detection(train_x[b, l, :, :, :])
            exist, img, conf = crop_detection(train_x[b, l, :, :, :], rec)
            # img = np.resize(img, (224, 224, 3))
            if exist:
                img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
                # normalize
                img = img / 255.
                # cv2.imwrite('./images/{}_{}_{}.jpeg'.format(i, b, l), img)

            crop_list['exist'].append(exist)
            crop_list['img'].append(img)
            crop_list['conf'].append(conf)

        b = b - drop_count

        for j in range(num_seq_img):
            ss_images = crop_list['img'][(j * 10):(j * 10) + 10]
            ss_scores = crop_list['conf'][(j * 10):(j * 10) + 10]
            ss_exist = crop_list['exist'][(j * 10):(j * 10) + 10]

            ss_image = []
            ss_score = []

            for t in range(len(ss_exist)):
                if ss_exist[t]:
                    ss_score.append(ss_scores[t])
                    ss_image.append(ss_images[t])

            # print(b, i, sum(ss_exist))
            if sum(ss_exist) == 0:
                # if sum(ss_scores) == 0 :
                # train_input = np.delete(train_input, [b], 0)
                # print(crop_list['exist'])
                features = np.delete(features, [b], 0)
                train_y = np.delete(train_y, [b], 0)
                drop_count += 1
                break

            else:
                idx = ss_score.index(max(ss_score))
                input_img = ss_image[idx]
                # cv2.imwrite('./images/{}.jpeg'.format(num), input_img)
                # train_input[b, i, :, :, :] = input_img
                input_img = np.expand_dims(input_img, axis=0)
                feature = fe_model(input_img)
                features[b, j, :] = feature
        '''
        print('####################################')
        print('{}_{}_{}'.format(i, b, l))
        print(features.shape)
        print(train_y.shape)
        print(drop_count)
        print('####################################')
        '''

    return features, train_y

def gpu_limit(GB) :
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("########################################")
    print('{} GPU(s) is(are) available'.format(len(gpus)))
    print("########################################")
    # set the only one GPU and memory limit
    memory_limit = 1024 * GB
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
            print("Use only one GPU{} limited {}MB memory".format(gpus[0], memory_limit))
        except RuntimeError as e:
            print(e)
    else:
        print('GPU is not available')