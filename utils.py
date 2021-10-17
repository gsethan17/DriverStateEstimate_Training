import cv2
import numpy as np
from dl_models import face_detector, crop_detection
import tensorflow as tf

def my_loss_val(trues, preds) :
    mt = np.zeros((4, 4))

    for t in range(len(trues)) :
        true = trues[t]
        pred = preds[t]

        mt[true, pred] += 1

    num = 0.0


    if np.sum(mt[:,3]) != 0 :
        precision_hn = mt[3, 3] / np.sum(mt[:, 3])
        num += 1.
    else :
        precision_hn = 0.0

    if np.sum(mt[0, :]) != 0 :
        recall_ad = mt[0, 0] / np.sum(mt[0, :])
        num += 1.
    else :
        recall_ad = 0.0

    if np.sum(mt[1, :]) != 0:
        recall_es = mt[1, 1] / np.sum(mt[1, :])
        num += 1.
    else:
        recall_es = 0.0

    if np.sum(mt[2, :]) != 0 :
        recall_sf = mt[2, 2] / np.sum(mt[2, :])
        num += 1.
    else :
        recall_sf = 0.0

    if num == 0 :
        loss = 0
    else :
        loss = (precision_hn + recall_ad + recall_es + recall_sf) / num
        # loss = -tf.math.log(loss).numpy()

    return loss


def f1_loss_val(trues, preds):
    mt = np.zeros((4, 4))

    for t in range(len(trues)):
        true = trues[t]
        pred = preds[t]

        mt[true, pred] += 1

    num_pre = 0.0
    num_recall = 0.0

    if np.sum(mt[:, 3]) != 0:
        precision_hn = mt[3, 3] / np.sum(mt[:, 3])
        num_pre += 1.
    else:
        precision_hn = 0.0

    if np.sum(mt[3, :]) != 0:
        recall_hn = mt[3, 3] / np.sum(mt[3, :])
        num_recall += 1.
    else:
        recall_hn = 0.0

    if np.sum(mt[:, 0]) != 0:
        precision_ad = mt[0, 0] / np.sum(mt[:, 0])
        num_pre += 1.
    else:
        precision_ad = 0.0

    if np.sum(mt[0, :]) != 0:
        recall_ad = mt[0, 0] / np.sum(mt[0, :])
        num_recall += 1.
    else:
        recall_ad = 0.0

    if np.sum(mt[:, 1]) != 0:
        precision_es = mt[1, 1] / np.sum(mt[:, 1])
        num_pre += 1.
    else:
        precision_es = 0.0

    if np.sum(mt[1, :]) != 0:
        recall_es = mt[1, 1] / np.sum(mt[1, :])
        num_recall += 1.
    else:
        recall_es = 0.0

    if np.sum(mt[:, 2]) != 0:
        precision_sf = mt[2, 2] / np.sum(mt[:, 2])
        num_pre += 1.
    else:
        precision_sf = 0.0

    if np.sum(mt[2, :]) != 0:
        recall_sf = mt[2, 2] / np.sum(mt[2, :])
        num_recall += 1.
    else:
        recall_sf = 0.0

    if num_pre == 0 or num_recall == 0:
        loss = 0

    else:
        avg_pre = (precision_sf + precision_ad + precision_es + precision_hn) / num_pre
        avg_recall = (recall_sf + recall_ad + recall_es + recall_hn) / num_recall
        loss = 2 * (avg_pre * avg_recall) / (avg_pre + avg_recall)

    return loss


def f1_loss(trues, preds, type = 'majority') :
    mt = np.zeros((4, 4))

    for t in range(len(trues)):
        true = np.argmax(trues[t], axis=-1)
        pred = np.argmax(preds[t], axis=-1)

        mt[true, pred] += 1

    num_pre = 0.0
    num_recall = 0.0

    if type == 'majority' :
        if np.sum(mt[:,3]) != 0 :
            precision_hn = mt[3, 3] / np.sum(mt[:, 3])
            num_pre += 1.
        else :
            precision_hn = 0.0

        if np.sum(mt[3,:]) != 0 :
            recall_hn = mt[3, 3] / np.sum(mt[3, :])
            num_recall += 1.
        else :
            recall_hn = 0.0
    else :
        precision_hn = 0.0
        recall_hn = 0.0

    if np.sum(mt[:, 0]) != 0 :
        precision_ad = mt[0, 0] / np.sum(mt[:, 0])
        num_pre += 1.
    else :
        precision_ad = 0.0

    if np.sum(mt[0, :]) != 0 :
        recall_ad = mt[0, 0] / np.sum(mt[0, :])
        num_recall += 1.
    else :
        recall_ad = 0.0

    if np.sum(mt[:, 1]) != 0:
        precision_es = mt[1, 1] / np.sum(mt[:, 1])
        num_pre += 1.
    else:
        precision_es = 0.0

    if np.sum(mt[1, :]) != 0:
        recall_es = mt[1, 1] / np.sum(mt[1, :])
        num_recall += 1.
    else:
        recall_es = 0.0

    if np.sum(mt[:, 2]) != 0 :
        precision_sf = mt[2, 2] / np.sum(mt[:, 2])
        num_pre += 1.
    else :
        precision_sf = 0.0

    if np.sum(mt[2, :]) != 0 :
        recall_sf = mt[2, 2] / np.sum(mt[2, :])
        num_recall += 1.
    else :
        recall_sf = 0.0

    if num_pre == 0 or num_recall == 0 :
        loss = 0

    else :
        avg_pre = (precision_sf + precision_ad + precision_es + precision_hn) / num_pre
        avg_recall = (recall_sf + recall_ad + recall_es + recall_hn) / num_recall
        loss = 2 * (avg_pre * avg_recall) / (avg_pre + avg_recall + 1e-16)

    return loss

def my_loss(trues, preds) :
    mt = np.zeros((4, 4))

    for t in range(len(trues)):
        true = np.argmax(trues[t], axis=-1)
        pred = np.argmax(preds[t], axis=-1)

        mt[true, pred] += 1

    num = 0.0


    if np.sum(mt[:,3]) != 0 :
        precision_hn = mt[3, 3] / np.sum(mt[:, 3])
        num += 1.
    else :
        precision_hn = 0.0

    if np.sum(mt[0, :]) != 0 :
        recall_ad = mt[0, 0] / np.sum(mt[0, :])
        num += 1.
    else :
        recall_ad = 0.0

    if np.sum(mt[1, :]) != 0:
        recall_es = mt[1, 1] / np.sum(mt[1, :])
        num += 1.
    else:
        recall_es = 0.0

    if np.sum(mt[2, :]) != 0 :
        recall_sf = mt[2, 2] / np.sum(mt[2, :])
        num += 1.
    else :
        recall_sf = 0.0

    if num == 0 :
        loss = 0
    else :
        loss = (precision_hn + recall_ad + recall_es + recall_sf) / num
        # loss = -tf.math.log(loss).numpy()

    return loss

def weighted_f1(true, pred, label_weight) :
    w_ce_loss = weighted_cross_entropy(true, pred, label_weight)
    f1loss = f1_loss(true, pred)
    if f1loss == 0 :
        loss = w_ce_loss
    else :
        f1loss = -tf.math.log(f1loss).numpy()
        loss = w_ce_loss + f1loss

    return loss

def weighted_myloss(true, pred, label_weight) :
    w_ce_loss = weighted_cross_entropy(true, pred, label_weight)
    myloss = my_loss(true, pred)
    if myloss == 0 :
        loss = w_ce_loss
    else :
        myloss = -tf.math.log(myloss).numpy()
        loss = w_ce_loss + myloss

    return loss


def cal_acc(trues, preds) :
    #           0, 1, 2, 3
    # 0
    # 1
    # 2
    # 3
    mt = np.zeros((4, 4))

    for t in range(len(trues)) :
        true = trues[t]
        pred = preds[t]

        mt[true, pred] += 1

    tp = 0

    avg_fp = 0
    avg_fn = 0
    avg_tn = 0

    for i in range(4) :
        # overall
        tp += mt[i, i]

        # average
        avg_fn += np.sum(mt[i, :]) - mt[i, i]
        avg_fp += np.sum(mt[:, i]) - mt[i, i]

        for j in range(4) :
            if not j == i :
                avg_tn += np.sum(mt[j, :]) - mt[j, i]

    overall_acc = tp / np.sum(mt)
    average_acc = (tp + avg_tn) / (tp + avg_tn + avg_fn + avg_fp)

    return overall_acc, average_acc


def macro_soft_f1(y, y_hat, label_weights = [1., 1., 1., 1.]):
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """

    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    label_weights = tf.cast(label_weights, tf.float32)

    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    cost = 1 - soft_f1  # reduce 1 - soft-f1 in order to increase soft-f1
    w_macro_cost = tf.multiply(cost, label_weights)
    macro_cost = tf.reduce_mean(w_macro_cost)  # average on all labels

    return macro_cost

def WE_CE_softf1(true, pred, label_weight) :
    w_ce = weighted_cross_entropy(true, pred, label_weight)
    softf1 = macro_soft_f1(true, pred, label_weight)

    loss = w_ce + softf1

    return loss

def weighted_cross_entropy(true, pred, label_weight) :
    n = true.shape[0]

    nlls = []

    for i in range(n) :
        y = true[i]
        idx = tf.argmax(y)

        p = pred[i, idx]

        nll = -tf.math.log(p)

        weighted_nll = nll * label_weight[idx]

        nlls.append(weighted_nll)

    loss = tf.math.reduce_mean(nlls)

    return loss

def weighted_loss(y_true, y_pred, weight_list):
    cce = tf.keras.losses.CategoricalCrossentropy()
    len_y = len(y_true)
    if not len_y == len(y_pred):
        raise ValueError(f"The length of y{len_y} and prediction{len(y_true)} is not same!")
    total_loss = 0
    for i in range(len_y):
        tmp_loss = cce(y_pred[i], np.float32(y_true[i]))
        total_loss += weight_list[np.argmax(y_true[i])] * tmp_loss
    return total_loss / len_y


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
    window_size = int(img_length / num_seq_img)

    # make train_input
    input_ = np.zeros((batch_size, num_seq_img, 224, 224, 3))

    drop_count = 0

    for b in range(batch_size):

        crop_list = {}
        crop_list['exist'] = []
        crop_list['img'] = []
        crop_list['conf'] = []

        for l in range(img_length):
            # cv2.imwrite('./images/{}_{}.jpeg'.format(b, l), train_x[b, l, :, :, :])
            rec, _ = detector.get_detection(train_x[b, l, :, :, :])
            exist, img, conf = crop_detection(train_x[b, l, :, :, :], rec)
            # img = np.resize(img, (224, 224, 3))
            if exist:
                img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
                # cv2.imwrite('./images/{}_{}.jpeg'.format(b, l), img)
                # normalize
                img = img / 255.

            crop_list['exist'].append(exist)
            crop_list['img'].append(img)
            crop_list['conf'].append(conf)
        b = b - drop_count

        for j in range(num_seq_img):
            ss_images = crop_list['img'][(j * window_size):(j * window_size) + window_size]
            ss_scores = crop_list['conf'][(j * window_size):(j * window_size) + window_size]
            ss_exist = crop_list['exist'][(j * window_size):(j * window_size) + window_size]

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
                # cv2.imwrite('./images/{}_{}.jpeg'.format(b, j), input_img*255.)
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
    window_size = int(img_length / num_seq_img)

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
            ss_images = crop_list['img'][(j * window_size):(j * window_size) + window_size]
            ss_scores = crop_list['conf'][(j * window_size):(j * window_size) + window_size]
            ss_exist = crop_list['exist'][(j * window_size):(j * window_size) + window_size]

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