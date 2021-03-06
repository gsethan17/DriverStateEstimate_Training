from drivingmode_dataloader_v3 import driving_mode_dataloader
import os
import time
import pandas as pd
import numpy as np
from dl_models import face_detector, get_capnet, get_application
from CAPNet import get_model
import tensorflow as tf
from utils import get_feature, gpu_limit, get_input

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


def train_fs_e2e(modelkey, dataloader, label_weight, epochs, learning_rate, num_seq_img, save_path) :

    detector = face_detector('mmod', 0.5)

    model = get_application(modelkey, num_seq_img)

    # cf_model.load_weights(os.path.join(os.getcwd(), 'best'))
    # print('###################')

    VAL_LOSS = tf.keras.losses.CategoricalCrossentropy()
    LOSS = weighted_cross_entropy
    # LOSS = weighted_loss
    METRIC = tf.keras.metrics.CategoricalAccuracy()
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    patience = 5


    results = {}
    results['train_loss'] = []
    results['train_metric'] = []
    results['val_loss'] = []
    results['val_metric'] = []
    results['val_acc'] = []

    for epoch in range(epochs) :
        st_train = time.time()

        train_dataloader = iter(dataloader.get_train_data)
        val_dataloader = iter(dataloader.get_valid_data)

        temp_results = {}
        temp_results['train_loss'] = []
        temp_results['train_metric'] = []
        temp_results['val_loss'] = []
        temp_results['val_metric'] = []

        for i, ((_, train_x), train_y) in enumerate(train_dataloader) :
            input_, train_y = get_input(detector, train_x, train_y, num_seq_img)

            print(input_.shape, train_y.shape)
            if not input_.shape[0] == 0 :
                # Training
                with tf.GradientTape() as tape :
                    output = model(input_, training=True)
                    loss = LOSS(train_y, output, label_weight)
                    metric = METRIC(train_y, output)

                gradients = tape.gradient(loss, model.trainable_variables)
                OPTIMIZER.apply_gradients(zip(gradients, model.trainable_variables))

                temp_results['train_loss'].append(loss.numpy())
                temp_results['train_metric'].append(metric.numpy())

                print('Train', i, temp_results['train_loss'][-1], temp_results['train_metric'][-1])

        # print(temp_results)
        total_loss = sum(temp_results['train_loss'])
        total_metric = sum(temp_results['train_metric'])
        n_loss = len(temp_results['train_loss'])
        n_metric = len(temp_results['train_metric'])
        print(total_loss, n_loss, total_metric, n_metric)

        results['train_loss'].append(total_loss / n_loss)
        results['train_metric'].append(total_metric / n_metric)
        ed_train = time.time()
        for v, ((_, val_x), val_y) in enumerate(val_dataloader) :
            val_input, val_y = get_input(detector, val_x, val_y, num_seq_img)

            print(input_.shape, train_y.shape)

            trues = np.array([], dtype='int64')
            preds = np.array([], dtype='int64')

            if not val_input.shape[0] == 0 :
                val_output = model(val_input, training=False)
                # loss = LOSS(val_y, val_output, label_weight)
                loss = VAL_LOSS(val_y, val_output)
                metric = METRIC(val_y, val_output)

                trues = np.concatenate([trues, np.argmax(val_y, axis=-1)], axis=0)
                preds = np.concatenate([preds, np.argmax(val_output, axis=-1)], axis=0)

                temp_results['val_loss'].append(loss.numpy())
                temp_results['val_metric'].append(metric.numpy())

                print('Validation', v, temp_results['val_loss'][-1], temp_results['val_metric'][-1])

        tp = 0
        for t in range(len(trues)):
            true = trues[t]
            pred = preds[t]
            if true == pred:
                tp += 1

        acc = tp / len(trues)

        # print(temp_results)
        total_val_loss = sum(temp_results['val_loss'])
        total_val_metric = sum(temp_results['val_metric'])
        n_val_loss = len(temp_results['val_loss'])
        n_val_metric = len(temp_results['val_metric'])
        print(total_val_loss, n_val_loss, total_val_metric, n_val_metric, acc)

        results['val_loss'].append(total_val_loss / n_val_loss)
        results['val_metric'].append(total_val_metric / n_val_metric)
        results['val_acc'].append(acc)
        ed_val = time.time()

        print(
            "{:>3} / {:>3} || train_loss:{:8.4f}, train_metric:{:8.4f}, val_loss:{:8.4f}, val_metric:{:8.4f}, val_acc:{:8.4f} || TIME: Train {:8.1f}sec, Validation {:8.1f}sec".format(
                    epoch + 1, epochs,
                results['train_loss'][-1],
                results['train_metric'][-1],
                results['val_loss'][-1],
                results['val_metric'][-1],
                results['val_acc'][-1],
                (ed_train - st_train),
                (ed_val - ed_train)))

        if results['val_metric'][-1] == max(results['val_metric']) :
            weights_path = os.path.join(save_path, 'weights')
            if not os.path.isdir(weights_path) :
                os.makedirs(weights_path)
            model.save_weights(os.path.join(save_path, 'weights', 'best'))

        if epoch > (patience - 1) and max(results['val_acc'][(-1 * (patience + 1)):]) == results['val_acc'][(-1 * (patience + 1))]:
            break
        dataloader.shuffle_data()

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(save_path, 'train_results.csv'), index=False)


def train_fs(dataloader, label_weight, epochs, learning_rate, num_seq_img, save_path) :

    detector = face_detector('mmod', 0.5)

    fe_model, cf_model = get_capnet(num_seq_img, 0.2)

    # cf_model.load_weights(os.path.join(os.getcwd(), 'best'))
    # print('###################')

    VAL_LOSS = tf.keras.losses.CategoricalCrossentropy()
    LOSS = weighted_cross_entropy
    # LOSS = weighted_loss
    METRIC = tf.keras.metrics.CategoricalAccuracy()
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    patience = 10


    results = {}
    results['train_loss'] = []
    results['train_metric'] = []
    results['val_loss'] = []
    results['val_metric'] = []
    results['val_acc'] = []


    for epoch in range(epochs) :
        st_train = time.time()

        train_dataloader = iter(dataloader.get_train_data)
        val_dataloader = iter(dataloader.get_valid_data)

        temp_results = {}
        temp_results['train_loss'] = []
        temp_results['train_metric'] = []
        temp_results['val_loss'] = []
        temp_results['val_metric'] = []

        for i, ((_, train_x), train_y) in enumerate(train_dataloader) :
            features, train_y = get_feature(i, detector, fe_model, train_x, train_y, num_seq_img)

            print(train_x.shape, train_y.shape, features.shape)
            if not features.shape[0] == 0 :
                # Training
                with tf.GradientTape() as tape :
                    output = cf_model(features, training=True)
                    loss = LOSS(train_y, output, label_weight)
                    metric = METRIC(train_y, output)

                gradients = tape.gradient(loss, cf_model.trainable_variables)
                OPTIMIZER.apply_gradients(zip(gradients, cf_model.trainable_variables))

                temp_results['train_loss'].append(loss.numpy())
                temp_results['train_metric'].append(metric.numpy())

                print('Train', i, temp_results['train_loss'][-1], temp_results['train_metric'][-1])

        print(temp_results)
        total_loss = sum(temp_results['train_loss'])
        total_metric = sum(temp_results['train_metric'])
        n_loss = len(temp_results['train_loss'])
        n_metric = len(temp_results['train_metric'])
        print(total_loss, n_loss, total_metric, n_metric)

        results['train_loss'].append(total_loss / n_loss)
        results['train_metric'].append(total_metric / n_metric)
        ed_train = time.time()
        for v, ((_, val_x), val_y) in enumerate(val_dataloader) :
            val_features, val_y = get_feature(v, detector, fe_model, val_x, val_y, num_seq_img)

            print(val_x.shape, val_y.shape, val_features.shape)

            trues = np.array([], dtype='int64')
            preds = np.array([], dtype='int64')


            if not val_features.shape[0] == 0 :
                val_output = cf_model(val_features, training=False)
                # loss = LOSS(val_y, val_output, label_weight)
                loss = VAL_LOSS(val_y, val_output)
                metric = METRIC(val_y, val_output)

                trues = np.concatenate([trues, np.argmax(val_y, axis=-1)], axis=0)
                preds = np.concatenate([preds, np.argmax(val_output, axis=-1)], axis=0)

                temp_results['val_loss'].append(loss.numpy())
                temp_results['val_metric'].append(metric.numpy())

                print('Validation', v, temp_results['val_loss'][-1], temp_results['val_metric'][-1])

        tp = 0
        for t in range(len(trues)) :
            true = trues[t]
            pred = preds[t]
            if true == pred :
                tp += 1

        acc = tp / len(trues)

        # print(temp_results)
        total_val_loss = sum(temp_results['val_loss'])
        total_val_metric = sum(temp_results['val_metric'])
        n_val_loss = len(temp_results['val_loss'])
        n_val_metric = len(temp_results['val_metric'])
        print(total_val_loss, n_val_loss, total_val_metric, n_val_metric, acc)

        results['val_loss'].append(total_val_loss / n_val_loss)
        results['val_metric'].append(total_val_metric / n_val_metric)
        results['val_acc'].append(acc)
        ed_val = time.time()

        print(
            "{:>3} / {:>3} || train_loss:{:8.4f}, train_metric:{:8.4f}, val_loss:{:8.4f}, val_metric:{:8.4f}, val_acc:{:8.4f} || TIME: Train {:8.1f}sec, Validation {:8.1f}sec".format(
                epoch + 1, epochs,
                results['train_loss'][-1],
                results['train_metric'][-1],
                results['val_loss'][-1],
                results['val_metric'][-1],
                results['val_acc'][-1],
                (ed_train - st_train),
                (ed_val - ed_train)))

        if results['val_metric'][-1] == max(results['val_metric']) :
            weights_path = os.path.join(save_path, 'weights')
            if not os.path.isdir(weights_path) :
                os.makedirs(weights_path)
            cf_model.save_weights(os.path.join(save_path, 'weights', 'best'))

        if epoch > (patience - 1) and max(results['val_acc'][(-1 * (patience + 1)):]) == results['val_acc'][(-1 * (patience + 1))]:
            break
        dataloader.shuffle_data()

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(save_path, 'train_results.csv'), index=False)

def test_fs_e2e(modelkey, dataloader, num_seq_img, save_path) :
    detector = face_detector('mmod', 0.5)

    model = get_application(modelkey, num_seq_img)

    LOSS = tf.keras.losses.CategoricalCrossentropy()
    METRIC = tf.keras.metrics.CategoricalAccuracy()

    results = {}
    results['name'] = []
    results['test_odo'] = []
    results['test_loss'] = []
    results['test_metric'] = []

    # weights load
    weights_path = os.path.join(save_path, 'weights', 'best')
    model.load_weights(weights_path)

    n_tests, test_odos, test_startodos = dataloader.get_test_num()

    for test_num in range(n_tests) :
        st_test = time.time()

        temp_results = {}
        temp_results['loss'] = []
        temp_results['metric'] = []

        name = test_startodos[test_num]

        test_dataloader = dataloader.get_test_data(name)
        # test_dataloader = dataloader.get_test_data(test_num+1)

        flag = False

        for i, ((_, test_x), test_y) in enumerate(test_dataloader) :
            # print(i)
            test_input, test_y = get_input(detector, test_x, test_y, num_seq_img)

            print(test_input.shape, test_y.shape)

            if not test_input.shape[0] == 0 :
                test_output = model(test_input, training=False)
                loss = LOSS(test_y, test_output)
                metric = METRIC(test_y, test_output)

                temp_results['loss'].append(loss.numpy())
                temp_results['metric'].append(metric.numpy())

                if not flag :
                    true = test_y
                    pred = test_output.numpy()
                    flag = True
                    # print(true, pred)
                else :
                    true = np.concatenate((true, test_y), axis = 0)
                    pred = np.concatenate((pred, test_output.numpy()), axis = 0)

        save_results_path = os.path.join(save_path, 'test_results')
        if not os.path.isdir(save_results_path):
            os.makedirs(save_results_path)

        np.save(os.path.join(save_results_path, '{}_{}_{}_true.npy'.format(
            test_num+1, name, test_odos[test_num]
        )), true)
        np.save(os.path.join(save_results_path, '{}_{}_{}_pred.npy'.format(
            test_num+1, name, test_odos[test_num]
        )), pred)


        total_test_loss = sum(temp_results['loss'])
        total_test_metric = sum(temp_results['metric'])
        n_test_loss = len(temp_results['loss'])
        n_test_metric = len(temp_results['metric'])
        # print(total_test_loss, n_test_loss, total_test_metric, n_test_metric)

        results['name'].append(name)
        results['test_odo'].append(test_odos[test_num])
        results['test_loss'].append(total_test_loss / n_test_loss)
        results['test_metric'].append(total_test_metric / n_test_metric)
        ed_test = time.time()

        print(
            "{:>3} / {:>3} || test_loss:{:8.4f}, test_metric:{:8.4f} || TIME: Test {:8.1f}sec".format(
                test_num + 1, n_tests,
                results['test_loss'][-1],
                results['test_metric'][-1],
                (ed_test - st_test)))

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_path, 'test_results.csv'), index=False)

def test_fs(dataloader, num_seq_img, save_path) :
    detector = face_detector('mmod', 0.5)

    fe_model, cf_model = get_capnet(num_seq_img, 0.2)

    LOSS = tf.keras.losses.CategoricalCrossentropy()
    METRIC = tf.keras.metrics.CategoricalAccuracy()

    results = {}
    results['name'] = []
    results['test_odo'] = []
    results['test_loss'] = []
    results['test_metric'] = []

    # weights load
    weights_path = os.path.join(save_path, 'weights', 'best')
    cf_model.load_weights(weights_path)

    n_tests, test_odos, test_startodos = dataloader.get_test_num()

    for test_num in range(n_tests) :
        st_test = time.time()

        temp_results = {}
        temp_results['loss'] = []
        temp_results['metric'] = []

        name = test_startodos[test_num]

        test_dataloader = dataloader.get_test_data(name)
        # test_dataloader = dataloader.get_test_data(test_num+1)

        flag = False

        for i, ((_, test_x), test_y) in enumerate(test_dataloader) :
            # print(i)
            test_features, test_y = get_feature(i, detector, fe_model, test_x, test_y, num_seq_img)
            # print(test_x.shape, test_features.shape, test_y.shape)

            if not test_features.shape[0] == 0 :
                test_output = cf_model(test_features, training=False)
                loss = LOSS(test_y, test_output)
                metric = METRIC(test_y, test_output)

                temp_results['loss'].append(loss.numpy())
                temp_results['metric'].append(metric.numpy())

                if not flag :
                    true = test_y
                    pred = test_output.numpy()
                    flag = True
                    # print(true, pred)
                else :
                    true = np.concatenate((true, test_y), axis = 0)
                    pred = np.concatenate((pred, test_output.numpy()), axis = 0)

        save_results_path = os.path.join(save_path, 'test_results')
        if not os.path.isdir(save_results_path):
            os.makedirs(save_results_path)

        np.save(os.path.join(save_results_path, '{}_{}_{}_true.npy'.format(
            test_num+1, name, test_odos[test_num]
        )), true)
        np.save(os.path.join(save_results_path, '{}_{}_{}_pred.npy'.format(
            test_num+1, name, test_odos[test_num]
        )), pred)


        total_test_loss = sum(temp_results['loss'])
        total_test_metric = sum(temp_results['metric'])
        n_test_loss = len(temp_results['loss'])
        n_test_metric = len(temp_results['metric'])
        # print(total_test_loss, n_test_loss, total_test_metric, n_test_metric)

        results['name'].append(name)
        results['test_odo'].append(test_odos[test_num])
        results['test_loss'].append(total_test_loss / n_test_loss)
        results['test_metric'].append(total_test_metric / n_test_metric)
        ed_test = time.time()

        print(
            "{:>3} / {:>3} || test_loss:{:8.4f}, test_metric:{:8.4f} || TIME: Test {:8.1f}sec".format(
                test_num + 1, n_tests,
                results['test_loss'][-1],
                results['test_metric'][-1],
                (ed_test - st_test)))

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_path, 'test_results.csv'), index=False)


def main(applications, modelkey, driver, odometer, data, batch_size, learning_rate, pre_sec, image_size, no_response, epochs, num_seq_img) :
    data_path = os.path.join(os.getcwd(), 'nas_dms_dataset')
    dataloader = driving_mode_dataloader(
        dataset_path=data_path,
        max_km=odometer,
        driver=driver,
        data=data,
        pre_sec=pre_sec,
        batch_size=batch_size,
        valid_ratio=0.3,
        image_size=image_size,
        no_response=no_response,
    )

    (label_num, label_weight) = dataloader.get_label_weights()

    save_path = os.path.join(os.getcwd(), data, driver, str(odometer), modelkey, str(learning_rate))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    if data == 'front_image' :
        if modelkey in applications :
            train_fs_e2e(modelkey, dataloader, label_weight, epochs, learning_rate, num_seq_img, save_path)
            test_fs_e2e(modelkey, dataloader, num_seq_img, save_path)
        else :
            train_fs(dataloader, label_weight, epochs, learning_rate, num_seq_img, save_path)
            test_fs(dataloader, num_seq_img, save_path)

def get_va(applications, modelkey, driver, odometer, data, batch_size, learning_rate, pre_sec, image_size, no_response, epochs, num_seq_img) :
    data_path = os.path.join(os.getcwd(), 'nas_dms_dataset')
    dataloader = driving_mode_dataloader(
        dataset_path=data_path,
        max_km=odometer,
        driver=driver,
        data=data,
        pre_sec=pre_sec,
        batch_size=batch_size,
        valid_ratio=0,
        image_size=image_size,
        no_response=no_response,
    )
    # return -1
    # get detector
    global detector
    detector = face_detector('mmod', 0.5)

    # get model
    global model
    path_weights = os.path.join(os.getcwd(), 'weights', 'CAPNet_2', 'best_weights')
    model = get_model(key='CAPNet', preTrained=True,
                      weight_path = path_weights,
                      num_seq_image=num_seq_img,
                      input_size=(224, 224, 3),
                      )
    model.build(input_shape=(224, 224, 3))
    print(model.summary())

    save_path = os.path.join(os.getcwd(), 'VA_results', 'matching')

    odo = 9783
    trues, preds = predict_va(iter(dataloader.get_train_data), num_seq_img)
    if len(trues) != 0 :
        np.save(os.path.join(save_path, driver+'_'+str(odo)+'_trues.npy'), trues)
        np.save(os.path.join(save_path, driver+'_'+str(odo)+'_preds.npy'), preds)

    odo = 9808
    trues, preds = predict_va(iter(dataloader.get_valid_data), num_seq_img)
    if len(trues) != 0 :
        np.save(os.path.join(save_path, driver+'_'+str(odo)+'_trues.npy'), trues)
        np.save(os.path.join(save_path, driver+'_'+str(odo)+'_preds.npy'), preds)


    n_tests, test_odos, test_startodos = dataloader.get_test_num()
    # print(n_tests)
    # print(test_odos)
    # print(test_startodos)

    for test_num in range(n_tests) :
        odo = test_startodos[test_num]
        trues, preds = predict_va(dataloader.get_test_data(odo), num_seq_img)
        if len(trues) != 0 :
            np.save(os.path.join(save_path, driver+'_'+str(odo)+'_trues.npy'), trues)
            np.save(os.path.join(save_path, driver+'_'+str(odo)+'_preds.npy'), preds)


def predict_va(sub_dataloader, num_seq_img) :
    count = 0

    for i, ((_, x), y) in enumerate(sub_dataloader) :
        input_, y = get_input(detector, x, y, num_seq_img)
        # print(input_.shape, y.shape)

        if not input_.shape[0] == 0 :
            count += 1
            pred = model(input_, training=False)

            if i == 0 :
                trues = y
                preds = pred.numpy()

            else :
                trues = np.concatenate([trues, y], axis=0)
                preds = np.concatenate([preds, pred.numpy()], axis=0)

            print(trues.shape)
            print(preds.shape)
    if count != 0 :
        return trues, preds

    else :
        return [], []

if __name__ == '__main__' :
    gpu_limit(3)

    epochs = 1
    num_seq_img = 6

    applications = ['mobilenet', 'resnet']

    modelkey = 'CAPNet'
    # modelkey = 'mobilenet'
    # modelkey = applications[1]

    global driver
    # GeesungOh, TaesanKim, EuiseokJeong, JoonghooPark
    driver = 'EuiseokJeong'
    # driver = 'GeesungOh'

    # 500, 800, 1000, 1500, 2000
    odometer = 50

    # ['can', 'front_image', 'side_image', 'bio', 'audio']
    data = 'front_image'
    # data = 'audio'

    batch_size = 16
    learning_rate = 0.001

    pre_sec = 2
    image_size = 'large'
    # image_size = 'small'
    no_response='ignore'

    get_va(applications,
         modelkey,
         driver,
         odometer,
         data,
         batch_size,
         learning_rate,
         pre_sec,
         image_size,
         no_response,
         epochs,
         num_seq_img)



