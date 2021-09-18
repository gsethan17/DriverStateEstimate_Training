from drivingmode_dataloader_v3 import driving_mode_dataloader
import os
import time
import pandas as pd
import numpy as np
from dl_models import face_detector, get_capnet
import tensorflow as tf
from utils import get_feature, gpu_limit

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




def train_fs(dataloader, label_weight, epochs, learning_rate, num_seq_img, save_path) :

    detector = face_detector('mmod', 0.5)

    fe_model, cf_model = get_capnet(num_seq_img, 0.2)

    # cf_model.load_weights(os.path.join(os.getcwd(), 'best'))
    # print('###################')

    # LOSS = tf.keras.losses.CategoricalCrossentropy()
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


            if not val_features.shape[0] == 0 :
                val_output = cf_model(val_features, training=False)
                loss = LOSS(val_y, val_output, label_weight)
                metric = METRIC(val_y, val_output)

                temp_results['val_loss'].append(loss.numpy())
                temp_results['val_metric'].append(metric.numpy())

                print('Validation', v, temp_results['val_loss'][-1], temp_results['val_metric'][-1])


        print(temp_results)
        total_val_loss = sum(temp_results['val_loss'])
        total_val_metric = sum(temp_results['val_metric'])
        n_val_loss = len(temp_results['val_loss'])
        n_val_metric = len(temp_results['val_metric'])
        print(total_val_loss, n_val_loss, total_val_metric, n_val_metric)

        results['val_loss'].append(total_val_loss / n_val_loss)
        results['val_metric'].append(total_val_metric / n_val_metric)
        ed_val = time.time()

        print(
            "{:>3} / {:>3} || train_loss:{:8.4f}, train_metric:{:8.4f}, val_loss:{:8.4f}, val_metric:{:8.4f} || TIME: Train {:8.1f}sec, Validation {:8.1f}sec".format(
                epoch + 1, epochs,
                results['train_loss'][-1],
                results['train_metric'][-1],
                results['val_loss'][-1],
                results['val_metric'][-1],
                (ed_train - st_train),
                (ed_val - ed_train)))

        if results['val_metric'][-1] == max(results['val_metric']) :
            weights_path = os.path.join(save_path, 'weights')
            if not os.path.isdir(weights_path) :
                os.makedirs(weights_path)
            cf_model.save_weights(os.path.join(save_path, 'weights', 'best'))

        if epoch > (patience - 1) and max(results['val_metric'][(-1 * (patience + 1)):]) == results['val_metric'][(-1 * (patience + 1))]:
            break
        dataloader.shuffle_data()

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(save_path, 'train_results.csv'), index=False)

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


def main(driver, odometer, data, batch_size, learning_rate, pre_sec, image_size, no_response, epochs, num_seq_img) :
    data_path = os.path.join(os.getcwd(), 'nas_dms_dataset')
    dataloader = driving_mode_dataloader(
        dataset_path=data_path,
        max_km=odometer,
        driver=driver,
        data=data,
        pre_sec=pre_sec,
        batch_size=batch_size,
        valid_ratio=0.2,
        image_size=image_size,
        no_response=no_response,
    )

    (label_num, label_weight) = dataloader.get_label_weights()

    save_path = os.path.join(os.getcwd(), data, driver, str(odometer), 'CAPNet_BN', str(learning_rate))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    if data == 'front_image' :
        train_fs(dataloader, label_weight, epochs, learning_rate, num_seq_img, save_path)
        test_fs(dataloader, num_seq_img, save_path)


if __name__ == '__main__' :
    gpu_limit(3)

    epochs = 100
    num_seq_img = 6

    # GeesungOh, TaesanKim, EuiseokJeong, JoonghooPark
    driver = 'TaesanKim'
    # driver = 'GeesungOh'

    # 500, 800, 1000, 1500, 2000
    odometer = 500

    # ['can', 'front_image', 'side_image', 'bio', 'audio']
    data = 'front_image'
    # data = 'audio'

    batch_size = 16
    learning_rate = 0.001

    pre_sec = 4
    image_size = 'large'
    no_response='ignore'

    main(driver,
         odometer,
         data,
         batch_size,
         learning_rate,
         pre_sec,
         image_size,
         no_response,
         epochs,
         num_seq_img)



