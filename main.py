from drivingmode_dataloader import driving_mode_dataloader
import os
import time
import pandas as pd
from dl_models import face_detector, get_capnet
import tensorflow as tf
from utils import get_feature, gpu_limit


def train_fs(dataloader, epochs, num_seq_img, save_path) :

    detector = face_detector('mmod', 0.5)

    fe_model, cf_model = get_capnet(num_seq_img, 0.2)

    # cf_model.load_weights(os.path.join(os.getcwd(), 'best'))
    # print('###################')

    LOSS = tf.keras.losses.CategoricalCrossentropy()
    METRIC = tf.keras.metrics.CategoricalAccuracy()
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.01)

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

        for i, (_, train_x, train_y) in enumerate(train_dataloader) :
            '''
            print(train_x.shape, train_y.shape)

            batch_size = train_x.shape[0]
            img_length = train_x.shape[1]

            downsampling_rate = 10
            num_seq_img = int(img_length/downsampling_rate)

            # make train_input
            # train_input = np.zeros((batch_size, num_seq_img, 224, 224, 3))
            features = np.zeros((batch_size, num_seq_img, 512))

            drop_count = 0

            for b in range(batch_size) :

                crop_list = {}
                crop_list['exist'] = []
                crop_list['img'] = []
                crop_list['conf'] = []

                for l in range(img_length) :
                    rec, _ = detector.get_detection(train_x[b, l, :, :, :])
                    exist, img, conf = crop_detection(train_x[b, l, :, :, :], rec)
                    # img = np.resize(img, (224, 224, 3))
                    if exist :
                        img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
                        cv2.imwrite('./images/{}_{}_{}.jpeg'.format(i, b, l), img)

                    crop_list['exist'].append(exist)
                    crop_list['img'].append(img)
                    crop_list['conf'].append(conf)

                b = b - drop_count

                for j in range(num_seq_img) :
                    ss_images = crop_list['img'][(i * 10):(i * 10) + 10]
                    ss_scores = crop_list['conf'][(i * 10):(i * 10) + 10]
                    ss_exist = crop_list['exist'][(i * 10):(i * 10) + 10]

                    ss_image = []
                    ss_score = []

                    for t in range(len(ss_exist)) :
                        if ss_exist[t] :
                            ss_score.append(ss_scores[t])
                            ss_image.append(ss_images[t])

                    # print(b, i, sum(ss_exist))
                    if sum(ss_exist) == 0 :
                    # if sum(ss_scores) == 0 :
                        # train_input = np.delete(train_input, [b], 0)
                        print(crop_list['exist'])
                        features = np.delete(features, [b], 0)
                        train_y = np.delete(train_y, [b], 0)
                        drop_count += 1
                        break

                    else :
                        idx = ss_score.index(max(ss_score))
                        input_img = ss_image[idx]
                        # cv2.imwrite('./images/{}.jpeg'.format(num), input_img)
                        # train_input[b, i, :, :, :] = input_img
                        input_img = np.expand_dims(input_img, axis=0)
                        feature = fe_model(input_img)
                        features[b, j, :] = feature


                print('####################################')
                print('{}_{}_{}'.format(i, b, l))
                print(features.shape)
                print(train_y.shape)
                print(drop_count)
                print('####################################')
                
            '''
            features, train_y = get_feature(i, detector, fe_model, train_x, train_y, num_seq_img)

            print(train_x.shape, train_y.shape, features.shape)
            if not features.shape[0] == 0 :
                # Training
                with tf.GradientTape() as tape :
                    output = cf_model(features, training=True)
                    loss = LOSS(train_y, output)
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
        for v, (_, val_x, val_y) in enumerate(val_dataloader) :
            val_features, val_y = get_feature(v, detector, fe_model, val_x, val_y, num_seq_img)

            print(val_x.shape, val_y.shape, val_features.shape)


            if not val_features.shape[0] == 0 :
                val_output = cf_model(val_features, training=False)
                loss = LOSS(val_y, val_output)
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
            cf_model.save_weights(os.path.join(save_path, 'weights', 'best@{}'.format(epoch)))

        if epoch > (patience - 1) and max(results['val_metric'][(-1 * (patience + 1)):]) == results['val_metric'][(-1 * (patience + 1))]:
            break
        dataloader.shuffle_data()

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_path, 'train_results.csv'), index=False)

def test_fs(dataloader, save_path) :
    detector = face_detector('mmod', 0.5)

    fe_model, cf_model = get_capnet(num_seq_img, 0.2)

    LOSS = tf.keras.losses.CategoricalCrossentropy()
    METRIC = tf.keras.metrics.CategoricalAccuracy()
    weights_path = os.path.join(save_path, 'weights')
    weights_path = os.path.join(weights_path, os.listdirpath.join(weights_path)[0])

    cf_model.load_weights(weights_path)

    test_results = {}
    test_results['loss'] = []
    test_results['metric'] = []

    for test_num in range(dataloader.get_test_num()) :

        test_dataloader = dataloader.get_test_data(test_num+1)

        for i, (_, test_x, test_y) in enumerate(test_dataloader) :
            print(test_x.shape, test_y.shape)
            test_features, test_y = get_feature(i, detector, fe_model, test_x, test_y, num_seq_img)
            if not test_features.shape[0] == 0 :
                val_output = cf_model(test_features, training=False)
                loss = LOSS(test_y, val_output)
                metric = METRIC(test_y, val_output)

                test_results['loss'].append(loss.numpy())
                test_results['metric'].append(metric.numpy())

    df = pd.DataFrame(test_results)
    df.to_csv(os.path.join(save_path, 'test_results.csv'), index=False)


def main(driver, odometer, data, batch_size, pre_sec, image_size, no_response, epochs, num_seq_img) :
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

    save_path = os.path.join(os.getcwd(), data, driver, str(odometer), 'CAPNet based')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    if data == 'front_image' :
        train_fs(dataloader, epochs, num_seq_img, save_path)
        # test_fs(dataloader, num_seq_img, save_path)


if __name__ == '__main__' :
    gpu_limit(3)

    epochs = 100
    num_seq_img = 6

    # GeesungOh, TaesanKim, EuiseokJeong, JoonghooPark
    # driver = 'TaesanKim'
    driver = 'GeesungOh'

    # 500, 800, 1000, 1500, 2000
    odometer = 500

    # ['can', 'front_image', 'side_image', 'bio', 'audio']
    data = 'front_image'
    # data = 'audio'

    batch_size = 16

    pre_sec = 4
    image_size = 'large'
    no_response='ignore'

    main(driver,
         odometer,
         data,
         batch_size,
         pre_sec,
         image_size,
         no_response,
         epochs,
         num_seq_img)



