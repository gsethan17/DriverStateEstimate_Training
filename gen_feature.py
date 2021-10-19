from drivingmode_dataloader_v4 import driving_mode_dataloader
from utils import gpu_limit, save_img, get_feature
from dl_models import face_detector, get_capnet
import os
import numpy as np

def get_dataloader() :
    data_path = os.path.join(os.getcwd(), 'nas_dms_dataset')

    dataloader = driving_mode_dataloader(
        dataset_path=data_path,
        max_km=odometer,
        driver=driver,
        data=data,
        pre_sec=pre_sec,
        batch_size=batch_size,
        valid_ratio=val_ratio,
        image_size=image_size,
        no_response=no_response,
        get_serial=get_serial
    )

    return dataloader

def main(x, serial, save_img_path, save_feature_path) :
    save_img(x, serial, save_img_path)

    x_sub = x[:, :30, :, :, :]
    # print(x_sub.shape)

    features, serial_sub = get_feature(i, detector, fe_model, x, serial, num_seq_img)
    # print(features.shape, serial_sub.shape)
    # print(serial_sub)

    for j in range(features.shape[0]):
        save_feature = features[j, :, :]
        save_serial = serial_sub[j]

        save_name = os.path.join(save_feature_path, str(save_serial) + '.npy')
        print(save_feature.shape, save_serial)
        np.save(save_name, save_feature)

if __name__ == '__main__' :
    # gpu_limit(3)

    global num_seq_img
    num_seq_img = 6

    global driver
    # driver = 'GeesungOh'
    driver = 'TaesanKim'

    global val_ratio
    val_ratio = 0.0

    global odometer
    odometer = 500

    global data
    data = 'front_image'

    global batch_size
    batch_size = 32

    global pre_sec
    pre_sec = 4

    global image_size
    image_size = 'large'

    global no_response
    no_response = 'ignore'

    global get_serial
    get_serial = True

    save_img_path= os.path.join(os.getcwd(), 'save_img', 'front')
    save_feature_path = os.path.join(os.getcwd(), 'save_feature', 'front')
    if not os.path.isdir(save_img_path):
        os.makedirs(save_img_path)
    if not os.path.isdir(save_feature_path):
        os.makedirs(save_feature_path)


    loader = get_dataloader()


    detector = face_detector('mmod', 0.5)
    fe_model, cf_model = get_capnet(num_seq_img, 0.2)


    train_loader = iter(loader.get_train_data)
    val_loader = iter(loader.get_valid_data)

    # for i, ((_, x), y, serial) in enumerate(train_loader) :
    #     main(x, serial, save_img_path, save_feature_path)


    for i, ((_, x), y, serial) in enumerate(val_loader) :
        main(x, serial, save_img_path, save_feature_path)


    n_tests, test_odos, test_startodos = loader.get_test_num()
    for test_num in range(n_tests) :
        name = test_startodos[test_num]
        test_dataloader = loader.get_test_data(name)

        for i, ((_, x), y, serial) in enumerate(test_dataloader) :
            main(x, serial, save_img_path, save_feature_path)
