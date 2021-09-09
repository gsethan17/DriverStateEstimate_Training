import numpy as np
import cv2, os, random
import pandas as pd
import tensorflow as tf
import time
from glob import glob
from scipy.io.wavfile import read


class driving_mode_dataloader(tf.keras.utils.Sequence):
    def __init__(self, dataset_path, max_km, driver, data, pre_sec, batch_size=32, valid_ratio=1, image_size='big', no_response='ignore'):
        self.dms_dataset_path = dataset_path
        self.batch_size = batch_size
        self.driver = driver
        self.data = data
        self.image_size = image_size
        self.per_sec = self.get_per_sec()
        desired_data_list = ['can', 'front_image', 'side_image', 'bio', 'audio']
        if data not in desired_data_list:
            raise ValueError(f"{data} is not valid data_option. data_list must be one of {desired_data_list}")
        self.pre_sec = pre_sec
        self.driver_path = os.path.join(self.dms_dataset_path, self.driver)
        self.x_dict = {}
        self.no_response = no_response
        self.total_data_list = self.get_total_data_list()
        self.train_data_list, self.valid_data_list, self.test_data_list = self.get_data_list(max_km=max_km, valid_ratio=valid_ratio)
        self.train_indices = self.get_indices(self.train_data_list)
        self.valid_indices = self.get_indices(self.valid_data_list)
        self.test_indices = self.get_indices(self.test_data_list)
        self.get_data_info()

    def get_data(self, odd, st_index, end_index, status):
        if self.data == 'front_image':
            color_x, ir_x = self.get_image_np(view='front', odd=odd, st_index=st_index, end_index=end_index)
            return color_x, ir_x
        elif self.data == 'side_image':
            color_x, ir_x = self.get_image_np(view='side', odd=odd, st_index=st_index, end_index=end_index)
            return color_x, ir_x
        else:
            if odd in self.x_dict.keys():
                x = self.x_dict[odd][st_index:end_index]
            else:
                self.x_dict = {}
                if self.data == 'can':
                    can_df = pd.read_csv(glob(os.path.join(self.driver_path, odd, 'CAN', '*.csv'))[0], low_memory=False)
                    can_df = self.preprocess_can(can_df, replace=True)
                    self.x_dict[odd] = np.array(can_df.drop(['timestamp'], axis=1))
                    x = self.x_dict[odd][st_index:end_index]
                if self.data == 'bio':
                    bio_df = self.get_bio_df(odd)
                    self.x_dict[odd] = np.array(bio_df.drop(['timestamp'], axis=1))
                    x = self.x_dict[odd][st_index:end_index]
                if self.data == 'audio':
                    _, audio_np = read(glob(os.path.join(self.driver_path, odd, 'audio', '*.wav'))[0])
                    self.x_dict[odd] = audio_np
                    x = self.x_dict[odd][st_index:end_index]
            return x

    def get_image_np(self, view, odd, st_index, end_index):
        color_img_list, ir_img_list = [], []
        for index in range(st_index, end_index):
            color_frame, ir_frame = self.get_one_frame(view, odd, index)
            color_img_list.append(color_frame)
            ir_img_list.append(ir_frame)
        return np.stack(color_img_list), np.stack(ir_img_list)

    def get_one_frame(self, view, odd, frame_num):
        if view == 'front':
            video_path = os.path.join(self.dms_dataset_path, self.driver, odd, 'video', 'FrontView')
            if f"ir_{odd}" in self.x_dict.keys():
                color_cap = self.x_dict[f"color_{odd}"]
                ir_cap = self.x_dict[f"ir_{odd}"]
                color_cap.set(1, frame_num)
                ir_cap.set(1, frame_num)
                color_ret, color_frame = color_cap.read()
                ir_ret, ir_frame = ir_cap.read()
            else:
                self.x_dict = {}
                color_cap = cv2.VideoCapture(os.path.join(video_path, f'color_{view}2.avi'))
                ir_cap = cv2.VideoCapture(os.path.join(video_path, f'ir_{view}2.avi'))
                self.x_dict[f"color_{odd}"] = color_cap
                self.x_dict[f"ir_{odd}"] = ir_cap
                color_cap.set(1, frame_num)
                ir_cap.set(1, frame_num)
                color_ret, color_frame = color_cap.read()
                ir_ret, ir_frame = ir_cap.read()
        elif view == 'side':
            video_path = os.path.join(self.dms_dataset_path, self.driver, odd, 'video', 'SideView')
            if f"ir_{odd}" in self.x_dict.keys():
                # print('cap key existing')
                color_cap = self.x_dict[f"color_{odd}"]
                ir_cap = self.x_dict[f"ir_{odd}"]
                color_cap.set(1, frame_num)
                ir_cap.set(1, frame_num)
                color_ret, color_frame = color_cap.read()
                ir_ret, ir_frame = ir_cap.read()
            else:
                self.x_dict = {}
                color_cap = cv2.VideoCapture(os.path.join(video_path, f'color_{view}2.avi'))
                ir_cap = cv2.VideoCapture(os.path.join(video_path, f'ir_{view}2.avi'))
                self.x_dict[f"color_{odd}"] = color_cap
                self.x_dict[f"ir_{odd}"] = ir_cap
                color_cap.set(1, frame_num)
                ir_cap.set(1, frame_num)
                color_ret, color_frame = color_cap.read()
                ir_ret, ir_frame = ir_cap.read()
        else:
            raise ValueError(f'View mode {view} is not valid')
        if ir_ret and color_ret:
            return (color_frame, ir_frame)
        else:
            raise ValueError(f"There is no frame {self.driver}/{odd}/{view}/{frame_num}")

    def get_indices(self, odd_list):
        result_indices = []
        for i, odd in enumerate(odd_list):
            print('\r', f'[INFO] ({i + 1}/{len(odd_list)}) Extracting indices {self.driver}/{odd}', end='')
            tmp_indices = self.get_odd_indices(odd)
            result_indices += tmp_indices
        print()
        return result_indices

    def get_odd_indices(self, odd):
        label_df = pd.read_csv(glob(os.path.join(self.driver_path, odd, 'HMI', '*.csv'))[0])

        def one_hot(x):
            if x == 0:
                return np.nan
            if x == 1:
                return [1, 0, 0, 0]
            if x == 2:
                return [0, 1, 0, 0]
            if x == 3:
                return [0, 0, 1, 0]
            if x == 4:
                return [0, 0, 0, 1]

        label_df['status'] = label_df['status'].apply(one_hot)
        if self.no_response == 'ffill':
            label_df['status'] = label_df['status'].fillna(method='ffill')

        label_df = label_df.dropna(axis=0)
        result = []
        label_time_indices = [(odd, label_df['time'].iloc[i], label_df['status'].iloc[i]) for i in range(len(label_df))]
        if self.data == 'audio':
            audio_time = pd.read_csv(glob(os.path.join(self.driver_path, odd, 'audio', '*.csv'))[0])
            data_time_np = np.array(audio_time)
        if self.data == 'can':
            # load can_df and can_time_np
            can_df = pd.read_csv(glob(os.path.join(self.driver_path, odd, 'CAN', '*.csv'))[0], low_memory=False)
            can_df = self.preprocess_can(can_df, replace=True)
            data_time_np = can_df['timestamp']
        if self.data == 'bio':
            bio_df = self.get_bio_df(odd)
            data_time_np = np.array(bio_df['timestamp'])
        if self.data == 'front_image' or self.data == 'side_image':
            video_df = pd.read_csv(glob(os.path.join(self.driver_path, odd, 'video', '*.csv'))[0])
            data_time_np = np.array(video_df['timestamp'])
        for odd, tmp_time, status in label_time_indices:
            try:
                tmp_index = np.where(data_time_np < tmp_time)[0][-1]
                if self.data == 'audio':
                    tmp_index = tmp_index * 1024
                result.append((odd, (tmp_index - int(self.pre_sec * self.per_sec[self.data]), tmp_index), status))
            except:
                pass
        result = [(odd, (st_index, end_index), status) for (odd, (st_index, end_index), status) in result if st_index >= 0]
        return result

    # get bio_df and interpolate
    def get_bio_df(self, odd):
        bio_df = pd.DataFrame(columns=['timestamp'])
        bio_path = os.path.join(self.driver_path, odd, 'bio')
        for bio_data in ['EDA', 'BVP', 'HR']:
            df = pd.read_csv(os.path.join(bio_path, f'{bio_data}.csv'), header=None)
            st_time = float(df.iloc[0])
            time_per_data = 1 / int(df.iloc[1])
            df = np.array(df.iloc[3:])
            result_np = np.full((len(df), 2), -1, dtype=np.float64)
            for i in range(len(df)):
                if i == 0:
                    result_np[i] = np.array([st_time, df[i]], dtype=np.float64)
                else:
                    result_np[i] = np.array([st_time + time_per_data * i, df[i]], dtype=np.float64)
            df = pd.DataFrame(result_np, columns=['timestamp', bio_data])
            bio_df = pd.concat([bio_df, df]).sort_values(by='timestamp').reset_index(drop=True)
        bio_df = bio_df.interpolate()
        bio_df = bio_df.fillna(method='bfill')
        return bio_df

    # check data and get total data list
    def get_total_data_list(self):
        data_list = [x for x in os.listdir(self.driver_path) if len(x.split('.')) == 1]
        invalid_list = []
        for data in data_list:
            if self.check_data(data) == False:
                invalid_list.append(data)
        data_list = [x for x in data_list if x not in invalid_list]
        if len(data_list) <= 2:
            raise ValueError(f"The number of data is less than 3 ({len(data_list)})")

        int_data_list = [x.split('_')[0] for x in data_list]
        int_data_list = [int(x) for x in int_data_list]
        int_data_list.sort()

        result_list = []
        for i in int_data_list:
            for j in data_list:
                if j.split('_')[0] == str(i):
                    result_list.append(j)
        return result_list

    # split total_data_list into train, valid, test
    def get_data_list(self, max_km, valid_ratio):
        total_odd = 0
        train_data_list, valid_data_list, test_data_list = [], [], []
        for odd in self.total_data_list:
            total_odd += self.get_data_km(odd)
            if total_odd >= max_km:
                break
            else:
                train_data_list.append(odd)
        test_data_list = [x for x in self.total_data_list if x not in train_data_list]
        valid_num = int(len(train_data_list) * valid_ratio)
        if valid_num == 0:
            raise ValueError(f"The number of validation number is {valid_num}. You should set more max_km or less valid_ratio")
        valid_data_list = train_data_list[-valid_num:]
        train_data_list = [x for x in train_data_list if x not in valid_data_list]
        if len(train_data_list) == 0:
            raise ValueError(f'The length of train_data_list is {len(train_data_list)}')
        if len(valid_data_list) == 0:
            raise ValueError(f'The length of valid_data_list is {len(valid_data_list)}')
        if len(test_data_list) == 0:
            raise ValueError(f'The length of test_data_list is {len(test_data_list)}')
        return train_data_list, valid_data_list, test_data_list

    # get odometry each directory
    def get_data_km(self, odd):
        data_path = os.path.join(self.driver_path, odd)
        csv_list = list(filter(lambda x: 'csv' in x.split('.'), os.listdir(data_path)))
        if len(csv_list) != 1:
            print(f'The number of csv file({len(csv_list)}) is not one')
        km_path = os.path.join(os.path.join(self.driver_path, odd, csv_list[0]))
        try:
            km_df = pd.read_csv(km_path)
        except:
            raise ValueError(f'There is no odd df in {km_path}')

        return (km_df['END'] - km_df['START'])[0]

    def shuffle_data(self):
        tmp_dict = {}
        for odd, indices, status in self.train_indices:
            if odd in tmp_dict.keys():
                tmp_dict[odd].append((odd, indices, status))
            else:
                tmp_dict[odd] = [(odd, indices, status)]
        for odd in tmp_dict.keys():
            random.shuffle(tmp_dict[odd])
        result_list = []
        for odd in tmp_dict.keys():
            for data in tmp_dict[odd]:
                result_list.append(data)
        self.train_indices = result_list

        tmp_dict = {}
        for odd, indices, status in self.valid_indices:
            if odd in tmp_dict.keys():
                tmp_dict[odd].append((odd, indices, status))
            else:
                tmp_dict[odd] = [(odd, indices, status)]
        for odd in tmp_dict.keys():
            random.shuffle(tmp_dict[odd])
        result_list = []
        for odd in tmp_dict.keys():
            for data in tmp_dict[odd]:
                result_list.append(data)
        self.valid_indices = result_list

    # print the number of directory and total odometry
    def get_data_info(self):
        train_km = 0
        for odd in self.train_data_list:
            train_km += self.get_data_km(odd)
        valid_km = 0
        for odd in self.valid_data_list:
            valid_km += self.get_data_km(odd)
        test_km = 0
        for odd in self.test_data_list:
            test_km += self.get_data_km(odd)
        print(
            f"[INFO] Total odometry || Train({len(self.train_data_list)}): {train_km}km, Valid({len(self.valid_data_list)}): {valid_km}km, Test({len(self.test_data_list)}): {test_km}km")

    @property
    def get_train_data(self):
        if self.data in ['side_image', 'front_image']:
            color_list = []
            ir_list = []
            y_list = []
            for i, (odd, (st_index, end_index), status) in enumerate(self.train_indices):
                color_x, ir_x = self.get_data(odd, st_index, end_index, status)
                color_list.append(color_x)
                ir_list.append(ir_x)
                y_list.append(status)
                if (i + 1) % self.batch_size == 0:
                    if len(color_list) != 0:
                        yield np.stack(color_list), np.stack(ir_list), np.stack(y_list)
                        color_list = []
                        ir_list = []
                        y_list = []
                if i == len(self.train_indices) - 1:
                    if len(color_list) != 0:
                        yield np.stack(color_list), np.stack(ir_list), np.stack(y_list)
        else:
            x_list = []
            y_list = []
            for i, (odd, (st_index, end_index), status) in enumerate(self.train_indices):
                x_list.append(self.get_data(odd, st_index, end_index, status))
                y_list.append(status)
                if (i + 1) % self.batch_size == 0:
                    if len(x_list) != 0:
                        yield np.stack(x_list), np.stack(y_list)
                        x_list = []
                        y_list = []
                if i == len(self.train_indices) - 1:
                    if len(x_list) != 0:
                        yield np.stack(x_list), np.stack(y_list)

    @property
    def get_valid_data(self):
        if self.data in ['side_image', 'front_image']:
            color_list = []
            ir_list = []
            y_list = []
            for i, (odd, (st_index, end_index), status) in enumerate(self.valid_indices):
                color_x, ir_x = self.get_data(odd, st_index, end_index, status)

                color_list.append(color_x)
                ir_list.append(ir_x)
                y_list.append(status)
                if (i + 1) % self.batch_size == 0:
                    yield np.stack(color_list), np.stack(ir_list), np.stack(y_list)
                    color_list = []
                    ir_list = []
                    y_list = []
                if i == len(self.valid_indices) - 1:
                    yield np.stack(color_list), np.stack(ir_list), np.stack(y_list)
        else:
            x_list = []
            y_list = []
            for i, (odd, (st_index, end_index), status) in enumerate(self.valid_indices):
                x_list.append(self.get_data(odd, st_index, end_index, status))
                y_list.append(status)
                if i != 0 and (i + 1) % self.batch_size == 0:
                    yield np.stack(x_list), np.stack(y_list)
                    x_list = []
                    y_list = []
                if i == len(self.valid_indices) - 1:
                    yield np.stack(x_list), np.stack(y_list)

    def get_test_num(self):
        tmp = 0
        test_km_list = []
        for i in self.test_data_list:
            tmp += self.get_data_km(i)
            test_km_list.append(tmp)
        return len(self.test_data_list), test_km_list

    def get_test_data(self, test_num):
        tmp_odd = self.test_data_list[0]
        test_num_cnt = 0
        # print test_odometry
        test_odd = 0
        for i, test_data in enumerate(self.test_data_list):
            test_odd += self.get_data_km(test_data)
            if i + 1 == test_num:
                print(f"[INFO] {test_num} Test data: {test_odd}km")
                break

        if self.data in ['side_image', 'front_image']:
            color_list = []
            ir_list = []
            y_list = []
            for i, (odd, (st_index, end_index), status) in enumerate(self.test_indices):
                if tmp_odd != odd:
                    tmp_odd = odd
                    test_num_cnt += 1
                if test_num_cnt == test_num:
                    if len(color_list) != 0:
                        yield np.stack(color_list), np.stack(ir_list), np.stack(y_list)
                    break

                color_x, ir_x = self.get_data(odd, st_index, end_index, status)
                # print(((i) % self.batch_size), (odd, (st_index, end_index), status))

                color_list.append(color_x)
                ir_list.append(ir_x)
                y_list.append(status)
                if (i + 1) % self.batch_size == 0:
                    # try:
                    yield np.stack(color_list), np.stack(ir_list), np.stack(y_list)
                    color_list = []
                    ir_list = []
                    y_list = []
                if i == len(self.test_indices) - 1:
                    if len(color_list) != 0:
                        yield np.stack(color_list), np.stack(ir_list), np.stack(y_list)

        else:
            x_list = []
            y_list = []
            for i, (odd, (st_index, end_index), status) in enumerate(self.test_indices):
                if tmp_odd != odd:
                    tmp_odd = odd
                    test_num_cnt += 1
                if test_num_cnt == test_num:
                    if len(x_list) != 0:
                        try:
                            yield np.stack(x_list), np.stack(y_list)
                        except:
                            pass
                    break
                x_list.append(self.get_data(odd, st_index, end_index, status))
                y_list.append(status)
                if i != 0 and (i + 1) % self.batch_size == 0:
                    if len(x_list) != 0:
                        try:
                            yield np.stack(x_list), np.stack(y_list)
                        except:
                            pass
                        x_list = []
                        y_list = []

                if i == len(self.test_indices) - 1:
                    if len(x_list) != 0:
                        yield np.stack(x_list), np.stack(y_list)

    def check_data(self, data):
        result = True
        result_list = []
        data_dir = os.path.join(self.dms_dataset_path, self.driver, data)
        if self.data in data.split('_'):
            result = False
        HMI_list = os.listdir(os.path.join(data_dir, 'HMI'))
        if len(HMI_list) != 1:
            result = False
            result_list.append(f'HMI({int(len(HMI_list))})')
        odd_df_list = list(filter(lambda x: 'csv' in x.split('.'), os.listdir(data_dir)))
        if len(odd_df_list) != 1:
            result = False
            result_list.append(f'Odd_df({int(len(odd_df_list))})')

        if 'can' == self.data:
            can_list = os.listdir(os.path.join(data_dir, 'CAN'))
            if len(can_list) != 1:
                result = False
                result_list.append(f'can({int(len(can_list))})')
            try:
                odd_df = pd.read_csv(glob(os.path.join(data_dir, '*.csv'))[0])
                if 'VERSION' not in odd_df.columns:
                    result = False
                    result_list.append(f"No VERSION column in odd_df")
                elif int(odd_df['VERSION'][0].split('.')[1]) < 1:
                    result = False
                    result_list.append(f"Low can version({odd_df['VERSION'][0]})")
                else:
                    pass
            except:
                result = False
                pass

        if 'bio' == self.data:
            if not os.path.isfile(os.path.join(data_dir, 'bio', 'EDA.csv')):
                result = False
                result_list.append('eda(0)')
        if 'front_image' == self.data:
            try:
                cap = cv2.VideoCapture(os.path.join(self.driver_path, data, 'video', 'FrontView', 'color_front2.avi'))
                _, frame = cap.read()
                if frame.shape == (720, 1280, 3):
                    image_size = 'large'
                elif frame.shape == (480, 640, 3):
                    image_size = 'small'
                else:
                    raise ValueError(f"Wrong image size {frame.shape} in {self.driver}/{data}")
            except:
                result = False
                pass
            if image_size != self.image_size:
                result = False
                result_list.append(f'image size({frame.shape})')

            if len(glob(os.path.join(self.driver_path, data, 'video', '*.csv'))) != 1:
                result = False
            front_image_list = os.listdir(os.path.join(data_dir, 'video', 'FrontView'))
            if len(front_image_list) != 2:
                result = False
                result_list.append(f'front_image({int(len(front_image_list))})')
        if 'side_image' == self.data:
            try:
                cap = cv2.VideoCapture(os.path.join(self.driver_path, data, 'video', 'SideView', 'color_side2.avi'))
                _, frame = cap.read()
                if frame.shape == (720, 1280, 3):
                    image_size = 'large'
                elif frame.shape == (480, 640, 3):
                    image_size = 'small'
                else:
                    raise ValueError(f"Wrong image size {frame.shape} in {self.driver}/{data}")
            except:
                result = False
                pass
            if image_size != self.image_size:
                result = False
                result_list.append(f'image size({frame.shape})')
            if len(glob(os.path.join(self.driver_path, data, 'video', '*.csv'))) != 1:
                result = False
            side_image_list = os.listdir(os.path.join(data_dir, 'video', 'SideView'))
            if len(side_image_list) != 2:
                result = False
                result_list.append(f'side_image({int(len(side_image_list))})')
        if 'audio' == self.data:
            audio_list = os.listdir(os.path.join(data_dir, 'audio'))
            if len(audio_list) != 2:
                result = False
                result_list.append(f'audio({int(len(audio_list) / 2)})')
        if result == False:
            print(f'[INFO] No data in {self.driver}/{data}: {result_list}')
        return result

    def preprocess_can(self, df, replace):
        categorical_dict = {'CF_Tcu_TarGe': {'If N or P are detected(No frictional conncetion)': 0,
                                             'Reverse': -1, '1st speed': 1, '2nd speed': 2, '3rd speed': 3,
                                             '4th speed': 4, '5th speed': 5, '6th speed': 6},
                            'CF_Ems_EngStat': {'ES(Engine Stop)': 0, 'PL(Part Load)': 1, 'PU(Pull)': 2,
                                               'PUC(Fuel Cut off)': 3, 'ST(Start)': 4, 'IS(Idle speed)': 5},
                            'CYL_PRES_FLAG': {'On': 1, 'Off': 0},
                            'CF_Gway_HeadLampHigh': {'On': 1, 'Off': 0},
                            'CF_Gway_HeadLampLow': {'On': 1, 'Off': 0},
                            'CF_Hcu_DriveMode': {'Eco': 1},
                            'CR_Hcu_HevMod': {'Vehicle Stop': 0, 'Engine Generation': 1,
                                              'Engine Generation/Motor Drive': 2,
                                              'Engine Generation/ Regeneration': 3, 'Engine Brake / Regeneration': 4,
                                              'Regeneration': 5, 'EV Propulsion': 6,
                                              'Engine Only Propulsion': 7, 'Power Researve': 8, 'Power Assist': 9},
                            'CF_Ems_BrkForAct': {'On': 1, 'Off': 0},
                            'CF_Clu_InhibitD': {'(On)D': 1, 'Off': 0},
                            'CF_Clu_InhibitN': {'(On)N': 1, 'Off': 0},
                            'CF_Clu_InhibitP': {'(On)P': 1, 'Off': 0},
                            'CF_Clu_InhibitR': {'(On)R': 1, 'Off': 0}}

        for col in df.columns:
            if col in categorical_dict.keys():
                df[col] = df[col].interpolate(method='pad').fillna(method='bfill').fillna(method='ffill')

            elif col == 'CF_Clu_VehicleSpeed':
                df.loc[df[col] == '0x0~0xFE:Speed', 'CF_Clu_VehicleSpeed'] = 0
                df.loc[df[col].notna(), 'CF_Clu_VehicleSpeed'] = df.loc[df[col].notna(), 'CF_Clu_VehicleSpeed'].astype(
                    'float64')
                df[col] = df[col].interpolate().fillna(method='bfill').fillna(method='ffill').astype('int64')

            elif col == 'CR_Ems_AccPedDep_Pc':
                df.loc[df[col] == 'Accelerator Pedal not activated', 'CR_Ems_AccPedDep_Pc'] = 0
                df.loc[df[col] == 'Accelerator Pedal fully activated', 'CR_Ems_AccPedDep_Pc'] = 100
                df.loc[df[col].notna(), 'CR_Ems_AccPedDep_Pc'] = df.loc[df[col].notna(), 'CR_Ems_AccPedDep_Pc'].astype(
                    'float64')
                df[col] = df[col].interpolate().fillna(method='bfill').fillna(method='ffill')
            ### dtype: int
            elif col == 'CR_Brk_StkDep_Pc' or col == 'CR_Ems_EngSpd_rpm' or col == 'CR_Ems_VehSpd_Kmh' \
                    or col == 'BAT_SOC' or col == 'CR_Hcu_HigFueEff_Pc' or col == 'CR_Hcu_NorFueEff_Pc' \
                    or col == 'CR_Fatc_OutTempSns_C' or col == 'CR_Hcu_EcoLvl' or col == 'CR_Ems_EngColTemp_C':
                df[col] = df[col].interpolate().fillna(method='bfill').fillna(method='ffill').astype('int64')
            else:
                df[col] = df[col].interpolate().fillna(method='bfill').fillna(method='ffill')
        if replace == True:
            df = df.replace(categorical_dict)

        df = df.groupby('timestamp', as_index=False).mean()

        int_col_list = ['CF_Clu_InhibitD', 'CF_Clu_InhibitN', 'CF_Clu_InhibitP', 'CF_Clu_InhibitR', 'CF_Ems_EngStat',
                        'CF_Tcu_TarGe',
                        'CYL_PRES_FLAG', 'CF_Gway_HeadLampHigh', 'CF_Gway_HeadLampLow', 'CF_Ems_BrkForAct',
                        'CF_Hcu_DriveMode',
                        'CR_Hcu_HevMod', 'CR_Brk_StkDep_Pc', 'CR_Ems_EngSpd_rpm', 'CR_Ems_VehSpd_Kmh', 'BAT_SOC',
                        'CR_Hcu_HigFueEff_Pc',
                        'CR_Hcu_NorFueEff_Pc', 'CR_Fatc_OutTempSns_C', 'CR_Hcu_EcoLvl', 'CR_Ems_EngColTemp_C',
                        'CF_Clu_VehicleSpeed']
        for col in df.columns:
            if col in int_col_list:
                df[col] = df[col].astype('int64')
        # print(f'{time.time() - st_time:.2f}sec consumed for process CAN data')
        return df

    def get_per_sec(self):
        if self.image_size == 'large':
            tmp_result = {'can': 138, 'front_image': 15, 'side_image': 15, 'bio': 64, 'audio': 44100}
        elif self.image_size == 'small':
            tmp_result = {'can': 138, 'front_image': 30, 'side_image': 30, 'bio': 64, 'audio': 44100}
        else:
            raise ValueError(f"Wrong image size: {self.image_size}")
        return tmp_result


def main():
    epochs = 1
    nas_dms_dataset_path = os.path.join(os.getcwd(), 'nas_dms_dataset')
    dataloader = driving_mode_dataloader(
        nas_dms_dataset_path,  # data path
        max_km=100,  # maximum train data km
        driver='TaesanKim',  # dataset driver(directory name)
        batch_size=16,
        valid_ratio=0.3,  # The number of validation directory
        data='bio',  # ['can', 'front_image', 'side_image', 'bio', 'audio']
        pre_sec=1,  # pre second from self-reporting
        image_size='large',  # large = (720, 1280, 3), small = (480, 640, 3)
        no_response='ignore' # ignore or ffill(Fill values forward)

    )
    """
    x.shape
    data_length = (pre_sec * per_sec(data in each second))
    front_image, side_image: (batch_size, data_length, 720, 1280, 3)
    can: (batch_size, data_length, 36)
    bio: (batch_size, data_length, 3('EDA', 'BVP', 'HR'))
    audio: (batch_size, data_length)
    
    y.shape
    (batch_size, 4)
        1 -> [1,0,0,0]
        2 -> [0,1,0,0]
        3 -> [0,0,1,0]
        4 -> [0,0,0,1]
    """

    for epoch in range(epochs):
        print(f"[INFO] EPOCH {epoch + 1}/{epochs}")
        st_time = time.time()
        train_loader = iter(dataloader.get_train_data)
        valid_loader = iter(dataloader.get_valid_data)
        """
        for i, (x, y) in enumerate(iter(train_loader)): ->  When 'data' in ['bio', 'can', 'audio']
        for i, (color_x, ir_x, y) in enumerate(iter(dataloader.get_train_data)): -> When 'data' in ['side_image', 'front_image']
        """
        print('[INFO] Training start')
        for i, (x, y) in enumerate(train_loader):
        # for i, (color_x, ir_x, y) in enumerate(train_loader):
            print(x.shape, y.shape)
            """
            Training code
            """

        print('[INFO] Validtation start')
        for i, (x, y) in enumerate(valid_loader):
        # for i, (color_x, ir_x, y) in enumerate(valid_loader):
            print(x.shape, y.shape)
            """
            Validataion code
            """

        for test_num in range(dataloader.get_test_num()):
            test_loader = dataloader.get_test_data(test_num + 1)
            # for i, (color_x, ir_x, y) in enumerate(test_loader):
            for i, (x, y) in enumerate(test_loader):
                print(x.shape, y.shape)
                """
                Test for (i+1) directory (fron 1 to the number of test directory)
                """
                print(f"average time: {(time.time() - st_time):.2f}sec")
                st_time = time.time()
            print()
        # shuffle train and validation data
        dataloader.shuffle_data()


if __name__ == '__main__':
    main()
