import glob, os
from utils import gpu_limit
from dl_models import get_classifier
from train import load_labels, get_data_path, split_train_test, split_data_bylabel
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

def get_model(purpose, path) :
    model = get_classifier(6, 0.2, purpose)
    weight_path =os.path.join(path, 'weights', 'best_weight')
    # print(weight_path)
    model.load_weights(weight_path)

    return model

def get_test_data(driver, batch_size) :
    label = load_labels(driver)
    data_path_list = get_data_path(driver)

    _, test_list = split_train_test(data_path_list, driver)

    test_dic = split_data_bylabel(test_list, label)
    testloader = Dataloader(test_list, label, batch_size)

    return testloader, test_dic

class Dataloader(Sequence) :
    def __init__(self, paths, label, batch_size = 1, shuffle=True):
        self.paths = paths
        self.label = label
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self) :
        return int(np.ceil(len(self.paths)) / float(self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.paths))

        if self.shuffle :
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_path = [self.paths[i] for i in indices]
        batch_x = [np.load(path) for path in batch_path]
        batch_name = [os.path.basename(path).split('.')[0] for path in batch_path]
        batch_true = [self.label[os.path.basename(path).split('.')[0]] for path in batch_path]

        return tf.convert_to_tensor(batch_x), tf.convert_to_tensor(batch_true), batch_name

def test(model, testloader) :
    flag = False

    for x, y, name in testloader:
        pred = model(x)

        if not flag:
            preds = pred
            trues = y
            names = name
            flag = True

        else:
            preds = np.concatenate([preds, pred], axis=0)
            trues = np.concatenate([trues, y], axis=0)
            names += name


    return trues, preds, names

def get_recall_n_draw_histogram(output, path) :
    save_path = os.path.join(path, 'histogram')
    if not os.path.isdir(save_path) :
        os.makedirs(save_path)

    enmax_palette = ["#ff816e", "#ffbc8d", "#c0e8ff", "#ffec95"]
    color_codes_wanted = ['ad', 'es', 'sf', 'hn']
    c = lambda x: enmax_palette[color_codes_wanted.index(x)]

    guide = {'ad':0, 'es':1, 'sf':2, 'hn':3}

    recall = {}

    for label in output.keys() :
        pred = {}
        pred['ad'] = []
        pred['es'] = []
        pred['sf'] = []
        pred['hn'] = []

        tp = 0

        for preds in output[label] :
            pred['ad'].append(preds[0])
            pred['es'].append(preds[1])
            pred['sf'].append(preds[2])
            pred['hn'].append(preds[3])

            if preds[guide[label]] > 0.5 :
                tp += 1

        recall[label] = tp / len(output[label])

        for sub_label in pred.keys() :
            plt.figure()
            sns.distplot(pred[sub_label], bins=30, label=sub_label, color = c(sub_label))
            plt.legend(prop={'size': 14})
            plt.xlim([-0.1, 1.1])
            plt.savefig(os.path.join(save_path, str(label) +'_' + str(sub_label) + '.png'))

        plt.figure()
        for sub_label in pred.keys() :
            sns.distplot(pred[sub_label], bins=30, label=sub_label, color = c(sub_label))
        plt.legend(prop={'size': 14})
        plt.xlim([-0.1, 1.1])
        plt.savefig(os.path.join(save_path, str(label) + '.png'))

    return recall


def softmax(x):
    max = np.max(x, axis=1, keepdims=True)  # returns max of each row and keeps same dims
    e_x = np.exp(x - max)  # subtracts each row with its max value
    sum = np.sum(e_x, axis=1, keepdims=True)  # returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x

def get_threshold(trues, preds, path) :
    shape = preds.shape[-1]
    thres = np.arange(0, 1.01, 0.01)
    print(preds.shape)
    print(shape)

    if shape == 3 :
        f = open(os.path.join(path, 'threshold_minority.txt'), 'w')
        labels = ['ad', 'es', 'sf']
        names = ['Angry&Disgust', 'Exited&Surprised', 'Sad&Fatigue']
    elif shape == 1 :
        f = open(os.path.join(path, 'threshold_majority.txt'), 'w')
        labels = ['hn']
        names = ['Happy&Neutral']

    write = "label / threshold / f1 / precision / recall\n"
    f.write(write)

    for i, label in enumerate(labels[:shape]) :

        recalls = []
        precisions = []
        f1s = []

        positive = []
        negative = []

        for thre in thres :

            tp = []
            fp = []
            fn = []
            tn = []

            tp_count = 0
            true_count = 0
            pred_count = 0

            for j in range(len(trues)) :
                true_flag = False
                pred_flag = False

                if np.argmax(trues[j]) == i :
                    true_count += 1
                    true_flag = True

                if preds[j][i] > thre :
                    pred_count += 1
                    pred_flag = True



                if true_flag and pred_flag :
                    tp_count += 1
                    tp.append(preds[j][i])

                if true_flag != True and pred_flag == True :
                    fp.append(preds[j][i])

                if true_flag == True and pred_flag != True :
                    fn.append(preds[j][i])

                if true_flag != True and pred_flag != True :
                    tn.append(preds[j][i])

            recall = tp_count / (true_count + 1e-16)
            precision = tp_count / (pred_count + 1e-16)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-16)

            positive.append(tp + fn)
            negative.append(fp + tn)

            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)

        # print(max(recalls), recalls.index(max(recalls)))
        # print(max(precisions), precisions.index(max(precisions)))
        # print(max(f1s), f1s.index(max(f1s)))

        if shape == 3 :
            idx = f1s.index(max(f1s))
        elif shape == 1 :
            idx = precisions.index(max(precisions))
        print(positive[idx], len(positive[idx]))
        print(negative[idx], len(negative[idx]))
        # print(label, thres[idx], f1s[idx], precisions[idx], recalls[idx])
        write = label + '/' + str(round(thres[idx], 2)) + '/' + str(round(f1s[idx], 4)) + '/' + str(round(precisions[idx], 4)) + '/' + str(round(recalls[idx], 4)) + '\n'
        # write = "{} / {:.2f} / {:.3f} / {:.3f} / {:.3f}\n".format(label / thres[idx] / f1s[idx] / precisions[idx] / recalls[idx])
        f.write(write)

        plt.figure()
        sns.distplot(positive[idx], bins=30, label=names[i], color='green')
        sns.distplot(negative[idx], bins=30, label='others', color='red')
        # plt.axvline(thres[idx], 0, 0.8, linestyle='--', label='maximize the F1 Score', color='k')
        plt.axvline(thres[idx], 0, 0.8, linestyle='--', color='k')
        plt.legend(prop={'size': 14})
        plt.xlim([-0.1, 1.1])
        plt.savefig(os.path.join(path, str(label) + '.png'))
        # plt.show()

    f.close()


if __name__ == '__main__' :
    gpu_limit(3)

    drivers = ['T', 'G']
    batch_size = 16

    purpose = 'single'

    for driver in drivers :
        path_cases = glob.glob(os.path.join(os.getcwd(), 'fast_track', purpose, driver + '_*'))

        for path in path_cases :
            model = get_model(purpose, path)

            testloader, test_dic = get_test_data(driver, batch_size)

            trues, preds, names = test(model, testloader)

            '''
            output = {}

            for label in test_dic.keys() :
                output[label] = []
                for np_path in test_dic[label] :
                    name = os.path.basename(np_path).split('.')[0]

                    try :
                        idx = names.index(name)
                        # print(idx)
                        # print(preds[idx])

                        output[label].append(preds[idx])
                    except :
                        pass

            recall = get_recall_n_draw_histogram(output, path)

            with open(os.path.join(path, '0.5_recall.txt'), 'w') as file:
                file.write(json.dumps(recall))
            '''

            softmax_preds = softmax(preds[:, :-1])

            get_threshold(trues, softmax_preds, path)
            # get_threshold(trues, preds[:, :-1], path)

            # get_threshold(trues, preds[:, -1:], path)

            # break

