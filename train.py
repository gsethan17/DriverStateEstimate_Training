import pickle
import os
import glob
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from dl_models import get_classifier
from utils import f1_loss, weighted_loss

def macro_soft_f1(y, y_hat):
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

    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    cost = 1 - soft_f1  # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost)  # average on all labels

    return macro_cost


def macro_double_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
    soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)
    cost_class1 = 1 - soft_f1_class1  # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0  # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0)  # take into account both class 1 and class 0
    macro_cost = tf.reduce_mean(cost)  # average on all labels
    return macro_cost

def load_labels(driver) :
    path = os.path.join(os.getcwd(), 'labels', str(driver) + '_serial_label.pickle')

    with open(path, 'rb') as f :
        data = pickle.load(f)

    return data

def get_data_path(driver) :
    path = os.path.join(os.getcwd(), 'save_feature', 'front', str(driver) + '*')
    data_list = glob.glob(path)

    return data_list

def split_train_test(data_path_list, driver) :
    train_path = []
    test_path = []

    if driver == 'G' :
        standard_odo = 11981
    elif driver == 'T' :
        standard_odo = 11327

    for path in data_path_list :
        odo = os.path.basename(path).split('.')[0].split('_')[1]

        if int(odo) > standard_odo :
            test_path.append(path)
        else :
            train_path.append(path)

    return train_path, test_path


def split_data_bylabel(data_path_list, label) :
    dic = {}
    dic['hn'] = []
    dic['es'] = []
    dic['ad'] = []
    dic['sf'] = []

    for path in data_path_list :
        name = os.path.basename(path).split('.')[0]
        idx = np.argmax(label[name])

        if idx == 0 :
            dic['ad'].append(path)

        elif idx == 1 :
            dic['es'].append(path)

        elif idx == 2 :
            dic['sf'].append(path)

        elif idx == 3 :
            dic['hn'].append(path)

    return dic

def split_train_val(dic, val_ratio) :
    val = {}

    for state in dic.keys() :
        total_num = len(dic[state])
        num = int(total_num * val_ratio)
        items = random.sample(range(total_num), num)
        items.sort()
        items.reverse()

        val[state] = []
        for k in items :
            val[state].append(dic[state].pop(k))
    print('Train : ', len(dic['hn']), len(dic['es']), len(dic['ad']), len(dic['sf']))
    print('Val : ', len(val['hn']), len(val['es']), len(val['ad']), len(val['sf']))

    ws = [len(dic['ad']), len(dic['es']), len(dic['sf']), len(dic['hn'])]
    we = []
    N = sum(ws)
    for w in ws :
        we.append(N / (4 * w))
    print('weight : ', we)

    train_list = dic['hn'] + dic['es'] + dic['ad'] + dic['sf']
    val_list = val['hn'] + val['es'] + val['ad'] + val['sf']

    return train_list, val_list, we

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
        batch_true = [self.label[os.path.basename(path).split('.')[0]] for path in batch_path]

        return tf.convert_to_tensor(batch_x), tf.convert_to_tensor(batch_true)

def train(train_loader, val_loader, w) :
    model = get_classifier(6, 0.2)

    if loss_function == 'CE' :
        LOSS = weighted_loss
    elif loss_function == 'WCE' :
        LOSS = weighted_loss
    elif loss_function == 'F1' :
        LOSS = macro_soft_f1
    elif loss_function == 'DF1' :
        LOSS = macro_double_soft_f1
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate)
    METRIC = f1_loss

    results = {}
    stop_flag = False

    for epoch in range(epochs) :
        train_loss = 0.
        train_metric = 0.
        for i in range(len(train_loader)) :
            train_x, train_y = train_loader[i]

            with tf.GradientTape() as tape :
                pred = model(train_x, training = True)
                if loss_function == 'WCE' :
                    loss = LOSS(train_y, pred, w)
                else :
                    loss = LOSS(train_y, pred)
                metric = METRIC(train_y, pred)

            gradients = tape.gradient(loss, model.trainable_variables)
            OPTIMIZER.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss += loss.numpy()
            train_metric += metric

        train_loss = train_loss / len(train_loader)
        train_metric = train_metric / len(train_loader)


        train_loader.on_epoch_end()


        for i in range(len(val_loader)):
            val_x, val_y = val_loader[i]

            pred = model(val_x, training=False)

            if i == 0 :
                Y = val_y
                Y_hat = pred

            else :
                Y = np.vstack([Y, val_y])
                Y_hat = np.vstack([Y_hat, pred])


        val_loss = LOSS(Y, Y_hat)
        val_metric = METRIC(Y, Y_hat)

        print('[INFO] {:>3} / {:>3} || Train : {:6.3f}, {:6.3f} || Validation : {:6.3f}, {:6.3f}'.format(
                                                                        epoch + 1,
                                                                        epochs,
                                                                        train_loss,
                                                                        train_metric,
                                                                        val_loss,
                                                                        val_metric
                                                                        ))


        if epoch == 0 :
            results['t_loss'] = [train_loss]
            results['t_metric'] = [train_metric]
            results['v_loss'] = [val_loss]
            results['v_metric'] = [val_metric]
        else :
            results['t_loss'].append(train_loss)
            results['t_metric'].append(train_metric)
            results['v_loss'].append(val_loss)
            results['v_metric'].append(val_metric)


        if results['v_loss'][-1] == min(results['v_loss']) :
            weights_path = os.path.join(save_path, 'weights')
            stop_flag = False
            if not os.path.isdir(weights_path) :
                os.makedirs(weights_path)
            model.save_weights(os.path.join(save_path, 'weights', 'best_weight'))

        if epoch > (patience - 1) :
            if min(results['v_loss'][(-1 * (patience + 1)):]) == results['v_loss'][(-1 * (patience + 1))]:
                stop_flag = True

        if stop_flag :
            break

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_path, 'train_results.csv'), index=False)

    return model

def test(model, test_loader, save_path) :
    trues = np.array([], dtype='int64')
    preds = np.array([], dtype='int64')

    for i in range(len(test_loader)):
        test_x, test_y = test_loader[i]

        pred = model(test_x, training=False)


        trues = np.concatenate([trues, np.argmax(test_y, axis=-1)], axis=0)
        preds = np.concatenate([preds, np.argmax(pred, axis=-1)], axis=0)

        if i == 0:
            Y = test_y
            Y_hat = pred

        else:
            Y = np.vstack([Y, test_y])
            Y_hat = np.vstack([Y_hat, pred])


    index_list = ['Angry', 'Surprise', 'Sad', 'Happy']
    df_results = pd.DataFrame(columns=index_list, index=index_list)
    df_results = df_results.fillna(0)

    for i in range(len(trues)):
        idx_true = trues[i]
        idx_pred = preds[i]

        #         print(idx_true, idx_pred)
        df_results.iloc[idx_true, idx_pred] += 1


    print(df_results)
    df_results.to_csv(os.path.join(save_path, 'confusion_matrix.csv'), index=False)

    get_results(df_results, save_path)


def get_results(df, save_path) :
    index_list = ['Angry', 'Surprise', 'Sad', 'Happy']
    tp = {}
    fp = {}
    fn = {}
    tn = {}
    precision = {}
    recall = {}
    f1 = {}
    accuracy = {}

    for index in index_list :
        index_other = index_list.copy()
        index_other.pop(index_other.index(index))
        tp[index] = df.loc[index][index]
        fp[index] = df.loc[index_other][index].sum()
        fn[index] = df.loc[index][index_other].sum()
        tn[index] = df.sum().sum() - tp[index] - fp[index] - fn[index]
        accuracy[index] = (tp[index] + tn[index]) / df.sum().sum()

        precision[index] = tp[index] / (tp[index] + fp[index] + 1e-16)
        recall[index] = tp[index] / (tp[index] + fn[index] + 1e-16)
        f1[index] = 2 * (precision[index] * recall[index]) / (precision[index] + recall[index] + 1e-16)


    f = open(os.path.join(save_path, '{}.txt'.format(np.mean(list(f1.values())[:2]))), "w")

    write = 'tp :' + str(tp) + '\n' \
    + 'fp :' + str(fp) + '\n' \
    + 'fn :'+ str(fn) + '\n' \
    + 'tn :'+ str(tn) + '\n' \
    + 'precision :'+ str(precision) + '\n' \
    + 'recall :'+ str(recall) + '\n' \
    + 'f1 :'+ str(f1) + '\n' \
    + 'f1_average :'+ str(np.mean(list(f1.values()))) + '\n' \
    + 'f1_minority :'+ str(np.mean(list(f1.values())[:2])) + '\n' \
    + 'accuracy :'+ str(accuracy) + '\n' \
    + 'average accuracy :'+ str(np.mean(list(accuracy.values()))) + '\n' \
    + 'overall accuracy :'+ str(np.sum(list(tp.values())) / df.sum().sum())
    print(write)
    f.write(write)
    f.close()

def main(driver, random_seed, batch_size) :
    random.seed(random_seed)

    label = load_labels(driver)

    data_path_list = get_data_path(driver)

    train_list, test_list = split_train_test(data_path_list, driver)
    print(len(train_list), len(test_list))

    train_dic = split_data_bylabel(train_list, label)
    test_dic = split_data_bylabel(test_list, label)

    # print(len(train_dic['hn']), len(train_dic['es']), len(train_dic['ad']), len(train_dic['sf']))
    print('Test : ', len(test_dic['hn']), len(test_dic['es']), len(test_dic['ad']), len(test_dic['sf']))

    train_list, val_list, w = split_train_val(train_dic, val_ratio)
    print(len(train_list), len(val_list))

    trainloader = Dataloader(train_list, label, batch_size)
    valloader = Dataloader(val_list, label, batch_size)
    testloader = Dataloader(test_list, label, batch_size)


    model = train(trainloader, valloader, w)
    test(model, testloader, save_path)


if __name__ == '__main__' :
    drivers = ['G', 'T']
    val_ratio = 0.3
    random_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    loss_functions = ['CE', 'WCE', 'F1', 'DF1']

    global epochs
    epochs = 500
    batch_sizes = [16, 32, 64, 128, 256]
    global learning_rate
    learning_rates = [0.01, 0.001, 0.0001, 0.00001]

    global patience
    patience = 10

    global save_path
    global loss_function

    for random_seed in random_seeds :
        for batch_size in batch_sizes :
            for learning_rate in learning_rates :
                for driver in drivers :
                    for loss_function in loss_functions :

                        save_path = os.path.join(os.getcwd(), 'fast_track',
                                                 driver + '_' + str(loss_function) + '_' + str(batch_size) + '_' + str(learning_rate),
                                                 str(random_seed))
                        if not os.path.isdir(save_path):
                            os.makedirs(save_path)

                        main(driver, random_seed, batch_size)










