import os, glob, random
import numpy as np
import pandas as pd
from scipy import stats

def get_va_aff(affects):
    dic = {}

    nh_v = np.array([], dtype='float64')
    n_v = np.array([], dtype='float64')
    h_v = np.array([], dtype='float64')
    es_v = np.array([], dtype='float64')
    ad_v = np.array([], dtype='float64')
    a_v = np.array([], dtype='float64')
    d_v = np.array([], dtype='float64')
    sb_v = np.array([], dtype='float64')

    nh_a = np.array([], dtype='float64')
    n_a = np.array([], dtype='float64')
    h_a = np.array([], dtype='float64')
    es_a = np.array([], dtype='float64')
    ad_a = np.array([], dtype='float64')
    a_a = np.array([], dtype='float64')
    d_a = np.array([], dtype='float64')
    sb_a = np.array([], dtype='float64')

    for i, affect in enumerate(affects):
        if i == 0:
            df = pd.read_csv(affect)
            col = df.columns

        else:
            df = pd.read_csv(affect, names=list(col))

        n_v = np.concatenate([n_v, np.array(df[df['expression'] == 0]['valence'])], axis=0)  # neutral
        n_a = np.concatenate([n_a, np.array(df[df['expression'] == 0]['arousal'])], axis=0)  # neutral

        nh_v = np.concatenate([nh_v, np.array(df[df['expression'] == 0]['valence'])], axis=0)  # neutral
        nh_a = np.concatenate([nh_a, np.array(df[df['expression'] == 0]['arousal'])], axis=0)  # neutral

        h_v = np.concatenate([h_v, np.array(df[df['expression'] == 1]['valence'])], axis=0)  # happy
        h_a = np.concatenate([h_a, np.array(df[df['expression'] == 1]['arousal'])], axis=0)

        nh_v = np.concatenate([nh_v, np.array(df[df['expression'] == 1]['valence'])], axis=0)  # happy
        nh_a = np.concatenate([nh_a, np.array(df[df['expression'] == 1]['arousal'])], axis=0)

        sb_v = np.concatenate([sb_v, np.array(df[df['expression'] == 2]['valence'])], axis=0)  # sad
        sb_a = np.concatenate([sb_a, np.array(df[df['expression'] == 2]['arousal'])], axis=0)

        es_v = np.concatenate([es_v, np.array(df[df['expression'] == 3]['valence'])], axis=0)  # surprise
        es_a = np.concatenate([es_a, np.array(df[df['expression'] == 3]['arousal'])], axis=0)

        ad_v = np.concatenate([ad_v, np.array(df[df['expression'] == 6]['valence'])], axis=0)  # Anger
        ad_a = np.concatenate([ad_a, np.array(df[df['expression'] == 6]['arousal'])], axis=0)

        ad_v = np.concatenate([ad_v, np.array(df[df['expression'] == 5]['valence'])], axis=0)  # Disgust
        ad_a = np.concatenate([ad_a, np.array(df[df['expression'] == 5]['arousal'])], axis=0)

        a_v = np.concatenate([a_v, np.array(df[df['expression'] == 6]['valence'])], axis=0)  # Anger
        a_a = np.concatenate([a_a, np.array(df[df['expression'] == 6]['arousal'])], axis=0)

        d_v = np.concatenate([d_v, np.array(df[df['expression'] == 5]['valence'])], axis=0)  # Disgust
        d_a = np.concatenate([d_a, np.array(df[df['expression'] == 5]['arousal'])], axis=0)

    dic['HN'] = np.concatenate([np.expand_dims(nh_v, axis=-1), np.expand_dims(nh_a, axis=-1)], axis=1)
    dic['ES'] = np.concatenate([np.expand_dims(es_v, axis=-1), np.expand_dims(es_a, axis=-1)], axis=1)
    dic['AD'] = np.concatenate([np.expand_dims(ad_v, axis=-1), np.expand_dims(ad_a, axis=-1)], axis=1)
    dic['SF'] = np.concatenate([np.expand_dims(sb_v, axis=-1), np.expand_dims(sb_a, axis=-1)], axis=1)

    return dic

class Statistical_Analysis() :
    def __init__(self, drivers, num_sample, affect_va):
        self.base_path = os.path.join(os.getcwd(), 'VA_results', 'matching')
        self.drivers = drivers
        self.num_sample = num_sample
        self.affect_va = affect_va
        self.status = ['AD', 'ES', 'SF', 'HN']
        self.get_odos()
        self.get_va()
        self.total_va = self.split_va(self.total_va)
        for driver in self.drivers :
            self.va[driver] = self.split_va(self.va[driver])
        self.sample_total = self.sampling(self.total_va)
        self.sample_affect = self.sampling(self.affect_va)


    def get_odos(self):
        self.odos = {}
        for driver in self.drivers :
            path_list = glob.glob(os.path.join(self.base_path, driver + '*'))
            odos = [int(os.path.basename(path).split('_')[1]) for path in path_list]
            self.odos[driver] = sorted(set(odos))


    def get_va(self):
        self.va = {}

        total_flag = False

        for driver in self.drivers :
            odos = self.odos[driver]
            self.va[driver] = {}

            driver_flag = False

            for odo in odos :
                true_path = glob.glob(os.path.join(self.base_path, driver + '_' + str(odo) + '_trues*'))[0]
                pred_path = glob.glob(os.path.join(self.base_path, driver + '_' + str(odo) + '_preds*'))[0]

                true_va = np.load(true_path)
                pred_va = np.load(pred_path)
                # print(np.sum(np.isnan(true_va)))
                # if np.sum(np.isnan(true_va)) > 0 or np.sum(np.isnan(pred_va)) > 0 :
                #     print(driver, odo)

                if not driver_flag :
                    driver_true = true_va
                    driver_pred = pred_va
                    driver_flag = True
                else :
                    driver_true = np.concatenate([driver_true, true_va], axis=0)
                    driver_pred = np.concatenate([driver_pred, pred_va], axis=0)

                if not total_flag :
                    total_true = true_va
                    total_pred = pred_va
                    total_flag = True
                else :
                    total_true = np.concatenate([total_true, true_va], axis=0)
                    total_pred = np.concatenate([total_pred, pred_va], axis=0)

                self.va[driver]['True'] = driver_true
                self.va[driver]['Pred'] = driver_pred

        self.total_va = {}
        self.total_va['True'] = total_true
        self.total_va['Pred'] = total_pred

    def split_va(self, dic):
        flag = {}
        for status in self.status :
            flag[status] = False

        for i, true in enumerate(dic['True']) :
            idx = np.argmax(true)
            if not flag[self.status[idx]] :
                dic[self.status[idx]] = np.expand_dims(dic['Pred'][i], axis=0)
                flag[self.status[idx]] = True
            else :
                dic[self.status[idx]] = np.concatenate([dic[self.status[idx]], np.expand_dims(dic['Pred'][i], axis=0)], axis = 0)

        return dic

    def sampling(self, dic):
        sample = {}

        for status in self.status :
            sample[status] = []

            len_ = len(dic[status])
            nums = int(len_ / self.num_sample)
            idxs = [x for x in range(len_)]

            for i in range(self.num_sample) :
                sub_sample = []
                selects = random.sample(idxs, nums)

                for select in selects :
                    sub_sample.append(dic[status][select])

                sample[status].append(np.mean(sub_sample, axis=0))
                # print(np.mean(sub_sample, axis=0))
            sample[status] = np.array(sample[status])

        return sample

if __name__ == '__main__' :
    random.seed(2)

    affect_path = glob.glob(os.path.join(os.path.join(os.getcwd(), 'VA_results', 'AffectNet'), '*.csv'))
    dic_affect = get_va_aff(affect_path)
    # print(dic_affect.keys())

    drivers = ['GeesungOh', 'TaesanKim', 'EuiseokJeong', 'JoonghooPark']
    num_sample = 10
    SE = Statistical_Analysis(drivers, num_sample, dic_affect)

    status = SE.status

    # print(SE.total_va.keys())
    # print(SE.va.keys())
    # print(np.mean(SE.va['GeesungOh']['Pred'], axis=0))

    # print(SE.va['EuiseokJeong']['SF'])

    print(SE.va[drivers[2]].keys())
    for driver in drivers :
        print(np.sum(np.isnan(SE.va[driver]['SF'])))
        # print(SE.va[driver])

    for driver in drivers :
        print(driver)
        dic = SE.sampling(SE.va[driver])
        # print(driver)
        # print(dic.keys())
        for stat in SE.status :
            print(stat)
            popmean = np.mean(SE.va[driver][stat], axis=0)
            print(stats.ttest_1samp(dic[stat], popmean=popmean))
            print(stats.f_oneway(dic[SE.status[0]], dic[SE.status[1]], dic[SE.status[2]], dic[SE.status[3]]))
    #
    '''
    for status in SE.status :
        print(SE.sample_total[status].shape)
        print(SE.sample_affect[status].shape)

        print(status)
        print(stats.ttest_ind(a=SE.sample_total[status], b=SE.sample_affect[status]))
        print(np.mean(SE.sample_total[status], axis=0), np.mean(SE.sample_affect[status], axis=0))
        print(np.std(SE.sample_total[status], axis=0), np.std(SE.sample_affect[status], axis=0))

    print(stats.f_oneway(SE.sample_total[SE.status[0]], SE.sample_total[SE.status[1]], SE.sample_total[SE.status[2]], SE.sample_total[SE.status[3]]))
    print(stats.f_oneway(SE.sample_affect[SE.status[0]], SE.sample_affect[SE.status[1]], SE.sample_affect[SE.status[2]], SE.sample_affect[SE.status[3]]))
    '''
