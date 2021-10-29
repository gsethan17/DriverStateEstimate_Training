import numpy as np
import os
from dl_models import get_classifier
import glob

def get_model(driver, loss) :

    model = get_classifier(6, 0.2, purpose = 'single')

    path = os.path.join(os.getcwd(), 'fast_track', 'single', driver + '_' + loss)

    weight_path = os.path.join(path, 'weights', 'best_weight')
    # print(weight_path)
    model.load_weights(weight_path)

    return model

if __name__ == '__main__' :
    feature_path = os.path.join(os.getcwd(), 'nas_dms_dataset', 'ensemble', 'front')
    save_path = os.path.join(os.getcwd(), 'nas_dms_dataset', 'ensemble', 'predict', 'front')

    model_t_df1 = get_model(driver = 'T', loss = 'DF1')
    model_t_cpr = get_model(driver = 'T', loss = 'CPR')
    model_g_df1 = get_model(driver = 'G', loss = 'DF1')
    model_g_cpr = get_model(driver = 'G', loss = 'CPR')

    for path in glob.glob(os.path.join(feature_path, '*')) :
        input_ = np.load(path)
        input_ = np.expand_dims(input_, axis = 0)

        name = os.path.basename(path)

        if name.split('_')[0] == 'T' :

            df1 = model_t_df1(input_)

            cpr = model_t_cpr(input_)


        elif name.split('_')[0] == 'G':
            df1 = model_g_df1(input_)

            cpr = model_g_cpr(input_)

        df1_save_path = os.path.join(save_path, 'DF1', name)
        cpr_save_path = os.path.join(save_path, 'CPR', name)
        print(input_.shape, df1.shape, cpr.shape)

        np.save(df1_save_path, df1)
        np.save(cpr_save_path, cpr)
