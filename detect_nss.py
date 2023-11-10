from __future__ import division, absolute_import, print_function
import argparse
from common.util import *
from setup_paths import *
from nss.MSCN import *
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale, MinMaxScaler

def main(args):
    set_seed(args)

    assert args.dataset in DATASETS, \
        "Dataset parameter must be either 'mnist', 'cifar', or 'imagenet'"
    ATTACKS = ATTACK[DATASETS.index(args.dataset)]
    if args.dataset != 'imagenet':
        assert os.path.isfile('{}cnn_{}.pt'.format(checkpoints_dir, args.dataset)), \
            'model file not found... must first train model using train_model.py.'

    pgd_per = pgd_percent[DATASETS.index(args.dataset)]

    print('Loading the data and model...')
    # Load the model
    if args.dataset == 'mnist':
        from baseline.cnn.cnn_mnist import MNISTCNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.pt'.format(args.dataset))
        classifier = model_class.classifier
    elif args.dataset == 'cifar':
        from baseline.cnn.cnn_cifar10 import CIFAR10CNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.pt'.format(args.dataset))
        classifier = model_class.classifier
    elif args.dataset == 'imagenet':
        from baseline.cnn.cnn_imagenet import ImageNetCNN as myModel
        model_class = myModel(filename='cnn_{}.pt'.format(args.dataset))
        classifier = model_class.classifier
    elif args.dataset == 'svhn':
        from baseline.cnn.cnn_svhn import SVHNCNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.pt'.format(args.dataset))
        classifier = model_class.classifier
        
    # Load the dataset
    X_test, Y_test = model_class.x_test, model_class.y_test

    #-----------------------------------------------#
    #              Train NSS detector               #
    #-----------------------------------------------# 
    #extract nss features, from normal images
    x_train_f_path = '{}{}_normal_f.npy'.format(nss_results_dir, args.dataset)
    if not os.path.isfile(x_train_f_path):
        X_train_f = np.array([])
        for img in X_test:
            # parameters = calculate_ggd_aggd(img,'GGD', kernel_size=7, sigma=7/6)
            parameters = calculate_brisque_features(img)
            parameters = parameters.reshape((1,-1))
            if X_train_f.size==0:
                X_train_f = parameters
            else:
                X_train_f = np.concatenate((X_train_f, parameters), axis=0)
        np.save(x_train_f_path, X_train_f)
    else:
        X_train_f = np.load(x_train_f_path)
    

    # X_train_f = scale_features(X_train_f)
    # scaler = MinMaxScaler(feature_range=(-1,1)).fit(X_train_f)
    # X_train_f = scaler.transform(X_train_f)

    X_train_f_copy = X_train_f

    #extract nss features, from adversarial images -- PGD
    pgds = ['pgdi_0.03125', 'pgdi_0.0625', 'pgdi_0.125', 'pgdi_0.25', 'pgdi_0.3125', 'pgdi_0.5']
    adv_data_f_all = []
    for pgd in pgds:
        adv_data = np.load('%s%s_%s.npy' % (adv_data_dir, args.dataset, pgd))
        adv_data_f_path = '{}{}_{}_f.npy'.format(nss_results_dir,args.dataset, pgd)
        if not os.path.isfile(adv_data_f_path):
            adv_data_f = np.array([])
            for img in adv_data:
                # parameters = calculate_ggd_aggd(img,'GGD', kernel_size=7, sigma=7/6)
                parameters = calculate_brisque_features(img)
                parameters = parameters.reshape((1,-1))
                if adv_data_f.size==0:
                    adv_data_f = parameters
                else:
                    adv_data_f = np.concatenate((adv_data_f, parameters), axis=0)
            np.save(adv_data_f_path, adv_data_f)
        else:
            adv_data_f = np.load(adv_data_f_path)
        
        # adv_data_f = scaler.transform(adv_data_f)
        adv_data_f_all.append(adv_data_f)

    #correctly classified samples
    preds_test = classifier.predict(X_test)
    inds_correct = np.where(preds_test.argmax(axis=1) == Y_test.argmax(axis=1))[0]
    X_test = X_test[inds_correct]
    Y_test = Y_test[inds_correct]
    X_train_f = X_train_f[inds_correct]
    for i in range(len(adv_data_f_all)):
        adv_data_f_all[i] = adv_data_f_all[i][inds_correct]
    
    # samples = [200, 200, 300, 100, 100, 100]
    samples = np.array(np.floor(np.array(pgd_per)*len(inds_correct)), dtype=np.int32)

    success_inds = []
    for pgd in pgds:
        adv_data = np.load('%s%s_%s.npy' % (adv_data_dir, args.dataset, pgd))
        adv_data = adv_data[inds_correct]
        pred_adv = classifier.predict(adv_data)
        inds_success = np.where(pred_adv.argmax(axis=1) != Y_test.argmax(axis=1))[0]
        success_inds.append(inds_success)

    selected_inds = []
    inds = random.sample(list(success_inds[0]), samples[0])
    selected_inds.append(inds)
    for i in range(1, len(pgds)):
        all_inds=[]
        for j in range(len(selected_inds)):
            all_inds = np.concatenate((all_inds, selected_inds[j]))
        
        allowed_inds = list(set(success_inds[i])-set(all_inds))
        inds = random.sample(allowed_inds, np.min([samples[i], len(allowed_inds)]))
        selected_inds.append(inds)

    train_inds=[]
    for i in range(len(selected_inds)):
        train_inds = np.concatenate((train_inds, selected_inds[i]))
    train_inds = np.int32(train_inds)
    test_inds = np.asarray(list(set(range(len(inds_correct)))-set(train_inds)))
 
    #train the model
    x_normal_f = X_train_f[train_inds]
    y_normal_f = np.zeros(len(train_inds), dtype=np.uint8)
    x_adv_f = np.concatenate((adv_data_f_all[0][selected_inds[0]],\
                            adv_data_f_all[1][selected_inds[1]],\
                            adv_data_f_all[2][selected_inds[2]],\
                            adv_data_f_all[3][selected_inds[3]],\
                            adv_data_f_all[4][selected_inds[4]],\
                            adv_data_f_all[5][selected_inds[5]]))
    y_adv_f = np.ones(len(train_inds), dtype=np.uint8)
    
    x_train = np.concatenate((x_normal_f, x_adv_f))
    y_train = np.concatenate((y_normal_f, y_adv_f))

    min_ = np.min(x_train, axis=0)
    max_ = np.max(x_train, axis=0)
    x_train = scale_features(x_train, min_, max_)
    
    #mnist
    if args.dataset == 'mnist':
        c=1000000.0
        g=1e-08
    elif args.dataset == 'cifar':
        c=10000
        g=1e-05
    elif args.dataset == 'svhn':
        c=0.1
        g=1e-08
    else:
        c=10000000000
        g=0.0001
    clf = svm.SVC(C=10, kernel='sigmoid', gamma=0.01, probability=True,random_state=0)
    clf.fit(x_train, y_train)

    #-----------------------------------------------#
    #                 Evaluate NSS                  #
    #-----------------------------------------------# 
    ## Evaluate detector -- on adversarial attack
    Y_test_copy=Y_test
    X_test_copy=X_test
    X_train_f_copy=scale_features(X_train_f_copy, min_, max_)
    for attack in ATTACKS:
        Y_test=Y_test_copy
        X_test=X_test_copy
        X_train_f=X_train_f_copy
        results_all = []

        #Prepare data
        # Load adversarial samples
        X_test_adv = np.load('%s%s_%s.npy' % (adv_data_dir, args.dataset, attack))
        #get NSS for adv
        adv_data_f_path = '{}{}_{}_f.npy'.format(nss_results_dir, args.dataset, attack)
        if not os.path.isfile(adv_data_f_path):
            adv_data_f = np.array([])
            for img in X_test_adv:
                # parameters = calculate_ggd_aggd(img,'GGD', kernel_size=7, sigma=7/6)
                parameters = calculate_brisque_features(img)
                parameters = parameters.reshape((1,-1))
                if adv_data_f.size==0:
                    adv_data_f = parameters
                else:
                    adv_data_f = np.concatenate((adv_data_f, parameters), axis=0)
            np.save(adv_data_f_path, adv_data_f)
        else:
            adv_data_f = np.load(adv_data_f_path)
        adv_data_f = scale_features(adv_data_f, min_, max_)
        # adv_data_f = scaler.transform(adv_data_f)
        
        X_test_adv = X_test_adv[inds_correct]
        nss_adv = adv_data_f[inds_correct]
        X_train_f = X_train_f[inds_correct]

        pred_adv = classifier.predict(X_test_adv)
        # loss, acc_suc = classifier.evaluate(X_test_adv, Y_test, verbose=0)
        acc_suc = np.sum(np.argmax(pred_adv, axis=1) == np.argmax(Y_test, axis=1)) / len(Y_test)

        inds_success = np.where(pred_adv.argmax(axis=1) != Y_test.argmax(axis=1))[0]
        inds_fail = np.where(pred_adv.argmax(axis=1) == Y_test.argmax(axis=1))[0]     
        nss_adv_success = nss_adv[inds_success]
        nss_adv_fail = nss_adv[inds_fail]

        # prepare X and Y for detectors
        X_all = np.concatenate([X_train_f, nss_adv])
        Y_all = np.concatenate([np.zeros(len(X_train_f), dtype=bool), np.ones(len(X_train_f), dtype=bool)])
        X_success = np.concatenate([X_train_f[inds_success], nss_adv_success])
        Y_success = np.concatenate([np.zeros(len(inds_success), dtype=bool), np.ones(len(inds_success), dtype=bool)])
        X_fail = np.concatenate([X_train_f[inds_fail], nss_adv_fail])
        Y_fail = np.concatenate([np.zeros(len(inds_fail), dtype=bool), np.ones(len(inds_fail), dtype=bool)])

        #For Y_all
        if np.any(np.isnan(X_all)):
            X_all = np.nan_to_num(X_all)
        Y_all_pred = clf.predict(X_all)
        Y_all_pred_score = clf.decision_function(X_all)

        acc_all, tpr_all, fpr_all, tp_all, ap_all, fb_all, an_all = evalulate_detection_test(Y_all, Y_all_pred)
        fprs_all, tprs_all, thresholds_all = roc_curve(Y_all, Y_all_pred_score)
        roc_auc_all = auc(fprs_all, tprs_all)
        print("AUC: {:.4f}%, Overall accuracy: {:.4f}%, FPR value: {:.4f}%".format(100*roc_auc_all, 100*acc_all, 100*fpr_all))

        curr_result = {'type':'all', 'nsamples': len(inds_correct),	'acc_suc': acc_suc,	\
                'acc': acc_all, 'tpr': tpr_all, 'fpr': fpr_all, 'tp': tp_all, 'ap': ap_all, 'fb': fb_all, 'an': an_all,	\
                'tprs': list(fprs_all), 'fprs': list(tprs_all),	'auc': roc_auc_all}
        results_all.append(curr_result)

        #for Y_success
        if len(inds_success)==0:
            tpr_success=np.nan
            curr_result = {'type':'success', 'nsamples': 0,	'acc_suc': 0,	\
                    'acc': np.nan, 'tpr': np.nan, 'fpr': np.nan, 'tp': np.nan, 'ap': np.nan, 'fb': np.nan, 'an': np.nan,	\
                    'tprs': np.nan, 'fprs': np.nan,	'auc': np.nan}
            results_all.append(curr_result)
        else:
            if np.any(np.isnan(X_success)):
                X_success = np.nan_to_num(X_success)
            Y_success_pred = clf.predict(X_success)
            Y_success_pred_score = clf.decision_function(X_success)
            accuracy_success, tpr_success, fpr_success, tp_success, ap_success, fb_success, an_success = evalulate_detection_test(Y_success, Y_success_pred)
            fprs_success, tprs_success, thresholds_success = roc_curve(Y_success, Y_success_pred_score)
            roc_auc_success = auc(fprs_success, tprs_success)

            curr_result = {'type':'success', 'nsamples': len(inds_success),	'acc_suc': 0,	\
                    'acc': accuracy_success, 'tpr': tpr_success, 'fpr': fpr_success, 'tp': tp_success, 'ap': ap_success, 'fb': fb_success, 'an': an_success,	\
                    'tprs': list(fprs_success), 'fprs': list(tprs_success),	'auc': roc_auc_success}
            results_all.append(curr_result)

        #for Y_fail
        if len(inds_fail)==0:
            tpr_fail=np.nan
            curr_result = {'type':'fail', 'nsamples': 0,	'acc_suc': 0,	\
                    'acc': np.nan, 'tpr': np.nan, 'fpr': np.nan, 'tp': np.nan, 'ap': np.nan, 'fb': np.nan, 'an': np.nan,	\
                    'tprs': np.nan, 'fprs': np.nan,	'auc': np.nan}
            results_all.append(curr_result)
        else:
            if np.any(np.isnan(X_fail)):
                X_fail = np.nan_to_num(X_fail)
            Y_fail_pred = clf.predict(X_fail)
            Y_fail_pred_score = clf.decision_function(X_fail)
            accuracy_fail, tpr_fail, fpr_fail, tp_fail, ap_fail, fb_fail, an_fail = evalulate_detection_test(Y_fail, Y_fail_pred)
            fprs_fail, tprs_fail, thresholds_fail = roc_curve(Y_fail, Y_fail_pred_score)
            roc_auc_fail = auc(fprs_fail, tprs_fail)

            curr_result = {'type':'fail', 'nsamples': len(inds_fail),	'acc_suc': 0,	\
                    'acc': accuracy_fail, 'tpr': tpr_fail, 'fpr': fpr_fail, 'tp': tp_fail, 'ap': ap_fail, 'fb': fb_fail, 'an': an_fail,	\
                    'tprs': list(fprs_fail), 'fprs': list(tprs_fail),	'auc': roc_auc_fail}
            results_all.append(curr_result)

        import csv
        with open('{}{}_{}.csv'.format(nss_results_dir, args.dataset, attack), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results_all:
                writer.writerow(row)
        
        print('{:>15} attack - accuracy of pretrained model: {:7.2f}% \
            - detection rates ------ SAEs: {:7.2f}%, FAEs: {:7.2f}%'.format(attack, 100*acc_suc, 100*tpr_success, 100*tpr_fail))
    
    print('Done!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help="Dataset to use; either {}".format(DATASETS), required=True, type=str)
    parser.add_argument('-s', '--seed', help='set seed for model', default=123, type=int)
    args = parser.parse_args()
    main(args)