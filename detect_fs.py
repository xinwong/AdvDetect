import argparse
from common.util import *
from setup_paths import *
from fs.datasets.datasets_utils import * 
from fs.utils.squeeze import *

def get_distance(model, dataset, X1):
    X1_pred = model.predict(X1)
    vals_squeezed = []

    if dataset == 'mnist':
        X1_seqeezed_bit = bit_depth_py(X1, 1)
        vals_squeezed.append(model.predict(X1_seqeezed_bit))
        X1_seqeezed_filter_median = median_filter_py(X1, 2)
        vals_squeezed.append(model.predict(X1_seqeezed_filter_median))
    else:
        X1_seqeezed_bit = bit_depth_py(X1, 5)
        vals_squeezed.append(model.predict(X1_seqeezed_bit))
        X1_seqeezed_filter_median = median_filter_py(X1, 2)
        vals_squeezed.append(model.predict(X1_seqeezed_filter_median))
        X1_seqeezed_filter_local = non_local_means_color_py(np.transpose(X1, (0, 2, 3, 1)), 13, 3, 2)
        vals_squeezed.append(model.predict(np.transpose(X1_seqeezed_filter_local, (0, 3, 1, 2))))

    dist_array = []
    for val_squeezed in vals_squeezed:
        # print(val_squeezed.shape)
        dist = np.sum(np.abs(X1_pred - val_squeezed), axis=tuple(range(len(X1_pred.shape))[1:]))
        dist_array.append(dist)

    dist_array = np.array(dist_array)
    return np.max(dist_array, axis=0)

def train_fs(model, dataset, X1, train_fpr):
    distances = get_distance(model, dataset, X1)
    selected_distance_idx = int(np.ceil(len(X1) * (1-train_fpr)))
    threshold = sorted(distances)[selected_distance_idx-1]
    threshold = threshold
    print ("Threshold value: %f" % threshold)
    return threshold

def test(model, dataset, X, threshold):
    distances = get_distance(model, dataset, X)
    Y_pred = distances > threshold
    return Y_pred, distances

def main(args):
    set_seed(args)

    assert args.dataset in DATASETS, \
        "Dataset parameter must be either 'mnist', 'cifar' or 'imagenet'"
    ATTACKS = ATTACK[DATASETS.index(args.dataset)]

    if args.dataset != 'imagenet':
        assert os.path.isfile('{}cnn_{}.pt'.format(checkpoints_dir, args.dataset)), \
            'model file not found... must first train model using train_model.py.'

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
    X_test_all, Y_test_all = model_class.x_test, model_class.y_test

    #--------------
    # Evaluate the trained model.
    # Refine the normal and adversarial sets to only include samples for
    # which the original version was correctly classified by the model
    print ("Evaluating the pre-trained model...")
    Y_pred_all = classifier.predict(X_test_all)
    accuracy_all = calculate_accuracy(Y_pred_all, Y_test_all)
    print('Test accuracy on raw legitimate examples %.4f' % (accuracy_all))
    inds_correct = np.where(Y_pred_all.argmax(axis=1) == Y_test_all.argmax(axis=1))[0]
    X_test = X_test_all[inds_correct]
    Y_test = Y_test_all[inds_correct]
    Y_pred = Y_pred_all[inds_correct]
    indx_train = random.sample(range(len(X_test)), int(len(X_test)/2))
    indx_test = list(set(range(0, len(X_test)))-set(indx_train))
    print("Number of correctly predict images: %s" % (len(inds_correct)))
    x_indx_train = X_test[indx_train]
    y_indx_train = Y_test[indx_train]
    x_indx_test = X_test[indx_test]
    y_indx_test = Y_test[indx_test]
    #compute thresold - use test data to compute that
    threshold = train_fs(classifier, args.dataset, x_indx_train, 0.05)

    Y_test_copy=Y_test
    X_test_copy=X_test
    y_test_copy=y_indx_test
    x_test_copy=x_indx_test

    ## Evaluate detector
    #on adversarial attack
    for attack in ATTACKS:
        Y_test=Y_test_copy
        X_test=X_test_copy
        y_indx_test=y_test_copy
        x_indx_test=x_test_copy
        results_all = []

        #Prepare data
        # Load adversarial samples
        X_test_adv = np.load('{}{}_{}.npy'.format(adv_data_dir, args.dataset, attack))
        X_test_adv = reduce_precision_py(X_test_adv, 256)
    
        X_test_adv = X_test_adv[inds_correct]
        X_test_adv = X_test_adv[indx_test]
        
        # loss, acc_suc = model.evaluate(X_test_adv, y_test, verbose=0)
        X_test_adv_pred = classifier.predict(X_test_adv)
        acc_suc = np.sum(np.argmax(X_test_adv_pred, axis=1) == np.argmax(y_indx_test, axis=1)) / len(y_indx_test)

        inds_success = np.where(X_test_adv_pred.argmax(axis=1) != y_indx_test.argmax(axis=1))[0]
        inds_fail = np.where(X_test_adv_pred.argmax(axis=1) == y_indx_test.argmax(axis=1))[0]
        # inds_all_not_fail = list(set(range(0, len(inds_correct)))-set(inds_fail))
        X_test_adv_success = X_test_adv[inds_success]
        Y_test_success = y_indx_test[inds_success]
        X_test_adv_fail = X_test_adv[inds_fail]
        Y_test_fail = y_indx_test[inds_fail]

        # prepare X and Y for detectors
        X_all = np.concatenate([x_indx_test, X_test_adv])
        Y_all = np.concatenate([np.zeros(len(x_indx_test), dtype=bool), np.ones(len(x_indx_test), dtype=bool)])
        X_success = np.concatenate([x_indx_test[inds_success], X_test_adv_success])
        Y_success = np.concatenate([np.zeros(len(inds_success), dtype=bool), np.ones(len(inds_success), dtype=bool)])
        X_fail = np.concatenate([x_indx_test[inds_fail], X_test_adv_fail])
        Y_fail = np.concatenate([np.zeros(len(inds_fail), dtype=bool), np.ones(len(inds_fail), dtype=bool)])

        # for Y_all
        # if attack == ATTACKS[0]:
        Y_all_pred, Y_all_pred_score = test(classifier, args.dataset, X_all, threshold)
        acc_all, tpr_all, fpr_all, tp_all, ap_all, fb_all, an_all = evalulate_detection_test(Y_all, Y_all_pred)
        fprs_all, tprs_all, thresholds_all = roc_curve(Y_all, Y_all_pred_score)
        roc_auc_all = auc(fprs_all, tprs_all)
        print("AUC: {:.4f}%, Overall accuracy: {:.4f}%, FPR value: {:.4f}%".format(100*roc_auc_all, 100*acc_all, 100*fpr_all))

        curr_result = {'type':'all', 'nsamples': len(inds_correct),	'acc_suc': acc_suc,	\
                        'acc': acc_all, 'tpr': tpr_all, 'fpr': fpr_all, 'tp': tp_all, 'ap': ap_all, 'fb': fb_all, 'an': an_all,	\
                        'tprs': list(fprs_all), 'fprs': list(tprs_all),	'auc': roc_auc_all}
        results_all.append(curr_result)


        # for Y_success
        if len(inds_success)==0:
            tpr_success=np.nan
            curr_result = {'type':'success', 'nsamples': 0,	'acc_suc': 0,	\
                    'acc': np.nan, 'tpr': np.nan, 'fpr': np.nan, 'tp': np.nan, 'ap': np.nan, 'fb': np.nan, 'an': np.nan,	\
                    'tprs': np.nan, 'fprs': np.nan,	'auc': np.nan}
            results_all.append(curr_result)
        else:
            Y_success_pred, Y_success_pred_score = test(classifier, args.dataset, X_success, threshold)
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
            Y_fail_pred, Y_fail_pred_score = test(classifier, args.dataset, X_fail, threshold)
            accuracy_fail, tpr_fail, fpr_fail, tp_fail, ap_fail, fb_fail, an_fail = evalulate_detection_test(Y_fail, Y_fail_pred)
            fprs_fail, tprs_fail, thresholds_fail = roc_curve(Y_fail, Y_fail_pred_score)
            roc_auc_fail = auc(fprs_fail, tprs_fail)

            curr_result = {'type':'fail', 'nsamples': len(inds_fail),	'acc_suc': 0,	\
                    'acc': accuracy_fail, 'tpr': tpr_fail, 'fpr': fpr_fail, 'tp': tp_fail, 'ap': ap_fail, 'fb': fb_fail, 'an': an_fail,	\
                    'tprs': list(fprs_fail), 'fprs': list(tprs_fail),	'auc': roc_auc_fail}
            results_all.append(curr_result)
        
        import csv
        with open('{}{}_{}.csv'.format(fs_results_dir, args.dataset, attack), 'w', newline='') as csvfile:
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