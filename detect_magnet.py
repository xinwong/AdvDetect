from __future__ import division, absolute_import, print_function
import argparse
from common.util import *
from setup_paths import *
from sklearn.metrics import accuracy_score, precision_score, recall_score
from magnet.defensive_models import DenoisingAutoEncoder_1, DenoisingAutoEncoder_2
from magnet.worker import *

def test(dic, X, thrs):
    dist_all = []
    pred_labels = []
    for d in dic:
        m = dic[d].mark(torch.as_tensor(X))
        dist_all.append(m)
        pred_labels.append(m>thrs[d])
    
    labels = pred_labels[0]
    for i in range(1, len(pred_labels)):
        labels = labels | pred_labels[i]
    
    return labels, dist_all 

def main(args):
    set_seed(args)

    assert args.dataset in DATASETS, \
        "Dataset parameter must be either 'mnist', 'cifar', 'svhn' or 'imagenet'"
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
        clip_min, clip_max = 0,1
        v_noise=0.1
        p1=2
        p2=1
        type='error'
        t=10
        drop_rate={"I": 0.001, "II": 0.001}
        epochs=100
    elif args.dataset == 'cifar':
        from baseline.cnn.cnn_cifar10 import CIFAR10CNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.pt'.format(args.dataset))
        classifier = model_class.classifier
        clip_min, clip_max = 0,1
        v_noise=0.025
        p1=1
        p2=1
        type='error'
        t=40
        drop_rate={"I": 0.005, "II": 0.005}
        epochs=400
    elif args.dataset == 'imagenet':
        from baseline.cnn.cnn_imagenet import ImageNetCNN as myModel
        model_class = myModel(filename='cnn_{}.pt'.format(args.dataset))
        classifier = model_class.classifier
        clip_min, clip_max = 0,1
        v_noise=0.025
        p1=1
        p2=1
        type='error'
        t=10
        drop_rate={"I": 0.005, "II": 0.005}
        epochs=400
    elif args.dataset == 'svhn':
        from baseline.cnn.cnn_svhn import SVHNCNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.pt'.format(args.dataset))
        classifier = model_class.classifier
        clip_min, clip_max = 0,1
        v_noise=0.025
        p1=1
        p2=1
        type='error'
        t=40
        drop_rate={"I": 0.005, "II": 0.005}
        epochs=400

    # Load the dataset
    X_train, Y_train, X_test, Y_test = model_class.x_train, model_class.y_train, model_class.x_test, model_class.y_test
    print('max pixel value: {}, min pixel value: {}'.format(model_class.max_pixel_value, model_class.min_pixel_value))

    val_size = 5000
    x_val = X_train[:val_size, :, :, :]
    y_val = Y_train[:val_size]
    X_train = X_train[val_size:, :, :, :]
    Y_train = Y_train[val_size:]

    #Train detector -- if already trained, load it
    detector_i_filename = '{}_magnet_detector_i.pt'.format(args.dataset)
    detector_ii_filename = '{}_magnet_detector_ii.pt'.format(args.dataset)

    im_dim = [X_train.shape[1], X_train.shape[2], X_train.shape[3]]
    detector_I = DenoisingAutoEncoder_1(im_dim)
    detector_II = DenoisingAutoEncoder_2(im_dim)
    loader_train = GetLoader(X_train, Y_train)
    datas = Data.DataLoader(loader_train, batch_size=256, shuffle=True, drop_last=False, num_workers=4)
    
    if os.path.isfile('{}{}'.format(magnet_results_dir, detector_i_filename)):
        detector_I.load('{}{}'.format(magnet_results_dir, detector_i_filename))
    else:
        detector_I.train(datas, '{}{}'.format(magnet_results_dir, detector_i_filename), v_noise, clip_min, clip_max, epochs, if_save=True)

    if os.path.isfile('{}{}'.format(magnet_results_dir, detector_ii_filename)):
        detector_II.load('{}{}'.format(magnet_results_dir, detector_ii_filename))
    else:
        detector_II.train(datas, '{}{}'.format(magnet_results_dir, detector_ii_filename), v_noise, clip_min, clip_max, epochs, if_save=True)

    # Refine the normal, noisy and adversarial sets to only include samples for
    # which the original version was correctly classified by the model
    preds_test = classifier.predict(X_test)
    inds_correct = np.where(preds_test.argmax(axis=1) == Y_test.argmax(axis=1))[0]
    print("Number of correctly predict images: %s" % (len(inds_correct)))
    X_test = X_test[inds_correct]
    Y_test = Y_test[inds_correct]
    print("X_test: ", X_test.shape)

    # Make AEs ready
    if type=='error':
        if args.dataset=='cifar':
            detect_I = AEDetector(detector_II.model, p=p1)
            detect_II = AEDetector(detector_II.model, p=p2)
            reformer = SimpleReformer(detector_II.model)
        else:
            detect_I = AEDetector(detector_I.model, p=p1)
            detect_II = AEDetector(detector_II.model, p=p2)
            reformer = SimpleReformer(detector_I.model)
        detector_dict = dict()
        detector_dict["I"] = detect_I
        detector_dict["II"] = detect_II
    elif type=='prob':
        reformer = SimpleReformer(detector_I.model)
        reformer2 = SimpleReformer(detector_II.model)
        detect_I = DBDetector(reformer, reformer2, classifier.model, T=t)
        detector_dict = dict()
        detector_dict["I"] = detect_I

    operator = Operator(torch.as_tensor(x_val), torch.as_tensor(X_test), torch.as_tensor(Y_test), classifier.model, detector_dict, reformer)

    ## Evaluate detector
    #on adversarial attack
    Y_test_copy=Y_test
    X_test_copy=X_test
    for attack in ATTACKS:
        Y_test=Y_test_copy
        X_test=X_test_copy
        results_all = []

        #Prepare data
        # Load adversarial samples
        X_test_adv = np.load('{}{}_{}.npy'.format(adv_data_dir, args.dataset, attack))

        X_test_adv = X_test_adv[inds_correct]

        pred_adv = classifier.predict(X_test_adv)
        # loss, acc_suc = classifier.evaluate(X_test_adv, Y_test, verbose=0)
        acc_suc = np.sum(np.argmax(pred_adv, axis=1) == np.argmax(Y_test, axis=1)) / len(Y_test)

        
        inds_success = np.where(pred_adv.argmax(axis=1) != Y_test.argmax(axis=1))[0]
        inds_fail = np.where(pred_adv.argmax(axis=1) == Y_test.argmax(axis=1))[0]
        X_test_adv_success = X_test_adv[inds_success]
        Y_test_success = Y_test[inds_success]
        X_test_adv_fail = X_test_adv[inds_fail]
        Y_test_fail = Y_test[inds_fail]

        # prepare X and Y for detectors
        X_all = np.concatenate([X_test, X_test_adv])
        Y_all = np.concatenate([np.zeros(len(X_test), dtype=bool), np.ones(len(X_test), dtype=bool)])
        X_success = np.concatenate([X_test[inds_success], X_test_adv_success])
        Y_success = np.concatenate([np.zeros(len(inds_success), dtype=bool), np.ones(len(inds_success), dtype=bool)])
        X_fail = np.concatenate([X_test[inds_fail], X_test_adv_fail])
        Y_fail = np.concatenate([np.zeros(len(inds_fail), dtype=bool), np.ones(len(inds_fail), dtype=bool)])

        # --- get thresholds per detector
        testAttack = AttackData(torch.as_tensor(X_test_adv), torch.as_tensor(np.argmax(Y_test, axis=1)), attack)
        evaluator = Evaluator(operator, testAttack)
        thrs = evaluator.operator.get_thrs(drop_rate)

        #For Y_all 
        Y_all_pred, Y_all_pred_score = test(detector_dict, X_all, thrs)
        acc_all, tpr_all, fpr_all, tp_all, ap_all, fb_all, an_all = evalulate_detection_test(Y_all, Y_all_pred)
        fprs_all, tprs_all, thresholds_all = roc_curve(Y_all, Y_all_pred_score[0])
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
            Y_success_pred, Y_success_pred_score = test(detector_dict, X_success, thrs)
            accuracy_success, tpr_success, fpr_success, tp_success, ap_success, fb_success, an_success = evalulate_detection_test(Y_success, Y_success_pred)
            fprs_success, tprs_success, thresholds_success = roc_curve(Y_success, Y_success_pred_score[0])
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
            Y_fail_pred, Y_fail_pred_score = test(detector_dict, X_fail, thrs)
            accuracy_fail, tpr_fail, fpr_fail, tp_fail, ap_fail, fb_fail, an_fail = evalulate_detection_test(Y_fail, Y_fail_pred)
            fprs_fail, tprs_fail, thresholds_fail = roc_curve(Y_fail, Y_fail_pred_score[0])
            roc_auc_fail = auc(fprs_fail, tprs_fail)

            curr_result = {'type':'fail', 'nsamples': len(inds_fail),	'acc_suc': 0,	\
                    'acc': accuracy_fail, 'tpr': tpr_fail, 'fpr': fpr_fail, 'tp': tp_fail, 'ap': ap_fail, 'fb': fb_fail, 'an': an_fail,	\
                    'tprs': list(fprs_fail), 'fprs': list(tprs_fail),	'auc': roc_auc_fail}
            results_all.append(curr_result)

        import csv
        with open('{}{}_{}.csv'.format(magnet_results_dir, args.dataset, attack), 'w', newline='') as csvfile:
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