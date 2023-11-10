import argparse
from common.util import *
from setup_paths import *
from sklearn.decomposition import FastICA, PCA
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from thundersvm import OneClassSVM
import pickle
import time
import torch.nn.functional as F

def dense(input_shape):
    model = nn.Sequential(
        nn.Linear(input_shape[1], 10),
        nn.Softmax(dim=1)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    return model, criterion, optimizer

def process(Y):
    for i in range(len(Y)):
        if Y[i]==1:
            Y[i]=0
        elif Y[i]==-1:
            Y[i]=1
    return Y

def map(Y):
    # mapped_probabilities = 1 - (Y + 1) / 2
    mapped_probabilities = 0 - Y

    return mapped_probabilities

def batch(curr_model, data, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_samples = data.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    l_out_batches = []
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        batch = torch.from_numpy(data[start_idx:end_idx]).to(device)
        l_out_batch = curr_model(batch).cpu().detach().numpy()
        l_out_batches.append(l_out_batch)

    return l_out_batches

def main(args):
    set_seed(args)

    assert args.dataset in DATASETS, \
        "Dataset parameter must be either {}".format(DATASETS)
    ATTACKS = ATTACK[DATASETS.index(args.dataset)]

    print('Loading the data and model...')
    # Load the model
    if args.dataset == 'mnist':
        from baseline.cnn.cnn_mnist import MNISTCNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.pt'.format(args.dataset))
        classifier = model_class.classifier
        start_indx = 1
    elif args.dataset == 'cifar':
        from baseline.cnn.cnn_cifar10 import CIFAR10CNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.pt'.format(args.dataset))
        classifier = model_class.classifier
        start_indx = 1
    elif args.dataset == 'imagenet':
        from baseline.cnn.cnn_imagenet import ImageNetCNN as myModel
        model_class = myModel(filename='cnn_{}.pt'.format(args.dataset))
        classifier = model_class.classifier
        start_indx = 1
    elif args.dataset == 'svhn':
        from baseline.cnn.cnn_svhn import SVHNCNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.pt'.format(args.dataset))
        classifier = model_class.classifier
        start_indx = 1
       
    # Load the dataset
    X_train, Y_train, X_test, Y_test = model_class.x_train, model_class.y_train, model_class.x_test, model_class.y_test

    #-----------------------------------------------#
    #         Generate layers data Normal           #
    #       Load it if it is already generated      #
    #-----------------------------------------------# 
    # n_layers = len(classifier.layers)
    n_layers = len([i for i in classifier.model.named_children()])
    projector = PCA(n_components=5000)

    #for train
    for l_indx in range(start_indx, n_layers):
        layer_data_path = '{}{}_{}_normal.npy'.format(nic_layers_dir, args.dataset, l_indx)
        if not os.path.isfile(layer_data_path):

            if l_indx+1 == n_layers:
                curr_model = classifier.model
            else:
                curr_model = nn.Sequential(*list(classifier.model.children())[:l_indx+1])
            curr_model.eval()

            l_out = batch(curr_model, X_train, batch_size=1000)
            
            l_out = np.concatenate(l_out, axis=0)
            l_out = l_out.reshape((X_train.shape[0], -1))
            if l_out.shape[1]>5000:
                reduced_activations = projector.fit_transform(l_out)
                np.save(layer_data_path, reduced_activations)
            else:
                np.save(layer_data_path, l_out)
            print(layer_data_path)
    
        #for test
        layer_data_path = '{}{}_{}_test.npy'.format(nic_layers_dir, args.dataset, l_indx)
        if args.dataset =='svhn':
            X_test = X_test[:10000]
        if not os.path.isfile(layer_data_path):
            if l_indx+1 == n_layers:
                curr_model = classifier.model
            else:
                curr_model = nn.Sequential(*list(classifier.model.children())[:l_indx+1])        
            curr_model.eval()
            # print(curr_model)
            
            l_out = batch(curr_model, X_test, batch_size=1000)
            l_out = np.concatenate(l_out, axis=0)        
            l_out = l_out.reshape((X_test.shape[0], -1))
            if l_out.shape[1]>5000:
                # projector = PCA(n_components=5000)
                reduced_activations = projector.transform(l_out)
                np.save(layer_data_path, reduced_activations)
            else:
                np.save(layer_data_path, l_out)
            print(layer_data_path)

        #-----------------------------------------------#
        #        Generate layers data Adv attack        #
        #       Load it if it is already generated      #
        #-----------------------------------------------# 
        for attack in ATTACKS:
            X_adv = np.load('%s%s_%s.npy' % (adv_data_dir, args.dataset, attack))
            if args.dataset =='svhn':
                X_adv = X_adv[:10000]
            # for l_indx in range(start_indx, n_layers):
            layer_data_path = '{}{}_{}_{}.npy'.format(nic_layers_dir, args.dataset, l_indx, attack)
            if not os.path.isfile(layer_data_path):

                if l_indx+1 == n_layers:
                    curr_model = classifier.model
                else:
                    curr_model = nn.Sequential(*list(classifier.model.children())[:l_indx+1]) 
                curr_model.eval()

                l_out = batch(curr_model, X_adv, batch_size=1000)
                l_out = np.concatenate(l_out, axis=0)                    
                l_out = l_out.reshape((X_adv.shape[0], -1))
                if l_out.shape[1]>5000:
                    # projector = PCA(n_components=5000)
                    reduced_activations = projector.transform(l_out)
                    np.save(layer_data_path, reduced_activations)
                else:
                    np.save(layer_data_path, l_out)
                print(layer_data_path)

    #-----------------------------------------------#
    #                  Train PIs                    #
    #-----------------------------------------------# 
    min_features = 5000
    for l_indx in range(start_indx, n_layers):
        layer_data_path = '{}{}_{}_normal.npy'.format(nic_layers_dir, args.dataset, l_indx)
        model_path = '{}{}_{}_pi.model'.format(nic_layers_dir, args.dataset, l_indx)
        pi_predict_normal_path = '{}{}_{}_pi_predict_normal.npy'.format(nic_layers_dir, args.dataset, l_indx)
        pi_decision_normal_path = '{}{}_{}_pi_decision_normal.npy'.format(nic_layers_dir, args.dataset, l_indx)
        if not os.path.isfile(model_path):
            if os.path.isfile(layer_data_path):
                layer_data = np.load(layer_data_path)
                n_features = np.min([min_features, layer_data.shape[1]])
                layer_data = layer_data[:,:n_features]
                clf = OneClassSVM(nu=0.01, kernel="rbf", gamma=1, verbose=True)
                st = time.time()
                clf.fit(layer_data)
                predict_result = clf.predict(layer_data)
                decision_result = clf.decision_function(layer_data)
                #Saving
                clf.save_to_file(model_path)
                # with open(model_path, 'wb') as file:
                #     pickle.dump(clf, file)

                np.save(pi_predict_normal_path, predict_result)
                np.save(pi_decision_normal_path, decision_result)
                et = time.time()
                t=round((et-st)/60, 2)
                print('Training PI on {}, layer {} is completed on {} min(s).'.format(args.dataset, l_indx, t))

    #-----------------------------------------------#
    #                  Train VIs                    #
    #-----------------------------------------------# 
    # Create an empty dictionary to store the models
    model_dict = {}
    for l_indx in range(start_indx, n_layers-1):
        layer_data_path_current = '{}{}_{}_normal.npy'.format(nic_layers_dir, args.dataset, l_indx)
        layer_data_path_next = '{}{}_{}_normal.npy'.format(nic_layers_dir, args.dataset, l_indx+1)
        model_path = '{}{}_{}_vi.model'.format(nic_layers_dir, args.dataset, l_indx)
        vi_train_path = '{}{}_{}_vi_train.npy'.format(nic_layers_dir, args.dataset, l_indx)
        vi_predict_normal_path = '{}{}_{}_vi_predict_normal.npy'.format(nic_layers_dir, args.dataset, l_indx)
        vi_decision_normal_path = '{}{}_{}_vi_decision_normal.npy'.format(nic_layers_dir, args.dataset, l_indx)
        if not os.path.isfile(model_path):
            if os.path.isfile(layer_data_path_current) & os.path.isfile(layer_data_path_next):
                layer_data_current = np.load(layer_data_path_current)
                layer_data_next = np.load(layer_data_path_next)
                n_features_current = np.min([min_features, layer_data_current.shape[1]])
                layer_data_current = layer_data_current[:,:n_features_current]
                n_features_next = np.min([min_features, layer_data_next.shape[1]])
                layer_data_next = layer_data_next[:,:n_features_next]

                # model_current, _, _ = dense(layer_data_current.shape)
                # model_next, _, _ = dense(layer_data_next.shape)
                # layer_data_current = torch.Tensor(layer_data_current)
                # layer_data_next = torch.Tensor(layer_data_next)
                # vi_current = model_current(layer_data_current)
                # vi_next = model_next(layer_data_next)
                # vi_current = vi_current.detach().numpy()
                # vi_next = vi_next.detach().numpy()
                model_current = LogisticRegression(max_iter=200).fit(layer_data_current, np.argmax(Y_train, axis=1))
                model_next = LogisticRegression(max_iter=200).fit(layer_data_next, np.argmax(Y_train, axis=1))
                vi_current = model_current.predict_proba(layer_data_current)
                vi_next = model_next.predict_proba(layer_data_next)
                vi_current_ = F.softmax(torch.from_numpy(vi_current), dim=0).detach().cpu().numpy()
                vi_next_ = F.softmax(torch.from_numpy(vi_next), dim=0).detach().cpu().numpy()
                model_dict[l_indx] = {'current': model_current, 'next': model_next}

                vi_train = np.concatenate((vi_current_, vi_next_), axis=1)
                np.save(vi_train_path, vi_train)

                clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma='scale', verbose=True)
                st = time.time()
                clf.fit(vi_train)
                predict_result = clf.predict(vi_train)
                decision_result = clf.decision_function(vi_train)
                #Saving
                #clf.save_to_file(model_path)
                s = pickle.dumps(clf)
                f = open(model_path, "wb+")
                f.write(s)
                f.close()
                np.save(vi_predict_normal_path, predict_result)
                np.save(vi_decision_normal_path, decision_result)
                et = time.time()
                t=round((et-st)/60, 2)
                print('Training VI on {}, layer {} is completed on {} min(s).'.format(args.dataset, l_indx, t))

    #-----------------------------------------------#
    #                  Train NIC                    #
    # Train detector -- if already trained, load it #
    #-----------------------------------------------# 
    nic_model_path = '{}{}_nic.model'.format(nic_results_dir, args.dataset)
    nic_train_path = '{}{}_nic_train.npy'.format(nic_layers_dir, args.dataset)
    nic_predict_normal_path = '{}{}_nic_predict_normal.npy'.format(nic_layers_dir, args.dataset)
    nic_decision_normal_path = '{}{}_nic_decision_normal.npy'.format(nic_layers_dir, args.dataset)
    if not os.path.isfile(nic_train_path):
        #collect pis
        pis = np.array([])
        for l_indx in range(start_indx, n_layers):
            pi_decision_normal_path = '{}{}_{}_pi_decision_normal.npy'.format(nic_layers_dir, args.dataset, l_indx)
            if os.path.isfile(pi_decision_normal_path):
                pi = np.load(pi_decision_normal_path)
                if pis.size == 0:
                    pis = pi
                else:
                    print("pis.shape:{}, pi.shape:{}".format(pis.shape, pi.shape))
                    # pis = np.concatenate((pis, pi), axis=1)
                    pis = np.column_stack((pis, pi))
        #collect pis
        vis = np.array([])
        for l_indx in range(start_indx, n_layers-1):
            vi_decision_normal_path = '{}{}_{}_vi_decision_normal.npy'.format(nic_layers_dir, args.dataset, l_indx)
            if os.path.isfile(vi_decision_normal_path):
                vi = np.load(vi_decision_normal_path).reshape(-1, 1)
                if vis.size == 0:
                    vis = vi
                else:
                    # vis = np.concatenate((vis, vi), axis=1)
                    vis = np.column_stack((vis, vi))

        #nic train data
        nic_train = np.concatenate((pis, vis), axis=1)
        np.save(nic_train_path, nic_train)
    else:
        nic_train = np.load(nic_train_path)

    train_inds_path='{}{}_train_inds.npy'.format(nic_results_dir, args.dataset)
    if not os.path.isfile(train_inds_path):
        train_inds=random.sample(range(len(nic_train)), int(0.8*len(nic_train)))
        np.save(train_inds_path, train_inds)
    else:
        train_inds = np.load(train_inds_path)
    test_inds=np.asarray(list(set(range(len(nic_train)))-set(train_inds)))
        
    if not os.path.isfile(nic_model_path):
        #train nic
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma='scale', verbose=True)
        st = time.time()
        clf.fit(nic_train[train_inds])
        predict_result = clf.predict(nic_train[train_inds])
        decision_result = clf.decision_function(nic_train[train_inds])
        #Saving
        #clf.save_to_file(nic_model_path)
        s = pickle.dumps(clf)
        f = open(nic_model_path, "wb+")
        f.write(s)
        f.close()
        np.save(nic_predict_normal_path, predict_result)
        np.save(nic_decision_normal_path, decision_result)
        et = time.time()
        t=round((et-st)/60, 2)
        print('Training NIC on {} is completed on {} min(s).'.format(args.dataset, t))



    # Refine the normal, noisy and adversarial sets to only include samples for
    # which the original version was correctly classified by the model
    preds_test = classifier.predict(X_test)
    if args.dataset =='svhn':
        Y_test = Y_test[:10000]
    inds_correct = np.where(preds_test.argmax(axis=1) == Y_test.argmax(axis=1))[0]
    print("Number of correctly predict images: %s" % (len(inds_correct)))
    X_test = X_test[inds_correct]
    Y_test = Y_test[inds_correct]
    print("X_test: ", X_test.shape)

    nic_test = nic_train[test_inds]
 
    #-----------------------------------------------#
    #                 Evaluate NIC                  #
    #-----------------------------------------------# 
    ## Evaluate detector -- on adversarial attack
    for attack in ATTACKS:
        results_all = []

        #get pi decision for each layer
        #a-load pi_model_normal of the layer, b- load/save the decisions of adv
        pis = np.array([])
        for l_indx in range(start_indx, n_layers):
            layer_adv_path = '{}{}_{}_{}.npy'.format(nic_layers_dir, args.dataset, l_indx, attack)
            model_path = '{}{}_{}_pi.model'.format(nic_layers_dir, args.dataset, l_indx)
            pi_decision_adv_path = '{}{}_{}_pi_decision_{}.npy'.format(nic_layers_dir, args.dataset, l_indx, attack)
            if not os.path.isfile(pi_decision_adv_path):
                if os.path.isfile(layer_adv_path) & os.path.isfile(model_path):
                    layer_data = np.load(layer_adv_path)
                    n_features = np.min([min_features, layer_data.shape[1]])
                    layer_data = layer_data[:,:n_features]
                    clf = OneClassSVM(nu=0.01, kernel="rbf", gamma=0.1, verbose=True)
                    clf.load_from_file(model_path)
                    # with open(model_path, 'rb') as file:
                    #     clf = pickle.load(file)

                    decision_result = clf.decision_function(layer_data)
                    np.save(pi_decision_adv_path, decision_result)
            else:
                decision_result = np.load(pi_decision_adv_path)
            
            if pis.size == 0:
                pis = decision_result
            else:
                # pis = np.concatenate((pis, decision_result), axis=1)
                pis = np.column_stack((pis, decision_result))

        #get vi decision for each layer
        #a-load vi_model_normal of the layer, b- load/save the decisions of adv
        vis = np.array([])
        for l_indx in range(start_indx, n_layers-1):
            layer_adv_path_current = '{}{}_{}_{}.npy'.format(nic_layers_dir, args.dataset, l_indx, attack)
            layer_adv_path_next = '{}{}_{}_{}.npy'.format(nic_layers_dir, args.dataset, l_indx+1, attack)
            model_path = '{}{}_{}_vi.model'.format(nic_layers_dir, args.dataset, l_indx)
            vi_decision_adv_path = '{}{}_{}_vi_decision_{}.npy'.format(nic_layers_dir, args.dataset, l_indx, attack)
            if not os.path.isfile(vi_decision_adv_path):
                if os.path.isfile(layer_adv_path_current) & os.path.isfile(layer_adv_path_next) & os.path.isfile(model_path):
                    layer_data_current = np.load(layer_adv_path_current)
                    n_features = np.min([min_features, layer_data_current.shape[1]])
                    layer_data_current = layer_data_current[:,:n_features]
                    layer_data_next = np.load(layer_adv_path_next)
                    n_features = np.min([min_features, layer_data_next.shape[1]])
                    layer_data_next = layer_data_next[:,:n_features]

                    # model_current, _, _ = dense(layer_data_current.shape)
                    # model_next, _, _ = dense(layer_data_next.shape)
                    # layer_data_current = torch.Tensor(layer_data_current)
                    # layer_data_next = torch.Tensor(layer_data_next)
                    # vi_current = model_current(layer_data_current).detach().numpy()
                    # vi_next = model_next(layer_data_next).detach().numpy()
                    model_current = model_dict[l_indx]['current']
                    model_next = model_dict[l_indx]['next']
                    vi_current = model_current.predict_proba(layer_data_current)
                    vi_next = model_next.predict_proba(layer_data_next)
                    vi_current_ = F.softmax(torch.from_numpy(vi_current), dim=0).detach().cpu().numpy()
                    vi_next_ = F.softmax(torch.from_numpy(vi_next), dim=0).detach().cpu().numpy()
                    vi_adv_train = np.concatenate((vi_current_, vi_next_), axis=1)

                    clf = pickle.load(open(model_path, 'rb'))
                    decision_result = clf.decision_function(vi_adv_train).reshape(-1, 1)
                    np.save(vi_decision_adv_path, decision_result)
            else:
                decision_result = np.load(vi_decision_adv_path)
            
            if vis.size == 0:
                vis = decision_result
            else:
                # vis = np.concatenate((vis, decision_result), axis=1)
                vis = np.column_stack((vis, decision_result))

        nic_adv = np.concatenate((pis, vis), axis=1)
        
        #Prepare data
        # Load adversarial samples
        X_test_adv = np.load('%s%s_%s.npy' % (adv_data_dir, args.dataset, attack))

        nic_adv = nic_adv[inds_correct]
        X_test_adv = X_test_adv[inds_correct]

        pred_adv = classifier.predict(X_test_adv)
        # loss, acc_suc = classifier.evaluate(X_test_adv, Y_test, verbose=0)
        acc_suc = np.sum(np.argmax(pred_adv, axis=1) == np.argmax(Y_test, axis=1)) / len(Y_test)

        inds_success = np.where(pred_adv.argmax(axis=1) != Y_test.argmax(axis=1))[0]
        inds_fail = np.where(pred_adv.argmax(axis=1) == Y_test.argmax(axis=1))[0]
        nic_adv_success = nic_adv[inds_success]
        nic_adv_fail = nic_adv[inds_fail]


        # prepare X and Y for detectors
        X_all = np.concatenate([nic_test, nic_adv])
        Y_all = np.concatenate([np.zeros(len(nic_test), dtype=bool), np.ones(len(nic_adv), dtype=bool)])
        X_success = np.concatenate([nic_test, nic_adv_success])
        Y_success = np.concatenate([np.zeros(len(nic_test), dtype=bool), np.ones(len(inds_success), dtype=bool)])
        X_fail = np.concatenate([nic_test, nic_adv_fail])
        Y_fail = np.concatenate([np.zeros(len(nic_test), dtype=bool), np.ones(len(inds_fail), dtype=bool)])

        # --- load nic detector
        clf = pickle.load(open(nic_model_path, 'rb'))

        #For Y_all 
        Y_all_pred = clf.predict(X_all)
        Y_all_pred = process(Y_all_pred)
        Y_all_pred_score = clf.decision_function(X_all)
        print(Y_all_pred_score,"!!!!!!!!!!!!!!!!!!!")
        Y_all_pred_score = map(Y_all_pred_score)

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
            Y_success_pred = clf.predict(X_success)
            Y_success_pred = process(Y_success_pred)
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
            Y_fail_pred = clf.predict(X_fail)
            Y_fail_pred = process(Y_fail_pred)
            Y_fail_pred_score = clf.decision_function(X_fail)
            accuracy_fail, tpr_fail, fpr_fail, tp_fail, ap_fail, fb_fail, an_fail = evalulate_detection_test(Y_fail, Y_fail_pred)
            fprs_fail, tprs_fail, thresholds_fail = roc_curve(Y_fail, Y_fail_pred_score)
            roc_auc_fail = auc(fprs_fail, tprs_fail)

            curr_result = {'type':'fail', 'nsamples': len(inds_fail),	'acc_suc': 0,	\
                    'acc': accuracy_fail, 'tpr': tpr_fail, 'fpr': fpr_fail, 'tp': tp_fail, 'ap': ap_fail, 'fb': fb_fail, 'an': an_fail,	\
                    'tprs': list(fprs_fail), 'fprs': list(tprs_fail),	'auc': roc_auc_fail}
            results_all.append(curr_result)
        
        import csv
        with open('{}{}_{}.csv'.format(nic_results_dir, args.dataset, attack), 'w', newline='') as csvfile:
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