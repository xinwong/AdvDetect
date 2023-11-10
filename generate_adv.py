import argparse
import numpy as np
from common.util import *
from setup_paths import *
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, CarliniL2Method, CarliniLInfMethod, ProjectedGradientDescent, DeepFool, SpatialTransformation, SquareAttack, ZooAttack, AdversarialPatchPyTorch

def main(args):
    set_seed(args)

    assert args.dataset in ['mnist', 'cifar', 'svhn', 'imagenet'], \
        "dataset parameter must be either 'mnist', 'cifar', or 'imagenet'"
    print('Dataset: %s' % args.dataset)

    if args.dataset == 'mnist':
        from baseline.cnn.cnn_mnist import MNISTCNN as model
        model_mnist = model(mode='load', filename='cnn_{}.pt'.format(args.dataset))
        classifier = model_mnist.classifier
        epsilons=[8/256, 16/256, 32/256, 64/256, 80/256, 128/256]
        epsilons1=[5, 10, 15, 20, 25, 30, 40]
        epsilons2=[0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2]
        eps_sa=0.3
        x_test = model_mnist.x_test
        y_test = model_mnist.y_test
        translation = 10
        rotation = 60

    elif args.dataset == 'cifar':
        from baseline.cnn.cnn_cifar10 import CIFAR10CNN as model
        model_cifar = model(mode='load', filename='cnn_{}.pt'.format(args.dataset))
        classifier = model_cifar.classifier
        epsilons=[8/256, 16/256, 32/256, 64/256, 80/256, 128/256]
        epsilons1=[5, 10, 15, 20, 25, 30, 40]
        epsilons2=[0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2]
        eps_sa=0.125
        x_test = model_cifar.x_test
        y_test = model_cifar.y_test
        translation = 8
        rotation = 30

    elif args.dataset == 'svhn':
        from baseline.cnn.cnn_svhn import SVHNCNN as model
        model_svhn = model(mode='load', filename='cnn_{}.pt'.format(args.dataset))
        classifier = model_svhn.classifier
        epsilons=[8/256, 16/256, 32/256, 64/256, 80/256, 128/256]
        epsilons1=[5, 10, 15, 20, 25, 30, 40]
        epsilons2=[0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2]
        eps_sa=0.125
        x_test = model_svhn.x_test
        y_test = model_svhn.y_test
        translation = 10
        rotation = 60

    elif args.dataset == 'imagenet':
        from baseline.cnn.cnn_imagenet import ImageNetCNN as model
        model_imagenet = model(filename='cnn_{}.pt'.format(args.dataset))
        classifier = model_imagenet.classifier
        epsilons=[8/256, 16/256, 32/256, 64/256, 80/256, 128/256]
        epsilons1=[5, 10, 15, 20, 25, 30, 40]
        epsilons2=[0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2]
        eps_sa=0.125
        x_test = model_imagenet.x_val
        y_test = model_imagenet.y_val
        translation = 8
        rotation = 30

    # #FGSM
    # for e in epsilons:
    #     attack = FastGradientMethod(estimator=classifier, eps=e, eps_step=0.01, batch_size=256)
    #     adv_data = attack.generate(x=x_test)
    #     adv_file_path = adv_data_dir + args.dataset + '_fgsm_' + str(e) + '.npy'
    #     np.save(adv_file_path, adv_data)
    #     print('Done - {}'.format(adv_file_path))

    # #BIM
    # for e in epsilons:
    #     attack = BasicIterativeMethod(estimator=classifier, eps=e, eps_step=0.01, batch_size=32, max_iter=int(e*256*1.25))
    #     adv_data = attack.generate(x=x_test)
    #     adv_file_path = adv_data_dir + args.dataset + '_bim_' + str(e) + '.npy'
    #     np.save(adv_file_path, adv_data)
    #     print('Done - {}'.format(adv_file_path))
    
    # #PGD1
    # for e in epsilons1:
    #     attack = ProjectedGradientDescent(estimator=classifier, norm=1, eps=e, eps_step=4, batch_size=32)
    #     adv_data = attack.generate(x=x_test)
    #     adv_file_path = adv_data_dir + args.dataset + '_pgd1_' + str(e) + '.npy'
    #     np.save(adv_file_path, adv_data)
    #     print('Done - {}'.format(adv_file_path))
    
    # #PGD2
    # for e in epsilons2:
    #     attack = ProjectedGradientDescent(estimator=classifier, norm=2, eps=e, eps_step=0.1, batch_size=32)
    #     adv_data = attack.generate(x=x_test)
    #     adv_file_path = adv_data_dir + args.dataset + '_pgd2_' + str(e) + '.npy'
    #     np.save(adv_file_path, adv_data)
    #     print('Done - {}'.format(adv_file_path))
    
    # #PGDInf
    # for e in epsilons:
    #     attack = ProjectedGradientDescent(estimator=classifier, norm=np.inf, eps=e, eps_step=0.01, batch_size=32)
    #     adv_data = attack.generate(x=x_test)
    #     adv_file_path = adv_data_dir + args.dataset + '_pgdi_' + str(e) + '.npy'
    #     np.save(adv_file_path, adv_data)
    #     print('Done - {}'.format(adv_file_path))

    # #CWi
    # attack = CarliniLInfMethod(classifier=classifier, max_iter=20)
    # adv_data = attack.generate(x=x_test)
    # adv_file_path = adv_data_dir + args.dataset + '_cwi.npy'
    # np.save(adv_file_path, adv_data)
    # print('Done - {}'.format(adv_file_path))

    # #CW2 - SLOW
    # attack = CarliniL2Method(classifier=classifier, max_iter=10, confidence=10)
    # adv_data = attack.generate(x=x_test)
    # adv_file_path = adv_data_dir + args.dataset + '_cw2.npy'
    # np.save(adv_file_path, adv_data)
    # print('Done - {}'.format(adv_file_path))

    #DF
    attack = DeepFool(classifier=classifier)
    adv_data = attack.generate(x=x_test)
    adv_file_path = adv_data_dir + args.dataset + '_df.npy'
    np.save(adv_file_path, adv_data)
    print('Done - {}'.format(adv_file_path))

    #Spatial transofrmation attack
    attack = SpatialTransformation(classifier=classifier, max_translation=translation, max_rotation=rotation)
    adv_data = attack.generate(x=x_test)
    adv_file_path = adv_data_dir + args.dataset + '_sta.npy'
    np.save(adv_file_path, adv_data)
    print('Done - {}'.format(adv_file_path))

    #Square Attack
    attack = SquareAttack(estimator=classifier, max_iter=200, eps=eps_sa)
    adv_data = attack.generate(x=x_test, y=y_test)
    adv_file_path = adv_data_dir + args.dataset + '_sa.npy'
    np.save(adv_file_path, adv_data)
    print('Done - {}'.format(adv_file_path))

    #ZOO attack
    # attack = ZooAttack(classifier=classifier, batch_size=32)
    # adv_data = attack.generate(x=x_test, y=y_test)
    # adv_file_path = adv_data_dir + args.dataset + '_zoo.npy'
    # np.save(adv_file_path, adv_data)
    # print('Done - {}'.format(adv_file_path))

    #Adversarial Patch attack
    if args.dataset != 'mnist':
        attack = AdversarialPatchPyTorch(
                    estimator=classifier,
                    rotation_max=22.5,
                    scale_min=0.4,
                    scale_max=1.0,
                    learning_rate=1,
                    batch_size=32,
                    max_iter=1000,
                    patch_shape=(3, 100, 100),
                    verbose=True,
                    optimizer='pgd'
                )
        patch, patch_mask= attack.generate(x=x_test, y=y_test)
        save_image(torch.from_numpy(patch * patch_mask), 'ap.jpg')
        adv_data = attack.apply_patch(x=x_test, scale=0.2)
        adv_file_path = adv_data_dir + args.dataset + '_ap.npy'
        np.save(adv_file_path, adv_data)
        print('Done - {}'.format(adv_file_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True, type=str, help="Dataset to use; either 'mnist', 'cifar', or 'imagenet'")
    parser.add_argument('-s', '--seed', help='set seed for model', default=123, type=int)
    args = parser.parse_args()
    main(args)