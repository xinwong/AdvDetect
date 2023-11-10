# Common
checkpoints_dir = './checkpoints/'
adv_data_dir = './adv_data/'
DATASETS = ['mnist', 'cifar', 'svhn']
ATTACK = [['fgsm_0.03125', 'fgsm_0.0625','fgsm_0.125',\
            'bim_0.03125', 'bim_0.0625','bim_0.125',\
            'pgdi_0.03125', 'pgdi_0.0625','pgdi_0.125',\
            'cwi',\
            'df',\
            'sta',\
            'sa'
            ],
            ['fgsm_0.03125', 'fgsm_0.0625','fgsm_0.125',\
            'bim_0.03125', 'bim_0.0625','bim_0.125',\
            'pgdi_0.03125', 'pgdi_0.0625','pgdi_0.125',\
            'cwi',\
            'df',\
            'sta',\
            'sa',\
            'ap'
            ],
            ['fgsm_0.03125', 'fgsm_0.0625','fgsm_0.125',\
            'bim_0.03125', 'bim_0.0625','bim_0.125',\
            'pgdi_0.03125', 'pgdi_0.0625','pgdi_0.125',\
            'cwi',\
            'df',\
            'sta',\
            'sa',\
            'ap'
            ],
            ['fgsm_0.03125', 'fgsm_0.0625','fgsm_0.125',\
            'bim_0.03125', 'bim_0.0625','bim_0.125',\
            'pgdi_0.03125', 'pgdi_0.0625','pgdi_0.125',\
            'cwi',\
            'df',\
            'sta',\
            'sa',\
            'ap'
            ]
        ]

ALL_ATTACKS = ['fgsm_0.03125', 'fgsm_0.0625','fgsm_0.125',\
            'bim_0.03125', 'bim_0.0625','bim_0.125',\
            'pgdi_0.03125', 'pgdi_0.0625','pgdi_0.125',\
            'cwi',\
            'df',\
            'sta',\
            'sa',\
            'ap'
           ]
fieldnames = ['type',	'nsamples',	'acc_suc',	'acc',	'tpr',	'fpr',	'tp',	'ap',	'fb',	'an',	'tprs',	'fprs',	'auc']
env_param = 'env /remote-home/wangxin/miniconda3/envs/adv/bin/python -- ' 
detectors_dir = './'
results_path = './results/'

# FS detector
fs_results_dir = './results/fs/'

# KDE detector
# Optimal KDE bandwidths
BANDWIDTHS = {'mnist': 1.20, 'cifar': 0.26, 'svhn': 1.00, 'imagenet': 0.26}
kde_results_dir = './results/kde/'

# LID detector
k_lid = [20, 30, 30, 30]
lid_results_dir = './results/lid/'

# multiLID detector
k_multiLID = [20, 30, 29, 30]
multiLID_results_dir = './results/multiLID/'

# MagNet detector
magnet_results_dir = './results/magnet/'

# NSS detector
pgd_percent = [[0.02, 0.1, 0.18, 0.3, 0.3, 0.1], [0.1, 0.3, 0.1, 0.1, 0.1, 0.1], [0.3, 0.3, 0.1, 0.1, 0.1, 0.1], [0.3, 0.3, 0.1, 0.1, 0.1, 0.1]]
nss_results_dir = './results/nss/'

# NIC detector
nic_results_dir = './results/nic/'
nic_layers_dir = './results/nic/layers/'
