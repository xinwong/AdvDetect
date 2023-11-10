import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from common.util import *
from setup_paths import *
import csv
csv.field_size_limit(sys.maxsize)


CSVs_dir=[
    kde_results_dir,
    lid_results_dir,
    nss_results_dir,
    fs_results_dir,
    magnet_results_dir,
    nic_results_dir,
    multiLID_results_dir
]


fn = ['Attack', 'KDE_DR', 'KDE_FPR', 'LID_DR', 'LID_FPR', 'NSS_DR', 'NSS_FPR', \
            'FS_DR', 'FS_FPR', 'MagNet_DR', 'MagNet_FPR', 'NIC_DR', 'NIC_FPR', \
            'MultiLID_DR', 'MultiLID_FPR']


for ds in DATASETS:
    current_ds_s_csv_file = '{}detectors_s_{}.csv'.format(results_path, ds)
    current_ds_f_csv_file = '{}detectors_f_{}.csv'.format(results_path, ds)
    current_ds_all_csv_file = '{}detectors_all_{}.csv'.format(results_path, ds)

    s_dict = [{} for _ in range(len(ALL_ATTACKS))]
    f_dict = [{} for _ in range(len(ALL_ATTACKS))]
    all_dict = [{} for _ in range(len(ALL_ATTACKS))]

    ATTACKS=ATTACK[DATASETS.index(ds)]
    for atk in ATTACKS:
        att_indx = ALL_ATTACKS.index(atk)

        s = {'Attack':atk, 'KDE_DR': '-', 'KDE_FPR': '-', 'LID_DR': '-', 'LID_FPR': '-', 'NSS_DR': '-', 'NSS_FPR': '-', \
            'FS_DR': '-', 'FS_FPR': '-', 'MagNet_DR': '-', 'MagNet_FPR': '-', 'NIC_DR': '-', 'NIC_FPR': '-', \
            'MultiLID_DR': '-', 'MultiLID_FPR': '-'}
        f = {'Attack':atk, 'KDE_DR': '-', 'KDE_FPR': '-', 'LID_DR': '-', 'LID_FPR': '-', 'NSS_DR': '-', 'NSS_FPR': '-', \
            'FS_DR': '-', 'FS_FPR': '-', 'MagNet_DR': '-', 'MagNet_FPR': '-', 'NIC_DR': '-', 'NIC_FPR': '-', \
            'MultiLID_DR': '-', 'MultiLID_FPR': '-'}
        all = {'Attack':atk, 'KDE_DR': '-', 'KDE_FPR': '-', 'LID_DR': '-', 'LID_FPR': '-', 'NSS_DR': '-', 'NSS_FPR': '-', \
            'FS_DR': '-', 'FS_FPR': '-', 'MagNet_DR': '-', 'MagNet_FPR': '-', 'NIC_DR': '-', 'NIC_FPR': '-', \
            'MultiLID_DR': '-', 'MultiLID_FPR': '-'}

        for csv_dir in CSVs_dir:
            csv_dir_indx = CSVs_dir.index(csv_dir)*2
            current_result = []
            csv_file = '{}{}_{}.csv'.format(csv_dir, ds, atk)
            if os.path.isfile(csv_file):
                with open(csv_file, 'r') as file: 
                    data = csv.DictReader(file)
                    for row in data:
                        current_result.append(row)
                    
                    FPR=np.round(100*np.float32(current_result[0]['fpr']), decimals=2)
                    DRS=np.round(100*np.float32(current_result[1]['tpr']), decimals=2)
                    DRF=np.round(100*np.float32(current_result[2]['tpr']), decimals=2)
                    DR=np.round(100*np.float32(current_result[0]['acc']), decimals=2)
                    AUC=np.round(100*np.float32(current_result[0]['auc']), decimals=2)

                    key_dr=fn[csv_dir_indx+1]
                    key_fpr=fn[csv_dir_indx+2]
                    
                    s[key_dr] = DRS
                    s[key_fpr] = FPR
                    f[key_dr] = DRF
                    f[key_fpr] = FPR
                    all[key_dr] = DR
                    all[key_fpr] = AUC
            
        s_dict[att_indx] = s
        f_dict[att_indx] = f
        all_dict[att_indx] = all

    with open(current_ds_s_csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fn)
        writer.writeheader()
        for row in s_dict:
            writer.writerow(row)
    
    with open(current_ds_f_csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fn)
        writer.writeheader()
        for row in f_dict:
            writer.writerow(row)
    
    with open(current_ds_all_csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fn)
        writer.writeheader()
        for row in all_dict:
            writer.writerow(row)

print('Done!')