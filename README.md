# Adversarial Example Detection
This repo implements a set of detection methods for adversarial example detection (AED). It supports experiments on MNIST, CIFAR-10, SVHN, and ImageNet datasets with 7 detection methods: KDE, LID, NSS, FS, Magnet, NIC, and MultiLID.

### Setting Paths
Open `setup_paths.py` and configure the paths and other settings for the detection methods.

### Train Model
To train a model, run `train_model.py -d=<dataset> -b=<batch_size> -e=<epochs>`.

### Generate Adversarial Example
To generate adversarial examples, run `generate_adv.py -d=<dataset>`. After running the program, adversarial examples will be automatically generated and saved for subsequent detection.

### Detector
To run all the detectors, just execute `run_detectors.py`. If you want to run a specific detection method, execute `detect_{method_name}.py -d=<dataset>`, replacing `method_name` with the name of the method you wish to run. For example, `detect_fs.py -d=cifar`.
