# adversarial-detection
This is a repo for adversarial examples detection. It supports the MNIST, CIFAR-10, SVHN, and ImageNet datasets, and currently includes detection methods such as KDE, LID, NSS, FS, Magnet, NIC, and MultiLID.

## Setting Paths
Open `setup_paths.py` and configure the paths and other settings for the detection methods.

## Train model
To train a model, run `train_model.py -d=<dataset> -b=<batch_size> -e=<epochs>`.

## Generate adversarial example
To generate adversarial examples, run `generate_adv.py -d=<dataset>`.

## Detection
To run all the detector, just execute `run_detectors.py`. If you want to run a specific detection method, execute `detect_{method_name}.py`, replacing `method_name` with the name of the method you wish to run.
