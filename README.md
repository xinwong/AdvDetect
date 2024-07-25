## An open-source toolkit for Adversarial Example Detection
This repo implements a set of detection methods for adversarial example detection (AED). So far, it supports experiments on MNIST, CIFAR-10, SVHN, and ImageNet datasets with 7 detection methods: KDE, LID, NSS, FS, Magnet, NIC, and MultiLID. A brief description and reference of these methods can be found below. 

### Setting Paths
Open `setup_paths.py` and configure the paths and other settings for the detection methods.

### Train Model
To train a model, run `train_model.py -d=<dataset> -b=<batch_size> -e=<epochs>`.

### Generate Adversarial Example
To generate adversarial examples, run `generate_adv.py -d=<dataset>`. After running the program, adversarial examples will be automatically generated and saved for subsequent detection. After running the program, adversarial examples will be automatically generated and saved for subsequent detection. Additionally, the perturbation for $L_{\infty}$ are `epsilons = [8/256, 16/256, 32/256, 64/256, 80/256, 128/256]`, for L1 are `epsilons1 = [5, 10, 15, 20, 25, 30, 40]`, and for L2 are `epsilons2 = [0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2]`.

### Detector
To run all the detectors, just execute `run_detectors.py`. If you want to run a specific detection method, execute `detect_{method_name}.py -d=<dataset>`, replacing {method_name} with the name of the method you wish to run. For example, `detect_multiLID.py -d=cifar`.

## Attack & Detection Methods
Attack methods: 

1. FGSM: a one-step gradient sign attack method. **[one-step attack]** 

2. BIM:  an iterative multi-step attack method with equally divided step size. **[multi-step attack]**

3. PGD:   an interactive attack method with uniform initialization, large step size, and perturbation projection. **[the strongest first-order attack]**

4. CW: an optimization-based attack framework that minimize the L2 perturbation magnitude, while targetting maximum classification error. **[L2 optimization attack]**

5. DeepFool: a decision boundary based attack with adaptive perturabtion. **[boundary attack]**

6. Spatial Transformation Attack: spatially transform the samples to be adversarial, different from other attacks that perturb the pixel values. **[spatial attack]**

7. Square Attack: a score-based black-box L2 adversarial attack that selects localized square-shaped updates at random positions at each iteration. **[black-box attack]**

8. Adversarial Patch: a patch with large adversarial perturbations attached to a random square/round area of the image. **[pyshical attack]**

Detection methods: 

1. KDE: KDE reveals that adversarial samples tend to deviate from the normal data manifold in the deep space, resulting in relatively lower kernel densities.

2. LID: This method extracts features from each intermediate layer of a deep neural network and employs the Local Intrinsic Dimensionality metric to detect adversarial samples.

3. NSS: This method proposes to characterize the AEs through the use of natural scene statistics.

4. FS: This method employs feature squeezing to reduce the dimensionality of input samples and then detects adversarial samples based on the changes in the model's output before and after compression.

5. MagNet: This method detects adversarial samples by assessing the ability to reconstruct normal samples while being unable to reconstruct adversarial samples. The AEs can be easily distinguished from those of normal samples using MSCN coefficients as the NSS tool. 

6. NIC: This method proposes a novel technique to extract DNN invariants and use them to perform runtime adversarial sample detection. 

7. MultiLID: Based on a re-interpretation of the LID measure and several simple adaptations, this method surpasses the state-of-the-art on adversarial detection.
