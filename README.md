## An open-source toolkit for Adversarial Example Detection (AED)
This repo implements a set of detection methods for adversarial example detection (AED). So far, it supports experiments on MNIST, CIFAR-10, SVHN, and ImageNet datasets with 7 detection methods: KDE, LID, NSS, FS, Magnet, NIC, and MultiLID. A brief description and reference of these methods can be found below. 

### Setting Paths
Open `setup_paths.py` and configure the paths and other settings for the detection methods.

### Train Model
To train a model, run `train_model.py -d=<dataset> -b=<batch_size> -e=<epochs>`.

### Generate Adversarial Example
To generate adversarial examples, run `generate_adv.py -d=<dataset>`. After running the program, adversarial examples will be automatically generated and saved for subsequent detection. After running the program, adversarial examples will be automatically generated and saved for subsequent detection. Additionally, the perturbation for $L_{\infty}$ are `epsilons = [8/256, 16/256, 32/256, 64/256, 80/256, 128/256]`, for L1 are `epsilons1 = [5, 10, 15, 20, 25, 30, 40]`, and for L2 are `epsilons2 = [0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2]`.

### Detector
To run all the detectors, just execute `run_detectors.py`. If you want to run a specific detection method, execute `detect_{method_name}.py -d=<dataset>`, replacing {method_name} with the name of the method you wish to run. For example, `detect_multiLID.py -d=cifar`.

### Attack & Detection Methods
Attack methods: 

1. FGSM<sup>[8]</sup>: a one-step gradient sign attack method. **[one-step attack]** 

2. BIM<sup>[9]</sup>:  an iterative multi-step attack method with equally divided step size. **[multi-step attack]**

3. PGD<sup>[10]</sup>:   an interactive attack method with uniform initialization, large step size, and perturbation projection. **[the strongest first-order attack]**

4. CW<sup>[11]</sup>: an optimization-based attack framework that minimize the L2 perturbation magnitude, while targetting maximum classification error. **[L2 optimization attack]**

5. DeepFool<sup>[12]</sup>: a decision boundary based attack with adaptive perturabtion. **[boundary attack]**

6. Spatial Transformation Attack<sup>[13]</sup>: spatially transform the samples to be adversarial, different from other attacks that perturb the pixel values. **[spatial attack]**

7. Square Attack<sup>[14]</sup>: a score-based black-box L2 adversarial attack that selects localized square-shaped updates at random positions at each iteration. **[black-box attack]**

8. Adversarial Patch<sup>[15]</sup>: a patch with large adversarial perturbations attached to a random square/round area of the image. **[pyshical attack]**

Detection methods: 

1. [KDE](https://arxiv.org/pdf/1703.00410)<sup>[1]</sup>: KDE reveals that adversarial samples tend to deviate from the normal data manifold in the deep space, resulting in relatively lower kernel densities.

2. [LID](https://arxiv.org/pdf/1801.02613)<sup>[2]</sup>: This method extracts features from each intermediate layer of a deep neural network and employs the Local Intrinsic Dimensionality metric to detect adversarial samples.

3. [NSS](https://ieeexplore.ieee.org/document/9206959)<sup>[3]</sup>: This method proposes to characterize the AEs through the use of natural scene statistics.

4. [FS](https://arxiv.org/abs/1704.01155)<sup>[4]</sup>: This method employs feature squeezing to reduce the dimensionality of input samples and then detects adversarial samples based on the changes in the model's output before and after compression.

5. [MagNet](https://arxiv.org/abs/1705.09064)<sup>[5]</sup>: This method detects adversarial samples by assessing the ability to reconstruct normal samples while being unable to reconstruct adversarial samples. The AEs can be easily distinguished from those of normal samples using MSCN coefficients as the NSS tool. 

6. [NIC](https://www.cs.purdue.edu/homes/taog/docs/NDSS19.pdf)<sup>[6]</sup>: This method proposes a novel technique to extract DNN invariants and use them to perform runtime adversarial sample detection. 

7. [MultiLID](https://arxiv.org/pdf/2212.06776)<sup>[7]</sup>: Based on a re-interpretation of the LID measure and several simple adaptations, this method surpasses the state-of-the-art on adversarial detection.

## References
[1] Feinman R, Curtin R R, Shintre S, et al. Detecting adversarial samples from artifacts [J]. arXiv preprint arXiv:170300410, 2017.

[2] Ma X, Li B, Wang Y, et al. Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality[C]//International Conference on Learning Representations, 2018.

[3] Kherchouche A, Fezza S A, Hamidouche W, et al. Detection of adversarial examples in deep neural networks with natural scene statistics[C]//2020 International Joint Conference on Neural Networks (IJCNN). IEEE, 2020: 1-7.

[4] Xu W, Evans D, Qi Y. Feature squeezing: Detecting adversarial examples in deep neural networks [J]. arXiv preprint arXiv:170401155, 2017.

[5] Meng D, Chen H. Magnet: a two-pronged defense against adversarial examples[C]// Proceedings of the 2017 ACM SIGSAC conference on computer and communications security, 2017.

[6] Ma S, Liu Y, Tao G, et al. Nic: Detecting adversarial samples with neural network invariant checking[C]// 26th Annual Network And Distributed System Security Symposium (NDSS 2019), 2019. Internet Soc.

[7] Lorenz P, Keuper M, Keuper J. Unfolding Local Growth Rate Estimates for (Almost) Perfect Adversarial Detection[J]. arXiv preprint arXiv:2212.06776, 2022.

[8] Goodfellow I J, Shlens J, Szegedy C. Explaining and harnessing adversarial examples[C]//International Conference on Learning Representations, 2015.

[9] Kurakin A, Goodfellow I J, Bengio S. Adversarial Examples in the Physical World[J]. Artificial Intelligence Safety and Security, 2018: 99-112.

[10] Madry A, Makelov A, Schmidt L, et al. Towards deep learning models resistant to adversarial attacks[C]//International Conference on Learning Representations, 2018.

[11] Carlini N, Wagner D. Towards evaluating the robustness of neural networks[C]//2017 IEEE symposium on security and privacy (sp). Ieee, 2017: 39-57.

[12] Moosavi-Dezfooli S M, Fawzi A, Frossard P. Deepfool: a simple and accurate method to fool deep neural networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 2574-2582.

[13] Engstrom L, Tran B, Tsipras D, et al. Exploring the landscape of spatial robustness[C]//International Conference on Learning Representations, 2019.

[14] Andriushchenko M, Croce F, Flammarion N, et al. Square attack: a query-efficient black-box adversarial attack via random search[C]//European Conference on Computer Vision, 2020.

[15] Brown T B, Mané D, Roy A, et al. Adversarial patch[J]. arXiv preprint arXiv:1712.09665, 2017.
