from common.util import *
from setup_paths import *
from baseline.models import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        basic_dropout_rate = 0.1
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(basic_dropout_rate)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(basic_dropout_rate + 0.1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(basic_dropout_rate + 0.2)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(basic_dropout_rate + 0.3)
        )
        self.layer7 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.classification_head(out)
        return out

class CIFAR10CNN:
    def __init__(self, mode='train', filename="cnn_cifar10.pt", epochs=100, batch_size=512):
        self.mode = mode #train or load
        self.filename = filename
        self.epochs = epochs
        self.batch_size = batch_size
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load data
        self.num_classes = 10
        (self.x_train, self.y_train), (self.x_test, self.y_test), self.min_pixel_value, self.max_pixel_value = load_cifar10()
        
        # Swap axes to PyTorch's NCHW format
        self.x_train = np.transpose(self.x_train, (0, 3, 1, 2)).astype(np.float32)
        self.x_test = np.transpose(self.x_test, (0, 3, 1, 2)).astype(np.float32)
        self.input_shape = self.x_train.shape[1:]
        print(self.input_shape)

        if mode=='train':
            # build model
            # self.classifier = ResNet18()
            self.classifier = Net()
            self.classifier = self.art_classifier(self.classifier)
            # train model
            self.classifier.fit(self.x_train, self.y_train, batch_size=self.batch_size, nb_epochs=self.epochs)  
            # save model
            torch.save(self.classifier.model, str(os.path.join(checkpoints_dir, self.filename)))
        elif mode=='load':
            self.classifier = self.art_classifier(torch.load(str(os.path.join(checkpoints_dir, self.filename))))
        else:
            raise Exception("Sorry, select the right mode option (train/load)")
        
        pred = self.classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(pred, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("mode option: {}. Accuracy on benign test examples: {}%".format(self.mode, accuracy * 100))
    
    def art_classifier(self, net):
        net.to(self.device)
        # summary(net, input_size=self.input_shape)
        
        mean = np.asarray((0.4914, 0.4822, 0.4465)).reshape((3, 1, 1))
        std = np.asarray((0.2023, 0.1994, 0.2010)).reshape((3, 1, 1))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        classifier = PyTorchClassifier(
            model=net,
            clip_values=(0, 1),
            loss=criterion,
            optimizer=optimizer,
            input_shape=self.input_shape,
            nb_classes=self.num_classes,
            preprocessing=(mean, std)
        )
        
        return classifier