from common.util import *
from setup_paths import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return out
    
class MNISTCNN:
    def __init__(self, mode='train', filename="cnn_mnist.pt", epochs=50, batch_size=128):
        self.mode = mode # train or load
        self.filename = filename
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available () else 'cpu')

        # Load data
        self.num_classes = 10
        (self.x_train, self.y_train), (self.x_test, self.y_test), self.min_pixel_value, self.max_pixel_value = load_mnist()

        # Swap axes to PyTorch's NCHW format
        self.x_train = np.transpose(self.x_train, (0, 3, 1, 2)).astype(np.float32)
        self.x_test = np.transpose(self.x_test, (0, 3, 1, 2)).astype(np.float32)

        self.input_shape = self.x_train.shape[1:]
        print(self.input_shape)

        if self.mode=='train':
            self.classifier = self.art_classifier(Net())
            # train model
            self.classifier.fit(self.x_train, self.y_train, batch_size=self.batch_size, nb_epochs=self.epochs)
            # save model
            torch.save(self.classifier.model, str(os.path.join(checkpoints_dir, self.filename)))
        elif self.mode=='load':
            self.classifier = self.art_classifier(torch.load(str(os.path.join(checkpoints_dir, self.filename))))
        else:
            raise Exception("Sorry, select the right mode option (train/load)")

        pred = self.classifier.predict(self.x_test, training_mode=False)
        accuracy = np.sum(np.argmax(pred, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("mode option: {}. Accuracy on benign test examples: {}%".format(self.mode, accuracy * 100))

    def art_classifier(self, net):
        net.to(self.device)
        # summary(net, input_size=self.input_shape)
        
        mean = [0.1307]
        std  = [0.3081]
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.01)
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
    