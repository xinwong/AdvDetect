from common.util import *
from setup_paths import *

class ImageNetCNN:
    def __init__(self, filename="cnn_imagenet.pt"):
        self.filename = filename
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load data
        self.num_classes = 1000
        path_imagenet = '/remote-home/wangxin/Data/imagenet_s/'

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        train_dataset = Datasets.ImageNet(root=path_imagenet, split='train', transform=transform)
        val_dataset = Datasets.ImageNet(root=path_imagenet, split='val', transform=transform)
        train_loader = Data.DataLoader(train_dataset, batch_size=10000, shuffle=True, num_workers=4)
        val_loader = Data.DataLoader(val_dataset, batch_size=5000, shuffle=False, num_workers=4)

        self.x_train, self.y_train = next(iter(train_loader))
        self.x_test, self.y_test = next(iter(val_loader))

        self.x_train, self.y_train = self.x_train.numpy(), F.one_hot(self.y_train, self.num_classes).numpy()
        self.x_test, self.y_test = self.x_test.numpy(), F.one_hot(self.y_test, self.num_classes).numpy()
        self.min_pixel_value, self.max_pixel_value = 0, 1

        self.input_shape = self.x_train.shape[1:]
        print(self.input_shape)

        self.classifier = self.art_classifier(torchvision.models.resnet50(weights='DEFAULT'))
        
    def art_classifier(self, net):
        net.to(self.device)
        # summary(net, input_size=self.input_shape)

        mean = np.asarray((0.485, 0.456, 0.406)).reshape((3, 1, 1))
        std = np.asarray((0.229, 0.224, 0.225)).reshape((3, 1, 1))
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