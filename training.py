# train.py
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import os

class DataLoaders:

    def __init__(self):
        self.transformations = self.get_transofrmation()
        self.train_loader, self.val_loader = self.get_data_loaders()

    def get_transofrmation(self, resize=255, center_crop=224,normalize_mean=None,normalize_std=None):
        if normalize_mean is None:
            normalize_mean = [0.485, 0.456, 0.406]
        if normalize_std is None:
            normalize_std = [0.229, 0.224, 0.225]
        transformations = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(center_crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])
        return transformations



    def get_data_loaders(self, train_set_floder="root/label/train", val_set_folder="root/label/valid"):
        # Load in each dataset and apply transformations
        train_set = datasets.ImageFolder(train_set_floder, transform=self.transformations)
        val_set = datasets.ImageFolder(val_set_folder, transform=self.transformations)

        # Put into a Dataloader
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True)
        return train_loader, val_loader

class Model:
    def __init__(self):
        self.model = models.densenet161(pretrained=True) # Get pretrained model
        self.criterion = None
        self.optimizer = None
        self.device = None
        self.dataModules = DataLoaders()
        self.configure_model()




    def configure_model(self):
        # Turn off training for their parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Create new classifier for model
        classifier_input = self.model.classifier.in_features
        num_labels = 3
        classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                                   nn.ReLU(),
                                   nn.Linear(1024, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, num_labels),
                                   nn.LogSoftmax(dim=1))
        # Replace default classifier with new classifier
        self.model.classifier = classifier

        # Find the device available to use
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Set the error function and optimizer
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters())


        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')

    def train(self):
        epochs = 10
        for epoch in range(epochs):
            train_loss = 0
            val_loss = 0
            accuracy = 0

            # Training the model
            self.model.train()
            for inputs, labels in self.dataModules.train_loader:
                inputs, labels = inputs.to(self.model.device), labels.to(self.model.device)
                self.model.optimizer.zero_grad()
                output = self.model.forward(inputs)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            # Evaluating the model
            self.model.eval()
            with torch.no_grad():
                for inputs, labels in self.dataModules.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    output = self.model.forward(inputs)
                    valloss = self.criterion(output, labels)
                    val_loss += valloss.item() * inputs.size(0)

                    output = torch.exp(output)
                    top_p, top_class = output.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            # Get the average loss for the entire epoch
            train_loss /= len(self.dataModules.train_loader.dataset)
            valid_loss = val_loss / len(self.dataModules.val_loader.dataset)

            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tAccuracy: {:.6f}'.format(epoch,
                                                                                                          train_loss,
                                                                                                          valid_loss,
                                                                                                          accuracy / len(
                                                                                                              self.dataModules.val_loader)))

        # Save the model
        torch.save(self.model.state_dict(), 'models/trained_model.pth')
        print("Model saved as models/trained_model.pth")

