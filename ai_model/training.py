# train.py
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import os
from DataLoaders import DataLoaders


class Model:
    def __init__(self):
        self.model = models.densenet161(pretrained=True) # Get pretrained model
        self.criterion = None
        self.optimizer = None
        self.device = None
        self.dataModules = DataLoaders()
        #self.configure_model()
        self.labels_map = None
        self.path_to_model = "models/trained_model.pth"


    def set_labels_map(self, labels_map):
        self.labels_map = labels_map


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
        if not os.path.exists('../models'):
            os.makedirs('../models')

    def predict(self, image):
        output = self.model.forward(image)
        output = torch.exp(output)
        probs, classes = output.topk(1, dim=1)
        return classes.item(), probs.item()

    def predict_from_tensor(self, image):
        classes, probs = self.predict(image)
        return self.labels_map[classes], probs

    def load_eval(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(base_dir, self.path_to_model)
        self.model.load_state_dict(torch.load(full_path))
        self.model.eval()


    def process_image(self, image_path):
        img = Image.open(image_path)
        width, height = img.size
        img = img.resize((255, int(255 * (height / width))) if width < height else (int(255 * (width / height)), 255))
        width, height = img.size
        left = (width - 224) / 2
        top = (height - 224) / 2
        right = (width + 224) / 2
        bottom = (height + 224) / 2
        img = img.crop((left, top, right, bottom))
        img = np.array(img)
        img = img.transpose((2, 0, 1))
        img = img / 255
        img[0] = (img[0] - 0.485) / 0.229
        img[1] = (img[1] - 0.456) / 0.224
        img[2] = (img[2] - 0.406) / 0.225
        img = img[np.newaxis, :]
        image = torch.from_numpy(img).float()
        return image
    def predict_from_path(self, path):
        image = self.process_image(path)
        top_class, top_prob = self.predict(image)
        predicted_label = self.labels_map[top_class]
        return top_prob, predicted_label



    def train(self):
        epochs = 10
        for epoch in range(epochs):
            train_loss = 0
            val_loss = 0
            accuracy = 0

            # Training the model
            self.model.train()
            for inputs, labels in self.dataModules.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
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
    def save(self, path='../models/trained_model.pth'):
        # Save the model
        torch.save(self.model.state_dict(), path)
        print("Model saved as models/trained_model.pth")

# ai = Model()
# ai.configure_model()
# ai.load_eval()
# ai.train()
#ai.save()