# train.py
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import os

# Specify transforms using torchvision.transforms
transformations = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load in each dataset and apply transformations
train_set = datasets.ImageFolder("root/label/train", transform=transformations)
val_set = datasets.ImageFolder("root/label/valid", transform=transformations)

# Put into a Dataloader
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True)

# Get pretrained model
model = models.densenet161(pretrained=True)
# Turn off training for their parameters
for param in model.parameters():
    param.requires_grad = False

# Create new classifier for model
classifier_input = model.classifier.in_features
num_labels = 3
classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, 512),
                           nn.ReLU(),
                           nn.Linear(512, num_labels),
                           nn.LogSoftmax(dim=1))
# Replace default classifier with new classifier
model.classifier = classifier

# Find the device available to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set the error function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters())

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Training loop
epochs = 10
for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    accuracy = 0

    # Training the model
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model.forward(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

    # Evaluating the model
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model.forward(inputs)
            valloss = criterion(output, labels)
            val_loss += valloss.item() * inputs.size(0)

            output = torch.exp(output)
            top_p, top_class = output.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    # Get the average loss for the entire epoch
    train_loss /= len(train_loader.dataset)
    valid_loss = val_loss / len(val_loader.dataset)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tAccuracy: {:.6f}'.format(epoch, train_loss, valid_loss, accuracy / len(val_loader)))

# Save the model
torch.save(model.state_dict(), 'models/trained_model.pth')
print("Model saved as models/trained_model.pth")
