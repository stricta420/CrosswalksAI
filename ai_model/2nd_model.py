from training import Model
import torch


class myModifiedModel(Model):
    def __init__(self):
        super().__init__()

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

        # Save the model
        torch.save(self.model.state_dict(), '../models/trained_model.pth')
        print("Model saved as models/trained_model.pth")

obj = myModifiedModel()
obj.train()