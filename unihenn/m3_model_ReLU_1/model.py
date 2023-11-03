import torch
from torchvision import datasets, transforms
import numpy as np

from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.001
training_epochs = 15
batch_size = 32

image_size = 28

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./../Data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./../Data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

print("train dataset:", train_dataset.data.shape)
print("test dataset :", test_dataset.data.shape)

import cnn_model

def train(model, train_loader, optimizer):
    model.train()
    train_loss = 0

    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        hypothesis = model(image)
        loss = criterion(hypothesis, label)
        loss.backward()
        optimizer.step()

        train_loss += loss / len(train_loader)

    return train_loss

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            loss = criterion(output, label)
            test_loss += loss / len(test_loader)

            prediction = output.max(1, keepdim = True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


model = cnn_model.CNN().to(device)
import torchsummary

torchsummary.summary(model, (1, image_size, image_size))

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_keeper = {'train':[], 'test':[]}

for epoch in range(training_epochs):
    train_loss = 0.0
    test_loss = 0.0

    '''
    Training phase
    '''
    train_loss = train(model, train_loader, optimizer)
    train_loss = train_loss.item()
    loss_keeper['train'].append(train_loss)
    
    '''
    Test phase
    '''
    test_loss, test_accuracy = evaluate(model, test_loader)
    test_loss = test_loss.item()
    loss_keeper['test'].append(test_loss)


    print("Epoch:%2d/%2d.. Training loss: %f.. Test loss: %f.. Test Accuracy: %f" 
          %(epoch + 1, training_epochs, train_loss, test_loss, test_accuracy))
    
train_loss_data = loss_keeper['train']
test_loss_data = loss_keeper['test']

plt.plot(train_loss_data, label = "Training loss")
plt.plot(test_loss_data, label = "Test loss")

plt.legend(), plt.grid(True)
plt.xlim(-2,training_epochs+3)
plt.show()

print(model)
torch.save(model.state_dict(), cnn_model.model_file)


def predict_from_loader(model, test_loader, num_images=10):
    """Predict classes for a number of images from the test_loader."""
    predictions = []
    ground_truths = []
    images_sampled = []

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            output = model(image)
            _, prediction = torch.max(output, 1)

            predictions.extend(prediction.cpu().tolist())
            ground_truths.extend(label.tolist())
            images_sampled.extend(image.cpu())

            if len(predictions) >= num_images:
                break

    return images_sampled[:num_images], predictions[:num_images], ground_truths[:num_images]


# Get predictions for 10 images from test_loader (you can change the number if needed)
sampled_images, predictions, ground_truths = predict_from_loader(model, test_loader, 10)

# Print results
for i in range(len(predictions)):
    print(f"Image {i + 1}: Predicted class = {predictions[i]}, True class = {ground_truths[i]}")

# If you want to visualize the predictions along with the images
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Display images
imshow(make_grid(sampled_images))