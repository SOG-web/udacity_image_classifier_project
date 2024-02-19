import argparse
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os

def data_transformation(data_directory):
    train_dir = os.path.join(data_directory, "train")
    valid_dir = os.path.join(data_directory, "valid")

    if not os.path.exists(data_directory):
        print(f"Data Directory doesn't exist: {data_directory}")
        raise FileNotFoundError
    if not os.path.exists(train_dir):
        print(f"Train folder doesn't exist: {train_dir}")
        raise FileNotFoundError
    if not os.path.exists(valid_dir):
        print(f"Valid folder doesn't exist: {valid_dir}")
        raise FileNotFoundError

    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(root=valid_dir, transform=valid_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)

    return train_loader, valid_loader, train_data.class_to_idx

def train_model(data_directory, model_arch='vgg19', save_directory='../saved_models', learning_rate=0.003, epochs=3, gpu=False):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, class_to_idx = data_transformation(data_directory)

    if model_arch.startswith("vgg"):
        model = getattr(torchvision.models, model_arch)(pretrained=True)
        in_features = model.classifier[0].in_features
    elif model_arch.startswith("densenet"):
        model = getattr(torchvision.models, model_arch)(pretrained=True)
        in_features = model.classifier.in_features
    else:
        raise ValueError("Unsupported model architecture")

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(in_features, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(2048, 102),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)

    print_every = 20

    for epoch in range(epochs):
        running_train_loss = 0
        for step, (images, labels) in enumerate(train_loader, 1):
            model.train()
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

            if step % print_every == 0 or step == 1 or step == len(train_loader):
                print(f"Epoch: {epoch+1}/{epochs} Batch % Complete: {step*100/len(train_loader):.2f}%")

        model.eval()
        with torch.no_grad():
            print("Validating Epoch....")
            running_valid_loss = 0
            running_accuracy = 0
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                valid_loss = criterion(outputs, labels)
                running_valid_loss += valid_loss.item()
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            average_train_loss = running_train_loss/len(train_loader)
            average_valid_loss = running_valid_loss/len(valid_loader)
            accuracy = running_accuracy/len(valid_loader)
            print(f"Train Loss: {average_train_loss:.3f}")
            print(f"Valid Loss: {average_valid_loss:.3f}")
            print(f"Accuracy: {accuracy*100:.3f}%")

    model.class_to_idx = class_to_idx
    checkpoint = {'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'epochs': epochs,
                  'optim_stat_dict': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'model_arch': model_arch
                  }

    checkpoint_path = os.path.join(save_directory, "checkpoint.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', help="Directory of the training images")
    parser.add_argument('--model_arch', dest='model_arch', help="Type of pre-trained model to be used",
                        default="vgg19", choices=['vgg11', 'vgg13', 'vgg16', 'vgg19', 'densenet121', 'densenet161', 'densenet169', 'densenet201'])
    parser.add_argument('--save_directory', dest='save_directory', help="Directory where the model will be saved after training", default='../saved_models')
    parser.add_argument('--learning_rate', dest='learning_rate', help="Learning rate when training the model. Default is 0.003", default=0.003, type=float)
    parser.add_argument('--epochs', dest='epochs', help="Number of epochs when training the model. Default is 3", default=3, type=int)
    parser.add_argument('--gpu', dest='gpu', help="Use GPU for training if available", action='store_true')
    args = parser.parse_args()

    train_model(args.data_directory, args.model_arch, args.save_directory, args.learning_rate, args.epochs, args.gpu)
