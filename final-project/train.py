import argparse
from torchvision import datasets, transforms, models
from torch import nn, optim
from workspace_utils import active_session
import torch


def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Gradient Descent Learning Rate')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')

    return parser.parse_args()


def load_data():
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_and_valid_transforms = transforms.Compose([transforms.Resize(255),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(
        valid_dir, transform=test_and_valid_transforms)
    test_data = datasets.ImageFolder(
        test_dir, transform=test_and_valid_transforms)

    # Using the image datasets and the transforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=50, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=50)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=50)

    return trainloader, validloader, testloader, train_data

def main():
    # Use GPU if it's available
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    in_arg = get_input_args()
    trainloader, validloader, testloader, train_data = load_data()

    # Build and train your network
    # model = models.in_arg.arch(pretrained=True)
    model = models.vgg16(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(nn.Linear(25088, 256),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(256, 102),
                            nn.LogSoftmax(dim=1))

    model.classifier = classifier

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), in_arg.learning_rate)
    model.to(device)


    with active_session():
        epochs = in_arg.epochs
        steps = 0
        running_loss = 0
        print_every = 5

        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1

                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                log_ps = model.forward(inputs)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in validloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            log_ps = model.forward(inputs)
                            batch_loss = criterion(log_ps, labels)

                            test_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(log_ps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"Test loss: {test_loss/len(validloader):.3f}.. "
                        f"Test accuracy: {accuracy/len(validloader):.3f}")
                    running_loss = 0
                    model.train()

        # Save the checkpoint
        model.class_to_idx = train_data.class_to_idx
        checkpoint = {'class_to_idx': model.class_to_idx,
                      'epochs': in_arg.epochs,
                      'learning_rate': in_arg.learning_rate,
                      'state_optimizer_dict': optimizer.state_dict(),
                      'state_model_dict': model.state_dict()}

        torch.save(checkpoint, 'checkpoint.pth')

if __name__ == "__main__":
    main()
