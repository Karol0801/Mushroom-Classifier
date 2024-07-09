import os
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import wandb


DIRECTORY = r"data\Mushrooms"


class OurMushroom:
    def __init__(self, image_paths, class_2_idx, transform=None):
        self.transform = transform
        self.image_paths = image_paths
        self.class_2_idx = class_2_idx

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.class_2_idx[self.image_paths[idx].split(os.path.sep)[-2]]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.image_paths)


def initialize_wandb():
    wandb.login()
    wandb.init(project='mushroom-classification', config={
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_classes': 12
    })
    print("W&B Initialized")


def get_data_transforms():
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform_train, transform_test


def prepare_data(transform_train, transform_test):
    image_paths = []
    for rel_path1 in os.listdir(DIRECTORY):
        for rel_path2 in tqdm(os.listdir(os.path.join(DIRECTORY, rel_path1))):
            full_path = os.path.join(DIRECTORY, rel_path1, rel_path2)
            image_paths.append(full_path)

    random.shuffle(image_paths)
    test_image_paths = random.sample(image_paths, k=int(len(image_paths) * 0.1))
    train_image_paths = [x for x in image_paths if x not in test_image_paths]
    print(f"test: {len(test_image_paths)}\ntrain: {len(train_image_paths)}")

    class_name = sorted(os.listdir(DIRECTORY))
    class_2_idx = {cls_name: number for number, cls_name in enumerate(class_name)}
    print(class_2_idx)

    train_data = OurMushroom(train_image_paths, class_2_idx, transform=transform_train)
    test_data = OurMushroom(test_image_paths, class_2_idx, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

    return train_loader, test_loader, class_2_idx, train_image_paths


def initialize_model(device, num_classes):
    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)

    return model, optimizer, criterion


def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device, num_epochs):
    best_accuracy = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, labels in tqdm(train_loader):
            optimizer.zero_grad()
            labels, images = labels.to(device), images.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = train_correct * 100.0 / len(train_loader.dataset)

        model.eval()
        test_loss, test_correct = 0.0, 0

        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                labels, images = labels.to(device), images.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = test_correct * 100.0 / len(test_loader.dataset)

        if test_accuracy >= best_accuracy:
            print(f"Detected better accuracy: {test_accuracy}. Replaced with {best_accuracy}")
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), "model/original_model_88.pth")

        print(f"Epoch: {epoch + 1} / {num_epochs}\n"
              f"Train Loss: {train_loss:.2f}, Train Accuracy: {train_accuracy:.2f}\n"
              f"Test Loss: {test_loss:.2f}, Test Accuracy: {test_accuracy:.2f}\n")

        # Log metrics to W&B
        wandb.log({
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
        })

    print("Training complete")


def main():
    initialize_wandb()
    transform_train, transform_test = get_data_transforms()
    train_loader, test_loader, class_2_idx, train_image_paths = prepare_data(transform_train, transform_test)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    model, optimizer, criterion = initialize_model(device, wandb.config.num_classes)
    train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device, wandb.config.epochs)

    wandb.finish()
    print("W&B run finished")


if __name__ == '__main__':
    main()
