import os
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Tuple, Dict, List
import wandb

TEST_SPLIT_RATIO = 0.1
BATCH_SIZE = 32


class ModifiedResNet(nn.Module):
    def __init__(self, original_model, num_classes):
        super(ModifiedResNet, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.fc1 = nn.Linear(original_model.fc.in_features, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class OurMushroom:
    def __init__(self, image_paths, class_2_idx, transform=None):
        self.transform = transform
        self.image_paths = image_paths
        self.class_2_idx = class_2_idx

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
        except IOError as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            return None, None

        label = self.class_2_idx[self.image_paths[idx].split(os.path.sep)[-2]]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.image_paths)


def initialize_wandb():
    try:
        wandb.login()
        wandb.init(project='mushroom-classification', config={
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_classes': 12
        })
        print("W&B Initialized")
    except Exception as e:
        print(f"Error initializing W&B: {e}")


def get_data_transforms():
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform_train, transform_test


def prepare_data(transform_train, transform_test):
    directory = r"data\Mushrooms"
    image_paths = []
    for rel_path1 in os.listdir(directory):
        for rel_path2 in tqdm(os.listdir(os.path.join(directory, rel_path1))):
            full_path = os.path.join(directory, rel_path1, rel_path2)
            image_paths.append(full_path)

    random.shuffle(image_paths)
    train_image_paths, test_image_paths = train_test_split(image_paths, test_size=TEST_SPLIT_RATIO, random_state=42)

    print(f"test: {len(test_image_paths)}\ntrain: {len(train_image_paths)}")

    class_name = sorted(os.listdir(directory))
    class_2_idx = {cls_name: number for number, cls_name in enumerate(class_name)}
    print(class_2_idx)

    train_data = OurMushroom(train_image_paths, class_2_idx, transform=transform_train)
    test_data = OurMushroom(test_image_paths, class_2_idx, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_data, BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, class_2_idx, train_image_paths


def initialize_model(device, num_classes):
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")

    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    model = ModifiedResNet(model, num_classes)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=wandb.config.learning_rate, momentum=0.9, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss().to(device)

    return model, optimizer, criterion


def train_one_epoch(model, train_loader, criterion, optimizer, device):
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
    return train_loss, train_accuracy


def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss, test_correct = 0.0, 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            labels, images = labels.to(device), images.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    test_loss /= len(test_loader)
    test_accuracy = test_correct * 100.0 / len(test_loader.dataset)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return test_loss, test_accuracy, precision, recall, f1


def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device, num_epochs, patience=10):
    best_accuracy = 0
    best_loss = float('inf')
    trigger_times = 0

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy, precision, recall, f1 = evaluate(model, test_loader, criterion, device)

        if test_accuracy >= best_accuracy and test_loss <= best_loss:
            print(f"Detected better model. Test accuracy: {test_accuracy}, Test loss: {test_loss}")
            best_accuracy = test_accuracy
            best_loss = test_loss
            model_filename = f"model/resnet101_{best_accuracy:.2f}.pth"
            torch.save(model.state_dict(), model_filename)
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping triggered. No improvement in {patience} epochs.")
                break

        print(f"Epoch: {epoch + 1} / {num_epochs}\n"
              f"Train Loss: {train_loss:.2f}, Train Accuracy: {train_accuracy:.2f}\n"
              f"Test Loss: {test_loss:.2f}, Test Accuracy: {test_accuracy:.2f}\n"
              f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}\n")

        # Log metrics to W&B
        wandb.log({
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
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
