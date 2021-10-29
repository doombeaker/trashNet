import os
import torch
import numpy as np
from torchvision import transforms, datasets
import torchvision
import time
import copy

saved_file = "./trash_classifier_40_model.pth"

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

data_dir = "datasets40"
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ["train", "val"]
}
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=256, shuffle=True, num_workers=16
    )
    for x in ["train", "val"]
}

dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
class_names = image_datasets["train"].classes

device = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE USED:", device)

model_ft = torchvision.models.mobilenet_v2(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False
model_ft.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.2), torch.nn.Linear(model_ft.last_channel, 40),
)

model_ft.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer_ft = torch.optim.AdamW(model_ft.classifier.parameters(), lr=0.01)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer_ft, step_size=10, gamma=0.1
)


def train(model, loss_fn, optimizer, scheduler, epoch, num_epoches=50):
    print("Epoch {0}/{1}".format(epoch + 1, num_epoches))
    print("-" * 10)

    running_loss = 0.0
    running_corrects = 0
    model.train()
    for inputs, labels in dataloaders["train"]:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels).item()

    scheduler.step()

    loss = running_loss / dataset_sizes["train"]
    acc = running_corrects / dataset_sizes["train"]
    return loss, acc


def test(model, loss_fn):
    running_loss = 0.0
    running_corrects = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in dataloaders["val"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()

    loss = running_loss / dataset_sizes["val"]
    acc = running_corrects / dataset_sizes["val"]
    return loss, acc


def train_and_test(model, loss_fn, optimizer, scheduler, num_epoches=5):
    since = time.time()

    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epoches):
        train_loss, train_acc = train(
            model, loss_fn, optimizer, scheduler, epoch, num_epoches
        )
        print("Training Loss: {0:.4f} Acc: {1:.4f}".format(train_loss, train_acc))

        test_loss, test_acc = test(model, loss_fn)
        print("Test Loss: {0:.4f} Acc: {1:.4f}".format(test_loss, test_acc))

        if test_acc > best_acc:
            best_acc = test_acc
            best_model = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {0:.0f} min {1:.0f} seconds.".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best acc:{0:.4f}".format(best_acc))

    model.load_state_dict(best_model)
    return model


if __name__ == "__main__":
    if os.path.exists(saved_file):
        print("Load existing file...")
        params = torch.load(saved_file)
        model_ft.load_state_dict(params)

    model_ft = train_and_test(model_ft, loss_fn, optimizer_ft, exp_lr_scheduler)
    torch.save(model_ft.state_dict(), saved_file)
