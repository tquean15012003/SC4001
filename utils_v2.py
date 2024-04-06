
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as v2
import numpy as np

from enum import Enum
from torchvision import models
from torchvision.datasets import Flowers102
from torch.utils.data import DataLoader, random_split

    

# Define transforms for data augmentation and normalization
data_transforms_train = v2.Compose(
    [
        v2.PILToTensor(),
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),  # to float32 in [0, 1]
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # typically from ImageNe
    ]    
)

data_transforms_test = v2.Compose(
    [
        v2.PILToTensor(),
        v2.CenterCrop(size=(224, 224)),
        v2.ToDtype(torch.float32, scale=True),  # to float32 in [0, 1]
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # typically from ImageNet
    ]
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cutmix = v2.CutMix(num_classes=102)
mixup = v2.MixUp(num_classes=102)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

def collate_fn(batch):
    return cutmix_or_mixup(*torch.utils.data.default_collate(batch))


def get_train_val_test_loader(mixup: bool = False):
    train_dataset = Flowers102(
        root="./data", split="train", download=True, transform=data_transforms_train
    )
    val_dataset = Flowers102(
        root="./data", split="val", download=True, transform=data_transforms_train
    )
    test_dataset = Flowers102(
        root="./data", split="test", download=True, transform=data_transforms_test
    )
    
    # Create DataLoader instances for each dataset
    if mixup:
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    train_val_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    return train_loader, train_val_loader, validation_loader, test_loader

class FineTuneType(Enum):
    """An Enum to indicate which layers we want to train.
    """
    "Train just the newly added layers (feature extraction)."
    NEW_LAYERS = 1
    "Train just the classifier layer in case it's different from the newly added layers (feature-extraction)."
    CLASSIFIER = 2
    "Train all the layers (fine-tuning)."
    ALL = 3

class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

backbones = ["resnet18", "resnet50", "resnet152"]
class Flowers102Classifier(nn.Module):
    """Define a custom model that wraps a pre-trained model for classification
    on the Flowers-102 dataset. We shall fine-tune this model on the Flowers
    classification task.
    """
    def __init__(self, backbone):
        super().__init__()
        assert backbone in backbones
        self.backbone = backbone
        self.pretrained_model = None
        self.classifier_layers = []
        self.new_layers = []
        
        if backbone == "resnet18":
            self.pretrained_model = models.resnet18(weights="DEFAULT")
        elif backbone == "resnet50":
            self.pretrained_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        elif backbone == "resnet152":
            self.pretrained_model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
        self.classifier_layers = [self.pretrained_model.fc]
        # Replace the final layer with a classifier for 102 classes for the Flowers 102 dataset.
        self.pretrained_model.fc = nn.Linear(in_features=self.pretrained_model.fc.in_features, out_features=102, bias=True)
        self.new_layers = [self.pretrained_model.fc]
            


        # Dummy Param to be able to check the device on which this model is.
        # From https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently
        self.dummy_param = nn.Parameter(torch.empty(0))
        
        # Initialize metrics buffer. element 0 is loss, and element 1 is accuracy.
        self.register_buffer("train_metrics", torch.tensor([float("inf"), 0.0]))
        self.register_buffer("val_metrics", torch.tensor([float("inf"), 0.0]))
        

    def forward(self, x):
        return self.pretrained_model(x)
    
    def update_metrics(self, run_type, loss, accuracy):
        metrics = self.train_metrics if run_type == "train" else self.val_metrics
        if loss is not None:
            metrics[0] = loss
        if accuracy is not None:
            metrics[1] = accuracy

    def get_metrics(self, run_type):
        metrics = self.train_metrics if run_type == "train" else self.val_metrics
        return dict(zip(["loss", "accuracy"], metrics.tolist()))

    def fine_tune(self, what: FineTuneType):
        # The requires_grad parameter controls whether this parameter is
        # trainable during model training.
        m = self.pretrained_model
        for p in m.parameters():
            p.requires_grad = False

        if what is FineTuneType.NEW_LAYERS:
            for l in self.new_layers:
                for p in l.parameters():
                    p.requires_grad = True

        elif what is FineTuneType.CLASSIFIER:
            for l in self.classifier_layers:
                for p in l.parameters():
                    p.requires_grad = True

        else:
            for p in m.parameters():
                p.requires_grad = True

    def train_one_epoch(self, loader, optimizer, epoch):
        """Train this model for a single epoch. Return the loss computed
        during this epoch.
        """
        device = self.dummy_param.device
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        num_batches = 0

        for (inputs, targets) in iter(loader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = self(inputs)
            loss = criterion(outputs, targets)

            running_loss, num_batches = running_loss + loss.item(), num_batches + 1

            loss.backward()
            optimizer.step()
        # end for

        print(f"[{epoch}] Train Loss: {running_loss / num_batches:0.5f}")
        return running_loss / num_batches

    def evaluate(self, loader, metric, epoch, run_type):
        """Evaluate the model on the specified dataset (provided using the DataLoader
        instance). Return the loss and accuracy.
        """
        device = self.dummy_param.device
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        running_accuracy = 0.0
        num_batches = 0
        with torch.inference_mode():
            for (inputs, targets) in iter(loader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = self(inputs)
                loss = criterion(outputs, targets)

                running_loss = running_loss + loss.item()
                num_batches = num_batches + 1
                running_accuracy += metric(outputs, targets).item()
            # end with
        # end for

        print(f"[{epoch}] {run_type} Loss: {running_loss / num_batches:.5f}, Accuracy: {running_accuracy / num_batches:.5f}")
        return running_loss / num_batches, running_accuracy / num_batches
    # end def
    
    def train_multiple_epochs_and_save_best_checkpoint(
        self,
        train_loader,
        train_val_loader,
        val_loader,
        accuracy,
        optimizer,
        scheduler,
        epochs,
        filename,
        training_run,
    ):
        """Train this model for multiple epochs. The caller is expected to have frozen
        the layers that should not be trained. We run training for "epochs" epochs.
        The model with the best val accuracy is saved after every epoch.
        
        After every epoch, we also save the train/val loss and accuracy.
        """
        es = EarlyStopper()
        best_val_accuracy = self.get_metrics("val")['accuracy']
        for epoch in range(1, epochs + 1):
            self.train()
            self.train_one_epoch(train_loader, optimizer, epoch)

            # Evaluate accuracy on the train dataset.
            self.eval()
            train_loss, train_acc = self.evaluate(train_val_loader, accuracy, epoch, "Train")
            training_run.train_loss.append(train_loss)
            # end with

            # Evaluate accuracy on the val dataset.
            self.eval()
            val_loss, val_acc = self.evaluate(val_loader, accuracy, epoch, "Val")
            training_run.val_loss.append(val_loss)
            training_run.val_accuracy.append(val_acc)
            if val_acc > best_val_accuracy:
                # Save this checkpoint.
                print(f"Current valdation accuracy {val_acc*100.0:.2f} is better than previous best of {best_val_accuracy*100.0:.2f}. Saving checkpoint.")
                self.update_metrics("train", train_loss, train_acc)
                self.update_metrics("val", val_loss, val_acc)
                torch.save(self.state_dict(), filename)
                best_val_accuracy = val_acc
            
            scheduler.step()
            if es.early_stop(val_loss):
              print(f"Early stopping at ${epoch}")
              break
        # end for (epoch)
    # end def

    def get_optimizer_params(self):
        """This method is used only during model fine-tuning when we need to
        set a linear or expotentially decaying learning rate (LR) for the
        layers in the model. We exponentially decay the learning rate as we
        move away from the last output layer.
        """
        options = []

        if self.backbone in ['resnet18', 'resnet50', 'resnet152']:
            # For the resnet class of models, we decay the LR exponentially and reduce
            # it to a third of the previos value at each step.
            layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']
            lr = 0.0001
            for layer_name in reversed(layers):
                options.append({
                    "params": getattr(self.pretrained_model, layer_name).parameters(),
                    'lr': lr,
                })
                lr = lr / 3.0
            # end for
        # end if
        return options
    
# Helper classes and methods to plot the key metrics during a training run.

class TrainingRun:
    """A TrainingRun class stores information about a single training run.
    """
    def __init__(self):
        self.train_loss = []
        self.train_accuracy = []
        self.val_loss = []
        self.val_accuracy = []
        self.test_accuracy = []
    # end def
    
    def plot(self, plt, ax):
        """Plot this training run using matplotlib.
        """
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        epochs = list(range(1, len(self.train_loss) + 1))

        ax.plot(epochs, self.train_loss, label="Train Loss", c="blue")
        ax.plot(epochs, self.val_loss, label="Val Loss", c="orange")
        plt.legend()

        ax2 = ax.twinx()
        ax2.set_ylabel("Accuracy %")

        ax2.plot(epochs, self.val_accuracy, label="Val Accuracy", c="green")
        if len(self.train_accuracy) == len(epochs):
            ax2.plot(epochs, self.train_accuracy, label="Train Accuracy", c="cyan")

        plt.legend()
    # end def
# end class

def plot_training_runs(training_runs):
    for backbone, tr in training_runs.items():
        fig = plt.figure(figsize=(5, 4))
        ax = plt.subplot()
        tr.plot(plt, ax)
        plt.title(f"Using pre-trained {backbone}")
        plt.show()
    # end for
# end def