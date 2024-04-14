
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as v2
import numpy as np

from enum import Enum
from torchvision import models
from torchvision.datasets import Flowers102
from torch.utils.data import DataLoader

    

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


def get_train_val_test_loader(train_batch_size: int = 32, other_batch_size: int = 32, mixup: bool = False):
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
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    train_val_loader = DataLoader(train_dataset, batch_size=other_batch_size, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=other_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=other_batch_size)
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
            training_run.train_accuracy.append(train_acc)
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
              print(f"Early stopping at epoch {epoch}")
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


## Define Triplet Loss
def pairwise_distance_torch(embeddings, device):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      embeddings: 2-D Tensor of size [number of data, feature dimension].
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """

    # pairwise distance matrix with precise embeddings
    precise_embeddings = embeddings.to(dtype=torch.float32)

    c1 = torch.pow(precise_embeddings, 2).sum(axis=-1)
    c2 = torch.pow(precise_embeddings.transpose(0, 1), 2).sum(axis=0)
    c3 = precise_embeddings @ precise_embeddings.transpose(0, 1)

    c1 = c1.reshape((c1.shape[0], 1))
    c2 = c2.reshape((1, c2.shape[0]))
    c12 = c1 + c2
    pairwise_distances_squared = c12 - 2.0 * c3

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.max(pairwise_distances_squared, torch.tensor([0.]).to(device))
    # Get the mask where the zero distances are at.
    error_mask = pairwise_distances_squared.clone()
    error_mask[error_mask > 0.0] = 1.
    error_mask[error_mask <= 0.0] = 0.

    pairwise_distances = torch.mul(pairwise_distances_squared, error_mask)

    # Explicitly set diagonals to zero.
    mask_offdiagonals = torch.ones((pairwise_distances.shape[0], pairwise_distances.shape[1])) - torch.diag(torch.ones(pairwise_distances.shape[0]))
    pairwise_distances = torch.mul(pairwise_distances.to(device), mask_offdiagonals.to(device))
    return pairwise_distances

def TripletSemiHardLoss(y_true, y_pred, device, margin=1.0):
    """Computes the triplet loss_functions with semi-hard negative mining.
       The loss_functions encourages the positive distances (between a pair of embeddings
       with the same labels) to be smaller than the minimum negative distance
       among which are at least greater than the positive distance plus the
       margin constant (called semi-hard negative) in the mini-batch.
       If no such negative exists, uses the largest negative distance instead.
       See: https://arxiv.org/abs/1503.03832.
       We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
       [batch_size] of multi-class integer labels. And embeddings `y_pred` must be
       2-D float `Tensor` of l2 normalized embedding vectors.
       Args:
         margin: Float, margin term in the loss_functions definition. Default value is 1.0.
         name: Optional name for the op.
       """

    labels, embeddings = y_true, y_pred

    # Reshape label tensor to [batch_size, 1].
    lshape = labels.shape
    labels = torch.reshape(labels, [lshape[0], 1])

    pdist_matrix = pairwise_distance_torch(embeddings, device)

    # Build pairwise binary adjacency matrix.
    adjacency = torch.eq(labels, labels.transpose(0, 1))
    # Invert so we can select negatives only.
    adjacency_not = adjacency.logical_not()

    batch_size = labels.shape[0]

    # Compute the mask.
    pdist_matrix_tile = pdist_matrix.repeat(batch_size, 1)
    adjacency_not_tile = adjacency_not.repeat(batch_size, 1)

    transpose_reshape = pdist_matrix.transpose(0, 1).reshape(-1, 1)
    greater = pdist_matrix_tile > transpose_reshape

    mask = adjacency_not_tile & greater

    # final mask
    mask_step = mask.to(dtype=torch.float32)
    mask_step = mask_step.sum(axis=1)
    mask_step = mask_step > 0.0
    mask_final = mask_step.reshape(batch_size, batch_size)
    mask_final = mask_final.transpose(0, 1)

    adjacency_not = adjacency_not.to(dtype=torch.float32)
    mask = mask.to(dtype=torch.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    axis_maximums = torch.max(pdist_matrix_tile, dim=1, keepdim=True)
    masked_minimums = torch.min(torch.mul(pdist_matrix_tile - axis_maximums[0], mask), dim=1, keepdim=True)[0] + axis_maximums[0]
    negatives_outside = masked_minimums.reshape([batch_size, batch_size])
    negatives_outside = negatives_outside.transpose(0, 1)

    # negatives_inside: largest D_an.
    axis_minimums = torch.min(pdist_matrix, dim=1, keepdim=True)
    masked_maximums = torch.max(torch.mul(pdist_matrix - axis_minimums[0], adjacency_not), dim=1, keepdim=True)[0] + axis_minimums[0]
    negatives_inside = masked_maximums.repeat(1, batch_size)

    semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = margin + pdist_matrix - semi_hard_negatives

    mask_positives = adjacency.to(dtype=torch.float32) - torch.diag(torch.ones(batch_size)).to(device)
    num_positives = mask_positives.sum()

    triplet_loss = (torch.max(torch.mul(loss_mat, mask_positives), torch.tensor([0.]).to(device))).sum() / num_positives
    triplet_loss = triplet_loss.to(dtype=embeddings.dtype)
    return triplet_loss


class TripletLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, input, target, **kwargs):
        return TripletSemiHardLoss(target, input, self.device)