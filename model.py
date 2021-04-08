import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

class MnistModel(pl.LightningModule):
    
    def __init__(self, data_dir='./', hidden_size=64, learning_rate=2e-4, freeze_last_layer=False):

        super().__init__()

        self.freeze_last_layer=freeze_last_layer
        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            
            #nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    #THIS IS WHERE IT'S GOING DOWN
    def configure_optimizers(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.normal_(m.bias.data, 0.0, 0.02)
            
       
        params = []
        self.init_weights = []
        self.init_biases = []
        
        n_layers = 4
        for i_layer in range(n_layers):
            self.init_weights += [list(self.model.modules())[-2*(n_layers-i_layer)+1].weight.clone()]
            self.init_biases += [list(self.model.modules())[-2*(n_layers-i_layer)+1].bias.clone()]
            
            if not self.freeze_last_layer or i_layer != n_layers - 1:
                params += [list(self.model.modules())[-2*(n_layers-i_layer)+1].weight, list(self.model.modules())[-2*(n_layers-i_layer)+1].bias]
                
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        
        return optimizer

    def reinit_layer(self, i_layer):
        n_layers = 4
        with torch.no_grad():
            list(self.model.modules())[-2*(n_layers - i_layer) + 1].weight.set_(self.init_weights[i_layer])
            list(self.model.modules())[-2*(n_layers - i_layer) + 1].bias.set_(self.init_biases[i_layer])
    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [50000, 10000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)