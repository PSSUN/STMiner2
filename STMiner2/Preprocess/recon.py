import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, TensorDataset

dtype = torch.cuda.FloatTensor


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features * 3, out_features)
        self.omega_0 = torch.tensor([[1], [2], [3]], dtype=torch.float32)
        self.omega_0 = self.omega_0.repeat(1, in_features).reshape(1, in_features * 3)

    def forward(self, input_x):
        input_x = input_x.repeat(1, 3)
        input_x = input_x * self.omega_0.to(input_x.device)
        return self.linear(torch.sin(input_x))


class INR(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

        mid = 1024
        self.INR = nn.Sequential(
            nn.Linear(2, mid),
            SineLayer(mid, mid),
            SineLayer(mid, mid),
            SineLayer(mid, out_dim),
        )

    def forward(self, coord):
        return self.INR(coord)


class INRModel(nn.Module):
    def __init__(
            self,
            X,
            spatial_coord,
            device,
            learning_rate=1e-4,
            reg_par=1e-4,
            epoch_num=100,
            batch_size=128,
            print_train_log_info=True,
    ):
        super().__init__()
        self.X = X
        self.coords = spatial_coord
        self.learning_rate_ = learning_rate
        self.device = device
        self.reg_par_ = reg_par
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.print_train_log_info = print_train_log_info

        out_dim = self.X.shape[1]
        self.INR = INR(out_dim=out_dim).to(device)

        self.optimizer_ = torch.optim.Adamax(
            self.INR.parameters(), lr=self.learning_rate_, weight_decay=self.reg_par_
        )

        dataset = TensorDataset(self.coords, self.X)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def fit(self):
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer_, T_0=50, T_mult=2)
        best_loss = float("inf")
        best_INR_recon = None

        for epoch in range(self.epoch_num):
            epoch_loss = 0
            epoch_recon = []

            for batch_coords, batch_X in self.dataloader:
                batch_coords = batch_coords.to(self.device)
                batch_X = batch_X.to(self.device)

                INR_recon = self.INR(batch_coords)
                self.optimizer_.zero_grad()

                loss = torch.norm((INR_recon - batch_X), p=2)
                loss = torch.sum(loss)
                epoch_loss += loss.item()

                loss.backward()
                self.optimizer_.step()

                epoch_recon.append(INR_recon.detach().cpu())

            scheduler.step()

            epoch_recon = torch.cat(epoch_recon, dim=0)
            avg_epoch_loss = epoch_loss / len(self.dataloader)

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_INR_recon = epoch_recon

            if self.print_train_log_info:
                print(f"Epoch {epoch + 1}/{self.epoch_num}, Loss: {avg_epoch_loss:.4f}", end="\r")

        return best_INR_recon


def spatial_reconstruction(adata, epoch_num=2000, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coords = adata.obsm["spatial"].astype(np.float32)
    node_feats = adata.X.astype(np.float32)

    if isinstance(node_feats, csr_matrix):
        node_feats = node_feats.toarray()

    node_feats = torch.from_numpy(node_feats).float().to(device)
    coords = torch.from_numpy(coords).float().to(device)

    print(f"out_dim: {node_feats.shape[1]}")

    model = INRModel(
        X=node_feats,
        spatial_coord=coords,
        device=device,
        learning_rate=1e-4,
        reg_par=0,
        epoch_num=epoch_num,
        batch_size=batch_size,
    )

    reconstructed_X = model.fit()
    # adata.X = reconstructed_X.cpu().detach().numpy()
    adata.uns['INR'] = reconstructed_X
