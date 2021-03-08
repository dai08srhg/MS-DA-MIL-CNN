# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import hydra
from pathlib import Path
from model.model import FeatureExtractor, ClassPredictor, DomainPredictor, DAMIL
from model.dataset import DaMilDataset
import os

PROJECT_DIR = Path().resolve()
CONFIG_DIR = PROJECT_DIR / 'conf'


def train(model: DAMIL, data_loader, loss_fn, optimizer, device, da_rate, epochs):
    """Train DAMIL network"""
    model.train()
    for epoch in range(epochs):
        # Calculate domain adversarial regularization in this epoch
        p = ((epoch+1) / epochs) * da_rate
        epoch_da_rate = (2 / (1 + np.exp(-10*p))) - 1
        for bag, class_label, domain_labels in data_loader:
            bag = bag.squeeze(0).to(device)
            class_label = class_label.squeeze(0).to(device)
            domain_labels = domain_labels.squeeze(0).to(device)
            # Initialize gradient
            optimizer.zero_grad() 
            class_y, domain_ys, _ = model(bag, epoch_da_rate)
            # Calculate each losses
            class_loss = loss_fn(class_y, class_label)
            domain_loss = loss_fn(domain_ys, domain_labels)
            total_loss = class_loss + domain_loss
            # Backpropagation
            total_loss.backward()
            # Update parameters
            optimizer.step()


@hydra.main(config_path=f'{CONFIG_DIR}/conf.yaml')
def main(cfg):
    device = cfg.device
    torch.backends.cudnn.benchmark = True  # cudnn benchmark
    # Make dataset
    data_root_dir = cfg.data_root_dir
    scale = cfg.scale
    dataset = DaMilDataset(data_root_dir=data_root_dir, scale=scale)
    # Make dataloader (set batch_size to 1)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True, num_workers = 1)

    # Build models
    feature_extoractor = FeatureExtractor()
    class_predictor = ClassPredictor()
    domain_num = dataset.domain_unique_num
    domain_predictor = DomainPredictor(domain_num=domain_num)
    model = DAMIL(feature_extoractor, class_predictor, domain_predictor)
    model = model.to(device)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()
    # Define optimizer
    lerning_rate = cfg.hyper_parameters.sgd.lerning_rate
    momentum = cfg.hyper_parameters.sgd.momentum
    optimizer = optim.SGD(model.parameters(), lr=lerning_rate, momentum=momentum)
    # epoch num
    epochs = cfg.hyper_parameters.epochs
    # Domain adversarial regularization rate
    da_rate = cfg.hyper_parameters.da_rate

    # Start Training
    train(model, data_loader, loss_fn, optimizer, device, da_rate, epochs)

    # Save parameters
    file_name = f'{cfg.params_file_prefix.DAMIL}_{scale}.pth'
    file_path = os.path.join(cfg.tmp_storage, file_name)
    model = model.to('cpu')
    torch.save(model.state_dict(), file_path)


if __name__ == "__main__":
    main()
