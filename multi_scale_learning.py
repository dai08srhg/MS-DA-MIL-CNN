# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import os
import hydra
from pathlib import Path
from model.model import FeatureExtractor, ClassPredictor, DomainPredictor, DAMIL, MSDAMIL
from model.dataset import MsDaMilDataset
import os

PROJECT_DIR = Path().resolve()
CONFIG_DIR = PROJECT_DIR / 'conf'


def train(model: MSDAMIL, data_loader, loss_fn, optimizer, device, epochs):
    """MSDAMILの学習を行う"""
    # 訓練開始
    model.train()
    for epoch in range(epochs):
        for scale1_bag, scale2_bag, class_label in data_loader:
            scale1_bag = scale1_bag.squeeze(0).to(device)
            scale2_bag = scale2_bag.squeeze(0).to(device)
            class_label = class_label.squeeze(0).to(device)
            #勾配初期化
            optimizer.zero_grad() 
            class_y, _ = model(scale1_bag, scale2_bag)
            # loss計算
            loss = loss_fn(class_y, class_label)
            # 逆伝搬
            loss.backward()
            # パラメータ更新
            optimizer.step()


@hydra.main(config_path=f'{CONFIG_DIR}/conf.yaml')
def main(cfg):
    device = cfg.device
    torch.backends.cudnn.benchmark = True  # cudnnベンチマークモード
    # dataset作成
    data_root_dir = cfg.data_root_dir
    dataset = MsDaMilDataset(data_root_dir=data_root_dir)
    # data loader作成 (必ずbatch_sizeは1)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True, num_workers = 1)

    # モデル構築
    # DAMILを構築して学習済みのパラメータをロードする
    feature_extractor = FeatureExtractor()
    domain_num = dataset.domain_unique_num
    domain_predictor = DomainPredictor(domain_num=domain_num)
    class_predictor = ClassPredictor()
    DAMIL_model = DAMIL(feature_extractor, class_predictor, domain_predictor)
    # scale1のパラメータロード
    file_name = f'{cfg.params_file_prefix}_scale1.pth'
    file_path = os.path.join(cfg.tmp_storage, file_name)
    DAMIL_model.load_state_dict(torch.load(file_path))
    # feature_extoractorのみ利用
    feature_extoractor_scale1 = DAMIL.feature_extractor
    # scale2のパラメータロード
    file_name = f'{cfg.params_file_prefix}_scale2.pth'
    file_path = os.path.join(cfg.tmp_storage, file_name)
    DAMIL_model.load_state_dict(torch.load(file_path))
    # feature_extoractorのみ利用
    feature_extoractor_scale2 = DAMIL.feature_extractor
    # MSDAMIL構築
    MSDAMIL_model = MSDAMIL(feature_extoractor_scale1, feature_extoractor_scale2, class_predictor)
    MSDAMIL_model = MSDAMIL_model.to(device)

    # 損失関数定義
    loss_fn = nn.CrossEntropyLoss()
    # optimizer定義
    lerning_rate = cfg.hyper_parameters.sgd.lerning_rate
    momentum = cfg.hyper_parameters.sgd.momentum
    optimizer = optim.SGD(MSDAMIL_model.parameters(), lr=lerning_rate, momentum=momentum)
    # エポック数
    epochs = cfg.hyper_parameters.epochs

    # Start Training
    train(MSDAMIL_model, data_loader, loss_fn, optimizer, device, epochs)

    # Save params
    file_name = f'{cfg.params_file_prefix.MSDAMIL}.pth'
    file_path = os.path.join(cfg.tmp_storage, file_name)
    MSDAMIL_model = MSDAMIL_model.to('cpu')
    torch.save(MSDAMIL_model.state_dict(), file_path)


if __name__ == "__main__":
    main()