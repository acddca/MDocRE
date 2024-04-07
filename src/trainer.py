# -*- coding:utf-8 -*-
# @Author: wangjl
# @Time: 2022/12/13 17:51
# @File: trainer.py
# @Email: wangjl.nju.2020@gmail.com.
import torch
from abc import abstractmethod


class Trainer:

    @abstractmethod
    def train_step(self, batch, epoch=-1):
        raise NotImplementedError

    @abstractmethod
    def predict_step(self, batch, epoch=-1):
        raise NotImplementedError

    def load(self, path):
        print(f"### Load model from {path}.")
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict["model"])
        self.opt = state_dict["config"]

    def save(self, path):
        params = {
            "model": self.model.state_dict(),
            "config": self.opt
        }
        torch.save(params, path)
        print(f"### Save model to {path}.")