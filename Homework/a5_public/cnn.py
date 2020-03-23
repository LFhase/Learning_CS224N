#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, reshape_dim, conv_out_dim):
        super().__init__()
        self.reshape_dim = reshape_dim
        self.conv_out_dim = conv_out_dim
        self.cnn = nn.Conv1d(self.reshape_dim, self.conv_out_dim, 5)

    def forward(self, x_reshape):
        x_conv_out = self.cnn(x_reshape)
        return x_conv_out


### END YOUR CODE
