#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch.nn as nn


class Highway(nn.Module):

    def __init__(self, conv_out_dim, e_word):
        super().__init__()
        self.conv_out_dim = conv_out_dim
        self.e_word = e_word
        self.linear_proj = nn.Linear(conv_out_dim, self.e_word)
        self.linear_gate = nn.Linear(self.conv_out_dim, self.e_word)

    def forward(self, x_conv_out):
        x_proj = nn.functional.relu(self.linear_proj(x_conv_out))
        x_gate = self.linear_gate(x_conv_out)
        x_highway = x_proj * x_gate + (1.0 - x_gate) * x_conv_out
        return x_highway


### END YOUR CODE
