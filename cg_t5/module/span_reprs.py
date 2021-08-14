#!/usr/bin/env python
# encoding: utf-8
'''
#-------------------------------------------------------------------#
#                   CONFIDENTIAL --- CUSTOM STUDIOS                 #     
#-------------------------------------------------------------------#
#                                                                   #
#                   @Project Name : t5p                 #
#                                                                   #
#                   @File Name    : span_reprs.py                      #
#                                                                   #
#                   @Programmer   : Jeffrey                         #
#                                                                   #  
#                   @Start Date   : 2021/3/30 11:37                 #
#                                                                   #
#                   @Last Update  : 2021/3/30 11:37                 #
#                                                                   #
#-------------------------------------------------------------------#
# Classes:                                                          #
#                                                                   #
#-------------------------------------------------------------------#
'''

"""Different batched non-parametric span representations."""
import torch
import torch.nn as nn
from module.span_utils import get_span_mask
from abc import ABC, abstractmethod


class SpanRepr(ABC, nn.Module):
    """Abstract class describing span representation."""

    def __init__(self, input_dim, use_proj=False, proj_dim=256):
        super(SpanRepr, self).__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.use_proj = use_proj
        if use_proj:
            self.proj = nn.Linear(input_dim, proj_dim)

    @abstractmethod
    def forward(self, encoded_input, start_ids, end_ids):
        raise NotImplementedError

    def get_input_dim(self):
        return self.input_dim

    @abstractmethod
    def get_output_dim(self):
        raise NotImplementedError


class AvgSpanRepr(SpanRepr, nn.Module):
    """Class implementing the avg span representation."""

    def forward(self, encoded_input, start_ids, end_ids):
        if self.use_proj:
            encoded_input = self.proj(encoded_input)
        span_lengths = (end_ids - start_ids + 1).unsqueeze(1)
        span_masks = get_span_mask(start_ids, end_ids, encoded_input.shape[1])
        span_lengths = span_lengths.cuda()
        span_masks = span_masks.cuda()
        span_repr = torch.sum(encoded_input * span_masks, dim=1) / span_lengths.float()
        return span_repr

    def get_output_dim(self):
        if self.use_proj:
            return self.proj_dim
        else:
            return self.input_dim


class DiffSpanRepr(SpanRepr, nn.Module):
    """Class implementing the diff span representation - [h_j - h_{i-1}]"""

    def forward(self, encoded_input, start_ids, end_ids):
        if self.use_proj:
            encoded_input = self.proj(encoded_input)
        batch_size = encoded_input.shape[0]
        # first_comp = (encoded_input[torch.arange(batch_size), end_ids, :]
        #               - encoded_input[torch.arange(batch_size), start_ids - 1, :])
        # second_comp = (encoded_input[torch.arange(batch_size), start_ids, :]
        #                - encoded_input[torch.arange(batch_size), end_ids + 1, :])
        # span_repr = torch.cat([first_comp, second_comp], dim=1)
        span_repr = (encoded_input[torch.arange(batch_size), end_ids, :]
                     - encoded_input[torch.arange(batch_size), start_ids - 1, :])
        return span_repr

    def get_output_dim(self):
        if self.use_proj:
            return self.proj_dim
        else:
            return self.input_dim


class EndPointRepr(SpanRepr, nn.Module):
    """Class implementing the diff span representation - [h_j; h_i]"""

    def forward(self, encoded_input, start_ids, end_ids):
        if self.use_proj:
            encoded_input = self.proj(encoded_input)
        batch_size = encoded_input.shape[0]
        span_repr = torch.cat([encoded_input[torch.arange(batch_size), start_ids, :],
                               encoded_input[torch.arange(batch_size), end_ids, :]], dim=1)
        return span_repr

    def get_output_dim(self):
        if self.use_proj:
            return 2 * self.proj_dim
        else:
            return 2 * self.input_dim


class DiffSumSpanRepr(SpanRepr, nn.Module):
    """Class implementing the diff_sum span representation - [h_j - h_i; h_j + h_i]"""

    def forward(self, encoded_input, start_ids, end_ids):
        if self.use_proj:
            encoded_input = self.proj(encoded_input)
        batch_size = encoded_input.shape[0]
        span_repr = torch.cat([
            encoded_input[torch.arange(batch_size), end_ids, :]
            - encoded_input[torch.arange(batch_size), start_ids, :],
            encoded_input[torch.arange(batch_size), end_ids, :]
            + encoded_input[torch.arange(batch_size), start_ids, :]
        ], dim=1)
        return span_repr

    def get_output_dim(self):
        if self.use_proj:
            return 2 * self.proj_dim
        else:
            return 2 * self.input_dim


class MaxSpanRepr(SpanRepr, nn.Module):
    """Class implementing the max-pool span representation."""

    def forward(self, encoded_input, start_ids, end_ids):
        if self.use_proj:
            encoded_input = self.proj(encoded_input)
        span_masks = get_span_mask(start_ids, end_ids, encoded_input.shape[1])
        # put -inf to irrelevant positions
        tmp_repr = encoded_input * span_masks - 1e10 * (1 - span_masks)
        span_repr = torch.max(tmp_repr, dim=1)[0]
        return span_repr

    def get_output_dim(self):
        if self.use_proj:
            return self.proj_dim
        else:
            return self.input_dim


class CoherentSpanRepr(SpanRepr, nn.Module):
    """Class implementing the coherent span representation."""

    def forward(self, encoded_input, start_ids, end_ids):
        if self.use_proj:
            encoded_input = self.proj(encoded_input)
        batch_size = encoded_input.shape[0]
        p_size = int(encoded_input.shape[2] / 4)
        h_start = encoded_input[torch.arange(batch_size), start_ids, :]
        h_end = encoded_input[torch.arange(batch_size), end_ids, :]

        coherence_term = torch.sum(
            h_start[:, 2 * p_size:3 * p_size] * h_end[:, 3 * p_size:], dim=1, keepdim=True)
        return torch.cat(
            [h_start[:, :p_size], h_end[:, p_size:2 * p_size], coherence_term], dim=1)

    def get_output_dim(self):
        if self.use_proj:
            return (self.proj_dim // 2 + 1)
        else:
            return (self.input_dim // 2 + 1)


class CoherentOrigSpanRepr(SpanRepr, nn.Module):
    """Class implementing the coherent span representation."""

    def forward(self, encoded_input, start_ids, end_ids):
        if self.use_proj:
            encoded_input = self.proj(encoded_input)
        batch_size = encoded_input.shape[0]
        d_b = (int(encoded_input.shape[2]) * 480) // 1024
        d_c = (int(encoded_input.shape[2]) * 32) // 1024
        h_start = encoded_input[torch.arange(batch_size), start_ids, :]
        h_end = encoded_input[torch.arange(batch_size), end_ids, :]

        coherence_term = torch.sum(
            h_start[:, 2 * d_b:2 * d_b + d_c] * h_end[:, 2 * d_b + d_c:], dim=1, keepdim=True)
        return torch.cat(
            [h_start[:, :d_b], h_end[:, d_b:2 * d_b], coherence_term], dim=1)

    def get_output_dim(self):
        if self.use_proj:
            return ((self.proj_dim * 960) // 1024 + 1)
        else:
            return ((self.input_dim * 960) // 1024 + 1)


class AttnSpanRepr(SpanRepr, nn.Module):
    """Class implementing the attention-based span representation."""

    def __init__(self, input_dim, use_proj=False, proj_dim=256, use_endpoints=False):
        """If use_endpoints is true then concatenate the end points to attention-pooled span repr.
        Otherwise just return the attention pooled term.
        """
        super(AttnSpanRepr, self).__init__(input_dim, use_proj=use_proj, proj_dim=proj_dim)
        self.use_endpoints = use_endpoints
        if use_proj:
            input_dim = proj_dim
        self.attention_params = nn.Linear(input_dim, 1)
        # Initialize weight to zero weight
        # self.attention_params.weight.data.fill_(0)
        # self.attention_params.bias.data.fill_(0)

    def forward(self, encoded_input, start_ids, end_ids):
        if self.use_proj:
            encoded_input = self.proj(encoded_input)

        span_mask = get_span_mask(start_ids, end_ids, encoded_input.shape[1])
        attn_mask = (1 - span_mask) * (-1e10)
        attn_logits = self.attention_params(encoded_input) + attn_mask
        attention_wts = nn.functional.softmax(attn_logits, dim=1)
        attention_term = torch.sum(attention_wts * encoded_input, dim=1)
        if self.use_endpoints:
            batch_size = encoded_input.shape[0]
            h_start = encoded_input[torch.arange(batch_size), start_ids, :]
            h_end = encoded_input[torch.arange(batch_size), end_ids, :]
            return torch.cat([h_start, h_end, attention_term], dim=1)
        else:
            return attention_term

    def get_output_dim(self):
        if not self.use_endpoints:
            if self.use_proj:
                return self.proj_dim
            else:
                return self.input_dim
        else:
            if self.use_proj:
                return 3 * self.proj_dim
            else:
                return 3 * self.input_dim


def get_span_module(input_dim, method="avg", use_proj=False, proj_dim=256):
    """Initializes the appropriate span representation class and returns the object.
    """
    if method == "avg":
        return AvgSpanRepr(input_dim, use_proj=use_proj, proj_dim=proj_dim)
    elif method == "max":
        return MaxSpanRepr(input_dim, use_proj=use_proj, proj_dim=proj_dim)
    elif method == "diff":
        return DiffSpanRepr(input_dim, use_proj=use_proj, proj_dim=proj_dim)
    elif method == "diff_sum":
        return DiffSumSpanRepr(input_dim, use_proj=use_proj, proj_dim=proj_dim)
    elif method == "endpoint":
        return EndPointRepr(input_dim, use_proj=use_proj, proj_dim=proj_dim)
    elif method == "coherent":
        return CoherentSpanRepr(input_dim, use_proj=use_proj, proj_dim=proj_dim)
    elif method == "coherent_original":
        return CoherentOrigSpanRepr(input_dim, use_proj=use_proj, proj_dim=proj_dim)
    elif method == "attn":
        return AttnSpanRepr(input_dim, use_proj=use_proj, proj_dim=proj_dim)
    elif method == "coref":
        return AttnSpanRepr(input_dim, use_proj=use_proj, proj_dim=proj_dim, use_endpoints=True)
    else:
        raise NotImplementedError
