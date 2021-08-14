#!/usr/bin/env python
# encoding: utf-8
'''
#-------------------------------------------------------------------#
#                   CONFIDENTIAL --- CUSTOM STUDIOS                 #     
#-------------------------------------------------------------------#
#                                                                   #
#                   @Project Name : t5p                 #
#                                                                   #
#                   @File Name    : model.py                      #
#                                                                   #
#                   @Programmer   : Jeffrey                         #
#                                                                   #  
#                   @Start Date   : 2021/3/27 20:09                 #
#                                                                   #
#                   @Last Update  : 2021/3/27 20:09                 #
#                                                                   #
#-------------------------------------------------------------------#
# Classes:                                                          #
#                                                                   #
#-------------------------------------------------------------------#
'''
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration
import numpy as np
from module.span_reprs import get_span_module

class T5PForSequenceClassificationSpan(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.t5p = T5ForConditionalGeneration.from_pretrained(args.pretrained_model_path)
        self.span_layer = get_span_module(args.class_hidden_size, method=args.span_layer, use_proj=args.use_proj,
                                          proj_dim=args.class_proj_dim)
        self.classification_head = nn.Linear(self.span_layer.get_output_dim(), args.class_size)
        self.class_size = args.class_size
        self.generate_weight = args.generate_weight
        self.class_weight = args.class_weight

    def get_start_and_end(self, input_ids):
        start_ids = []
        end_ids = []
        for i in range(len(input_ids)):
            start_ids.append(0)
            a_t2n = input_ids[i].cpu().numpy()
            index = np.argwhere(a_t2n == 102)
            end_ids.append(index[1][0])
        start_ids = torch.tensor(start_ids, dtype=torch.long)
        end_ids = torch.tensor(end_ids, dtype=torch.long)
        return start_ids, end_ids

    def forward(self, input_ids=None, labels=None, class_types=None):
        outputs = self.t5p(input_ids=input_ids, labels=labels)
        decoder_loss = outputs.loss
        encoder = self.t5p.get_encoder()
        encoder_outputs = encoder(
            input_ids=input_ids,
            attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=False,
        )
        encoder_hidden_states = encoder_outputs[0]
        start_ids, end_ids = self.get_start_and_end(input_ids)
        span_repre = self.span_layer(encoder_hidden_states, start_ids, end_ids)
        logits = self.classification_head(span_repre)
        if class_types is None:
            class_loss = 0
        else:
            loss_fct = CrossEntropyLoss()
            logits_view = logits.view(-1, self.class_size)
            labels_view = class_types.view(-1)
            class_loss = loss_fct(logits_view, labels_view)
        if self.generate_weight == 0:
            loss = self.class_weight * class_loss
            return (loss, logits)
        if self.class_weight == 0:
            loss = self.generate_weight * decoder_loss
            return (loss, logits)
        loss = self.generate_weight * decoder_loss + self.class_weight * class_loss
        return (loss, logits)

    def classify(self, input_tensors,return_prob):
        encoder = self.t5p.get_encoder()
        encoder_outputs = encoder(
            input_ids=input_tensors,
            attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=False,
        )
        encoder_hidden_states = encoder_outputs[0]
        start_ids, end_ids = self.get_start_and_end(input_tensors)
        span_repre = self.span_layer(encoder_hidden_states, start_ids, end_ids)
        logits = self.classification_head(span_repre)
        if return_prob==True:
            return logits
        logits = logits.argmax(dim=-1)
        return logits

    def generate(self, input_tensors,
                 decoder_start_token_id=None,
                 eos_token_id=None,
                 max_length=100,return_prob=False):
        text_output = self.t5p.generate(input_tensors,
                                        decoder_start_token_id=decoder_start_token_id,
                                        eos_token_id=eos_token_id,
                                        max_length=max_length).cpu().numpy()
        class_output = self.classify(input_tensors,return_prob)
        return text_output, class_output
