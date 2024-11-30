import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
import numpy as np
from utils import MyTokenizer
from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizer,
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
import logging

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # Ensure inputs and targets have the correct shape
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1)

        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)  # prevents nans when probability is 0
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class ReviewerModel(T5ForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        self.cls_head = nn.Linear(self.config.d_model, 2, bias=True)
        self.init()

    def init(self):
        nn.init.xavier_uniform_(self.lm_head.weight)
        factor = self.config.initializer_factor
        self.cls_head.weight.data.normal_(mean=0.0, \
            std=factor * ((self.config.d_model) ** -0.5))
        self.cls_head.bias.data.zero_()

    def forward(
        self, *argv, **kwargs
    ):
        if "cls" in kwargs:
            assert (
                "input_ids" in kwargs and \
                "labels" in kwargs and \
                "attention_mask" in kwargs
            )
            return self.cls(
                input_ids=kwargs["input_ids"],
                labels=kwargs["labels"],
                attention_mask=kwargs["attention_mask"],
            )
        if "input_labels" in kwargs:
            assert (
                "input_ids" in kwargs and \
                "input_labels" in kwargs and \
                "decoder_input_ids" in kwargs and \
                "attention_mask" in kwargs and \
                "decoder_attention_mask" in kwargs
            ), "Please give these arg keys."
            input_ids = kwargs["input_ids"]
            input_labels = kwargs["input_labels"]
            decoder_input_ids = kwargs["decoder_input_ids"]
            attention_mask = kwargs["attention_mask"]
            decoder_attention_mask = kwargs["decoder_attention_mask"]
            if "encoder_loss" not in kwargs:
                encoder_loss = True
            else:
                encoder_loss = kwargs["encoder_loss"]
            return self.review_forward(input_ids, input_labels, decoder_input_ids, attention_mask, decoder_attention_mask, encoder_loss)
        return super().forward(*argv, **kwargs)

    def cls(
        self,
        input_ids,
        labels,
        attention_mask,
    ):
        encoder_outputs = self.encoder( \
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            return_dict=False
        )
        hidden_states = encoder_outputs[0]
        first_hidden = hidden_states[:, 0, :]
        first_hidden = nn.Dropout(0.3)(first_hidden)
        logits = self.cls_head(first_hidden)
        loss_fct = FocalLoss()
        if labels is not None:
            loss = loss_fct(logits, labels)
            return loss
        return logits

    def review_forward(
        self,
        input_ids,
        input_labels,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        encoder_loss=True
    ):
        encoder_outputs = self.encoder( \
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            return_dict=False
        )
        hidden_states = encoder_outputs[0]
        decoder_inputs = self._shift_right(decoder_input_ids)
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_inputs,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=False,
            return_dict=False
        )
        sequence_output = decoder_outputs[0]
        if self.config.tie_word_embeddings: # this is True default
            sequence_output = sequence_output * (self.model_dim ** -0.5)
        if encoder_loss:
            cls_logits = nn.functional.linear(hidden_states, self.encoder.get_input_embeddings().weight)
        lm_logits = self.lm_head(sequence_output)
        if decoder_input_ids is not None:
            lm_loss_fct = FocalLoss(ignore_index=0)      # Warning: PAD_ID should be 0
            loss = lm_loss_fct(lm_logits.view(-1, lm_logits.size(-1)), decoder_input_ids.view(-1))
            if encoder_loss and input_labels is not None:
                cls_loss_fct = FocalLoss(ignore_index=-100)
                loss += cls_loss_fct(cls_logits.view(-1, cls_logits.size(-1)), input_labels.view(-1))
            return loss
        return cls_logits, lm_logits

def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e6))


def build_or_load_gen_model(args):
    config_class, model_class, tokenizer_class = T5Config, ReviewerModel, RobertaTokenizer
    
    config = config_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    tokenizer.special_dict = {
        f"<e{i}>" : tokenizer.get_vocab()[f"<e{i}>"] for i in range(99, -1, -1)
    }

    tokenizer.mask_id = tokenizer.get_vocab()["<mask>"]
    tokenizer.bos_id = tokenizer.get_vocab().get("<s>", tokenizer.pad_token_id)  # Handle missing <s> token
    tokenizer.pad_id = tokenizer.get_vocab()["<pad>"]
    tokenizer.eos_id = tokenizer.get_vocab()["</s>"]
    tokenizer.msg_id = tokenizer.get_vocab()["<msg>"]
    tokenizer.keep_id = tokenizer.get_vocab()["<keep>"]
    tokenizer.add_id = tokenizer.get_vocab()["<add>"]
    tokenizer.del_id = tokenizer.get_vocab()["<del>"]
    tokenizer.start_id = tokenizer.get_vocab()["<start>"]
    tokenizer.end_id = tokenizer.get_vocab()["<end>"]

    logger.info(
        "Finish loading model [%s] from %s",
        get_model_size(model),
        args.model_name_or_path,
    )

    if args.load_model_path is not None:
        model_path = os.path.join(args.load_model_path, "pytorch_model.bin")
        logger.info("Reload model from {}".format(model_path))
        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError:
            saved = model.cls_head
            model.cls_head = None
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.cls_head = saved
        model.to(args.local_rank)

    return config, model, tokenizer