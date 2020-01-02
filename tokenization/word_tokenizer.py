# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WordpieceTokenizer classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import unicodedata

from .utils import (load_vocab,
                   convert_to_unicode,
                   clean_text,
                   split_on_whitespace,
                   convert_by_vocab,
                   tokenize_chinese_chars)

class WordTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, opt,vocab = None, unk_token="[UNK]"):
        self.vocab = load_vocab(vocab)
        self.token2id = vocab
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.unk_token = unk_token
        self.blank_padding = opt.blank_padding
        self.max_length = opt.max_length


    def tokenize(self, text):
        """    Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform tokenization
            using the given vocabulary.

            For example:
                input = "unaffable"
                output = ["un", "##aff", "##able"]

            Args:
                text: A single token or whitespace separated tokens. This should have already been passed through `BasicTokenizer`.
            Returns:
                output_tokens: A list of wordpiece tokens.
                current_positions: A list of the current positions for the original words in text .
        """
        text = convert_to_unicode(text)
        text = clean_text(text)
        text = tokenize_chinese_chars(text)
        # output_tokens = []
        token_list = split_on_whitespace(text)
        # for chars in token_list:
        #     # current_positions.append([])
        #     if chars in self.vocab:
        #         output_tokens.append(chars)
        #     else:
        #         output_tokens.append(self.unk_token)
        return token_list

    def convert_tokens_to_ids(self, tokens, max_seq_length = None, blank_id = 0, unk_id = 1, uncased = True):
        return convert_by_vocab(self.vocab, tokens, max_seq_length, blank_id, unk_id, uncased=uncased)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)

    def tokenizer(self, item):
        """

        Args:
            sentence: string, the input sentence
            pos_head: [start, end], position of the head entity
            pos_end: [start, end], position of the tail entity
            is_token: if is_token == True, sentence becomes an array of token
        Return:
            Name of the relation of the sentence
        """
        if 'text' in item:
            sentence = item['text']
            is_token = False
        else:
            sentence = item['token']
            is_token = True
        pos_head = item['h']['pos']  # a list []
        pos_tail = item['t']['pos']

        # Sentence -> token
        if not is_token:
            if pos_head[0] > pos_tail[0]:
                pos_min, pos_max = [pos_tail, pos_head]
                rev = True
            else:
                pos_min, pos_max = [pos_head, pos_tail]
                rev = False
            sent_0 = self.tokenize(sentence[:pos_min[0]])  # get a preprocessed list
            sent_1 = self.tokenize(sentence[pos_min[1]:pos_max[0]])
            sent_2 = self.tokenize(sentence[pos_max[1]:])
            ent_0 = self.tokenize(sentence[pos_min[0]:pos_min[1]])
            ent_1 = self.tokenize(sentence[pos_max[0]:pos_max[1]])
            tokens = sent_0 + ent_0 + sent_1 + ent_1 + sent_2
            if rev:
                pos_tail = [len(sent_0), len(sent_0) + len(ent_0)]
                pos_head = [len(sent_0) + len(ent_0) + len(sent_1), len(sent_0) + len(ent_0) + len(sent_1) + len(ent_1)]
            else:
                pos_head = [len(sent_0), len(sent_0) + len(ent_0)]
                pos_tail = [len(sent_0) + len(ent_0) + len(sent_1), len(sent_0) + len(ent_0) + len(sent_1) + len(ent_1)]
        else:
            tokens = sentence  # be preprocessed sentence

        # Token -> index
        if self.blank_padding:
            indexed_tokens = self.convert_tokens_to_ids(tokens, self.max_length, self.token2id['[PAD]'],
                                                                  self.token2id['[UNK]'])
        else:
            indexed_tokens = self.convert_tokens_to_ids(tokens, unk_id=self.token2id['[UNK]'])

        # Position -> index
        # the distance of every word to entity
        pos1 = []  # left entity
        pos2 = []  # right entity
        pos1_in_index = min(pos_head[0], self.max_length)
        pos2_in_index = min(pos_tail[0], self.max_length)
        # calculate the distance of every word to entity
        for i in range(len(tokens)):
            pos1.append(min(i - pos1_in_index + self.max_length, 2 * self.max_length - 1))
            pos2.append(min(i - pos2_in_index + self.max_length, 2 * self.max_length - 1))

        if self.blank_padding:
            while len(pos1) < self.max_length:
                pos1.append(0)
            while len(pos2) < self.max_length:
                pos2.append(0)
            indexed_tokens = indexed_tokens[:self.max_length]
            pos1 = pos1[:self.max_length]
            pos2 = pos2[:self.max_length]
        # convert list to tensor
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)
        pos1 = torch.tensor(pos1).long().unsqueeze(0)  # (1, L)
        pos2 = torch.tensor(pos2).long().unsqueeze(0)  # (1, L)

        # Mask
        mask = []
        pos_min = min(pos1_in_index, pos2_in_index)
        pos_max = max(pos1_in_index, pos2_in_index)
        for i in range(len(tokens)):
            if i <= pos_min:
                mask.append(1)
            elif i <= pos_max:
                mask.append(2)
            else:
                mask.append(3)
        # Padding
        if self.blank_padding:
            while len(mask) < self.max_length:
                mask.append(0)
            mask = mask[:self.max_length]

        mask = torch.tensor(mask).long().unsqueeze(0)  # (1, L)
        return indexed_tokens, pos1, pos2, mask

