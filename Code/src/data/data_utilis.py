from transformers import BertTokenizer
import numpy as np


class DocREDTokenizer:

    def __init__(self, tokenizer_path):
        self.tokenizer_path = tokenizer_path
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)

    def tokenize(self, sents, entity_starts, entity_ends, output_word_lens=False):
        adjusted_sent = []
        adjusted_idx_list = []
        word_lens = []
        for idx_sent, sent in enumerate(sents):
            adjusted_token_idx_map = {}
            for idx_token, token in enumerate(sent):
                word_pieces = self.bert_tokenizer.tokenize(token)
                if (idx_sent, idx_token) in entity_starts:
                    word_pieces = ["*"] + word_pieces

                if (idx_sent, idx_token) in entity_ends:
                    word_pieces = word_pieces + ["*"]

                adjusted_token_idx_map[idx_token] = len(adjusted_sent)
                word_lens.append(len(word_pieces))
                adjusted_sent.extend(word_pieces)

            adjusted_token_idx_map[len(sent)] = len(adjusted_sent)
            word_lens.append(0)
            adjusted_idx_list.append(adjusted_token_idx_map)

        if output_word_lens:
            return adjusted_sent, adjusted_idx_list, word_lens

        return adjusted_sent, adjusted_idx_list

    def readjust(self, adjusted_sent, adjusted_idx_list, entities, word_lens, max_dist=50, max_len=510):
        sent_len = len(adjusted_sent)
        sent_masks = [0] * sent_len
        for entity in entities:
            for mention in entity:
                sent_idx = mention["sent_id"]
                sent_idx_map = adjusted_idx_list[sent_idx]

                pos1 = sent_idx_map[mention["pos"][0]]
                pos2 = sent_idx_map[mention["pos"][1]]
                entity_start = min(pos1, pos2)
                entity_end = max(pos1, pos2)

                low_bound = max(0, entity_start - max_dist)
                up_bound = min(sent_len, entity_end + max_dist)
                sent_masks[low_bound: up_bound] = [1] * (up_bound - low_bound)

        total_sent_len = sum(sent_masks)
        if total_sent_len < max_len:
            word_ids = [idx for idx in range(sent_len) if sent_masks[idx] == 0]
            np.random.shuffle(word_ids)
            word_ids = word_ids[:max_len - total_sent_len]
            for word_id in word_ids:
                sent_masks[word_id] = 1

        new_adjusted_sent = []
        new_adjusted_idx_list = []
        global_wordpiece_idx = 0
        global_word_idx = 0
        for idx_sent in range(len(adjusted_idx_list)):
            new_sent_idx_map = {}
            sent_idx_map = adjusted_idx_list[idx_sent]
            for idx_token in range(len(sent_idx_map)):
                new_sent_idx_map[idx_token] = len(new_adjusted_sent)
                for wordpiece_idx_in_word in range(word_lens[global_word_idx]):
                    if sent_masks[global_wordpiece_idx] == 1:
                        new_adjusted_sent.append(adjusted_sent[global_wordpiece_idx])

                    global_wordpiece_idx += 1

                global_word_idx += 1

            new_sent_idx_map[len(sent_idx_map)] = len(new_adjusted_sent)
            new_adjusted_idx_list.append(new_sent_idx_map)

        return new_adjusted_sent, new_adjusted_idx_list
