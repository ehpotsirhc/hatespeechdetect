#!/usr/bin/env python3

# XClass Classifier - Data Preprocessing Module
# Rewritten by Christophe Leung based on https://github.com/ZihanWangKi/XClass
# November 13, 2023

# Tested on Python 3.10.7; should be compatible between v3.6 through v3.11

# -----------------------------------------------------------------------------
from config import Constants
from utils import StaticRepUtils as SRU
from collections import Counter
import logging, string, tqdm, torch, os, pickle
import numpy as np

# =================================================================================================

# -----------------------------------------------------------------------------
# handles the main program's bootstrapping
class StaticReps:
    @staticmethod
    def prepare_sentence(tokenizer, text):
        # setting for BERT
        model_max_tokens = 512
        has_sos_eos = True
        ######################
        max_tokens = model_max_tokens
        if has_sos_eos:
            max_tokens -= 2
        sliding_window_size = max_tokens // 2

        if not hasattr(StaticReps.prepare_sentence, 'sos_id'):
            StaticReps.prepare_sentence.sos_id, StaticReps.prepare_sentence.eos_id = tokenizer.encode('', add_special_tokens=True)

        tokenized_text = tokenizer.basic_tokenizer.tokenize(text, never_split=tokenizer.all_special_tokens)
        tokenized_to_id_indicies = []

        tokenids_chunks = []
        tokenids_chunk = []

        for index, token in enumerate(tokenized_text + [None]):
            if token is not None:
                tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
            if token is None or len(tokenids_chunk) + len(tokens) > max_tokens:
                tokenids_chunks.append([StaticReps.prepare_sentence.sos_id] + tokenids_chunk + [StaticReps.prepare_sentence.eos_id])
                if sliding_window_size > 0:
                    tokenids_chunk = tokenids_chunk[-sliding_window_size:]
                else:
                    tokenids_chunk = []
            if token is not None:
                tokenized_to_id_indicies.append((len(tokenids_chunks), len(tokenids_chunk), len(tokenids_chunk) + len(tokens)))
                tokenids_chunk.extend(tokenizer.convert_tokens_to_ids(tokens))

        return tokenized_text, tokenized_to_id_indicies, tokenids_chunks


    @staticmethod
    def sentence_encode(tokens_id, model, layer):
        input_ids = torch.tensor([tokens_id], device=model.device)

        with torch.no_grad():
            hidden_states = model(input_ids)
        all_layer_outputs = hidden_states[2]

        layer_embedding = SRU.tensor_to_numpy(all_layer_outputs[layer].squeeze(0))[1: -1]
        return layer_embedding


    @staticmethod
    def sentence_to_wordtoken_embeddings(layer_embeddings, tokenized_text, tokenized_to_id_indicies):
        word_embeddings = []
        for text, (chunk_index, start_index, end_index) in zip(tokenized_text, tokenized_to_id_indicies):
            word_embeddings.append(np.average(layer_embeddings[chunk_index][start_index: end_index], axis=0))
        assert len(word_embeddings) == len(tokenized_text)
        return np.array(word_embeddings)


    @staticmethod
    def handle_sentence(model, layer, tokenized_text, tokenized_to_id_indicies, tokenids_chunks):
        layer_embeddings = [StaticReps.sentence_encode(tokenids_chunk, model, layer) for tokenids_chunk in tokenids_chunks]
        word_embeddings = StaticReps.sentence_to_wordtoken_embeddings(
            layer_embeddings, tokenized_text, tokenized_to_id_indicies)
        return word_embeddings


    @staticmethod
    def tokenize_and_count_tokens(texts, tokenizer):
        logging.info('Tokenizing texts')
        counts = Counter()
        for text in tqdm.tqdm(texts):
            tokenized_text, tokenized_to_id_indicies, tokenids_chunks = StaticReps.prepare_sentence(tokenizer, text)
            counts.update(word.translate(str.maketrans('','',string.punctuation)) for word in tokenized_text)
        del counts['']
        return (counts, tokenized_text, tokenized_to_id_indicies, tokenids_chunks)
    

    @staticmethod
    def tokenize_and_get_wordrep(texts, tokenizer, counts, args, model):
        logging.info('Retrieving word representations')
        updated_counts = {k: c for k, c in counts.items() if c >= args.vocab_min_occurrence}
        word_rep, word_count, tokenization_info = {}, {}, []

        for text in tqdm.tqdm(texts):
            tokenized_text, tokenized_to_id_indicies, tokenids_chunks = StaticReps.prepare_sentence(tokenizer, text)
            tokenization_info.append((tokenized_text, tokenized_to_id_indicies, tokenids_chunks))
            contextualized_word_representations = StaticReps.handle_sentence(
                model, args.layer, tokenized_text, tokenized_to_id_indicies, tokenids_chunks)
            for i in range(len(tokenized_text)):
                word = tokenized_text[i]
                if word in updated_counts.keys():
                    if word not in word_rep:
                        word_rep[word] = 0
                        word_count[word] = 0
                    word_rep[word] += contextualized_word_representations[i]
                    word_count[word] += 1
        return (word_rep, word_count, tokenization_info)


    @staticmethod
    def vocab_stats(word_rep, word_count):
        word_avg = {}
        for k,v in word_rep.items():
            word_avg[k] = word_rep[k]/word_count[k]
            
        vocab_words = list(word_avg.keys())
        static_word_representations = list(word_avg.values())
        vocab_occurrence = list(word_count.values()) 
        return (word_avg, vocab_words, static_word_representations, vocab_occurrence)


    @staticmethod
    def write_tokenized(args, tokenization_info):
        fname_tokenized = 'tokenization_lm-%s-%s.pickle' % (args.lm_type, args.layer)
        logging.info('Caching tokenized data to "%s"' % fname_tokenized)
        os.makedirs(Constants.DPATH_CACHED) if not Constants.DPATH_CACHED.exists() else None
        with open(Constants.DPATH_CACHED/fname_tokenized, 'wb') as f:
            pickle.dump({'tokenization_info': tokenization_info}, f, protocol=4)

    
    @staticmethod
    def write_staticreps(args, vocab_words, static_word_representations, vocab_occurrence):
        fname_staticreps = 'static_repr_lm-%s-%s.pickle' % (args.lm_type, args.layer)
        logging.info('Caching static word representations to "%s"' % fname_staticreps)
        os.makedirs(Constants.DPATH_CACHED) if not Constants.DPATH_CACHED.exists() else None
        with open(Constants.DPATH_CACHED/fname_staticreps, 'wb') as f:
            pickle.dump({
                "static_word_representations": static_word_representations,
                "vocab_words": vocab_words,
                "word_to_index": {v: k for k, v in enumerate(vocab_words)},
                "vocab_occurrence": vocab_occurrence,
            }, f, protocol=4)

    
    # -------------------------------------------------------------------------
    def main(self, args, texts):
        model_class, tokenizer_class, pretrained_weights = Constants.MODEL
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
        model.eval()
        model.cuda()

        counts, tokenized_text, tokenized_to_id_indicies, tokenids_chunks = self.tokenize_and_count_tokens(texts, tokenizer)
        word_rep, word_count, tokenization_info = self.tokenize_and_get_wordrep(texts, tokenizer, counts, args, model)
        word_avg, vocab_words, static_word_representations, vocab_occurrence = self.vocab_stats(word_rep, word_count)

        self.write_tokenized(args, tokenization_info)
        self.write_staticreps(args, vocab_words, static_word_representations, vocab_occurrence)

