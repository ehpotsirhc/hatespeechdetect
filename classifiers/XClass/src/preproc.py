#!/usr/bin/env python3

# XClass Classifier - Data Preprocessing Module
# Rewritten by Christophe Leung based on https://github.com/ZihanWangKi/XClass
# November 13, 2023

# Tested on Python 3.10.7; should be compatible between v3.6 through v3.11

# -----------------------------------------------------------------------------
from config import Constants
from utils import StaticRepUtils as SRU
from utils import ClassDocRepUtils as CDRU
from collections import Counter
from scipy.special import softmax
import logging, string, tqdm, torch, os, pickle
import numpy as np

# =================================================================================================

# -----------------------------------------------------------------------------
# Static Representation Computations; distill a set of vocabulary from the raw texts
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
        logging.info('Tokenizing texts...')
        counts = Counter()
        for text in tqdm.tqdm(texts):
            tokenized_text, tokenized_to_id_indicies, tokenids_chunks = StaticReps.prepare_sentence(tokenizer, text)
            counts.update(word.translate(str.maketrans('','',string.punctuation)) for word in tokenized_text)
        del counts['']
        return (counts, tokenized_text, tokenized_to_id_indicies, tokenids_chunks)
    

    @staticmethod
    def tokenize_and_get_wordrep(texts, tokenizer, counts, args, model):
        logging.info('Computing word representations...')
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
    def write_tokenized(fname_tokenized, tokenization_info):
        logging.info('Caching tokenized data to "%s"' % fname_tokenized)
        os.makedirs(Constants.DPATH_CACHED) if not Constants.DPATH_CACHED.exists() else None
        with open(Constants.DPATH_CACHED/fname_tokenized, 'wb') as f:
            pickle.dump({'tokenization_info': tokenization_info}, f, protocol=4)


    @staticmethod
    def write_staticreps(fname_staticreps, vocab_words, static_word_representations, vocab_occurrence):
        logging.info('Caching static word representations to "%s"' % fname_staticreps)
        os.makedirs(Constants.DPATH_CACHED) if not Constants.DPATH_CACHED.exists() else None
        with open(Constants.DPATH_CACHED/fname_staticreps, 'wb') as f:
            pickle.dump({
                'static_word_representations': static_word_representations,
                'vocab_words': vocab_words,
                'word_to_index': {v: k for k, v in enumerate(vocab_words)},
                'vocab_occurrence': vocab_occurrence,
            }, f, protocol=4)

    
    # -------------------------------------------------------------------------
    def main(self, args, texts):
        fname_tokenized = 'tokenization_lm-%s-%s.pickle' % (args.lm_type, args.layer)
        fname_staticreps = 'static_repr_lm-%s-%s.pickle' % (args.lm_type, args.layer)
        if (Constants.DPATH_CACHED/fname_tokenized).exists() and (Constants.DPATH_CACHED/fname_staticreps).exists():
            logging.info('Static Representations already computed. Using cached version.')
        else:
            logging.info('Computing Static Representations...')
            model_class, tokenizer_class, pretrained_weights = Constants.MODEL
            tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
            model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
            model.eval()
            model.cuda()

            counts, tokenized_text, tokenized_to_id_indicies, tokenids_chunks = self.tokenize_and_count_tokens(texts, tokenizer)
            word_rep, word_count, tokenization_info = self.tokenize_and_get_wordrep(texts, tokenizer, counts, args, model)
            word_avg, vocab_words, static_word_representations, vocab_occurrence = self.vocab_stats(word_rep, word_count)

            self.write_tokenized(fname_tokenized, tokenization_info)
            self.write_staticreps(fname_staticreps, vocab_words, static_word_representations, vocab_occurrence)
            logging.info('Completed Static Representation computations')



# -----------------------------------------------------------------------------
# Class-Oriented Document Representations; 
#   - clusters vocabulary with each class
#   - vectorizes each doc based on class vocab
class ClassDocReps:
    @staticmethod
    def read_staticreps(args):
        fname_staticreps = 'static_repr_lm-%s-%s.pickle' % (args.lm_type, args.layer)
        logging.info('Reading static word representations from "%s"' % fname_staticreps)
        with open(Constants.DPATH_CACHED/fname_staticreps, 'rb') as f:
            return pickle.load(f)


    @staticmethod
    def read_tokenized(args):
        fname_tokenized = 'tokenization_lm-%s-%s.pickle' % (args.lm_type, args.layer)
        logging.info('Reading tokenized data from "%s"' % fname_tokenized)
        with open(Constants.DPATH_CACHED/fname_tokenized, 'rb') as f:
            tokenization_info = pickle.load(f)['tokenization_info']
        return tokenization_info


    @staticmethod
    def rank_by_significance(embeddings, class_embeddings):
        similarities = CDRU.cosine_similarity_embeddings(embeddings, class_embeddings)
        significance_score = [np.max(softmax(similarity)) for similarity in similarities]
        significance_ranking = {i: r for r, i in enumerate(np.argsort(-np.array(significance_score)))}
        return significance_ranking


    @staticmethod
    def rank_by_relation(embeddings, class_embeddings):
        relation_score = CDRU.cosine_similarity_embeddings(embeddings, [np.average(class_embeddings, axis=0)]).reshape((-1))
        relation_ranking = {i: r for r, i in enumerate(np.argsort(-np.array(relation_score)))}
        return relation_ranking

    
    @staticmethod
    def mul(l):
        m = 1
        for x in l:
            m *= x + 1
        return m


    @staticmethod
    def average_with_harmonic_series(representations):
        weights = [0.0] * len(representations)
        for i in range(len(representations)):
            weights[i] = 1. / (i + 1)
        return np.average(representations, weights=weights, axis=0)


    @staticmethod
    def weights_from_ranking(rankings):
        if len(rankings) == 0:
            assert False
        if type(rankings[0]) == type(0):
            rankings = [rankings]
        rankings_num = len(rankings)
        rankings_len = len(rankings[0])
        assert all(len(rankings[i]) == rankings_len for i in range(rankings_num))
        total_score = []
        for i in range(rankings_len):
            total_score.append(ClassDocReps.mul(ranking[i] for ranking in rankings))

        total_ranking = {i: r for r, i in enumerate(np.argsort(np.array(total_score)))}
        if rankings_num == 1:
            assert all(total_ranking[i] == rankings[0][i] for i in total_ranking.keys())
        weights = [0.0] * rankings_len
        for i in range(rankings_len):
            weights[i] = 1. / (total_ranking[i] + 1)
        return weights


    @staticmethod
    def weight_sentence_with_attention(vocab, tokenized_text, contextualized_word_representations, class_representations,
                                    attention_mechanism):
        assert len(tokenized_text) == len(contextualized_word_representations)

        contextualized_representations = []
        static_representations = []

        static_word_representations = vocab["static_word_representations"]
        word_to_index = vocab["word_to_index"]
        for i, token in enumerate(tokenized_text):
            if token in word_to_index:
                static_representations.append(static_word_representations[word_to_index[token]])
                contextualized_representations.append(contextualized_word_representations[i])
        if len(contextualized_representations) == 0:
            print("Empty Sentence (or sentence with no words that have enough frequency)")
            return np.average(contextualized_word_representations, axis=0)

        significance_ranking = ClassDocReps.rank_by_significance(contextualized_representations, class_representations)
        relation_ranking = ClassDocReps.rank_by_relation(contextualized_representations, class_representations)
        significance_ranking_static = ClassDocReps.rank_by_significance(static_representations, class_representations)
        relation_ranking_static = ClassDocReps.rank_by_relation(static_representations, class_representations)
        if attention_mechanism == "none":
            weights = [1.0] * len(contextualized_representations)
        elif attention_mechanism == "significance":
            weights = ClassDocReps.weights_from_ranking(significance_ranking)
        elif attention_mechanism == "relation":
            weights = ClassDocReps.weights_from_ranking(relation_ranking)
        elif attention_mechanism == "significance_static":
            weights = ClassDocReps.weights_from_ranking(relation_ranking)
        elif attention_mechanism == "relation_static":
            weights = ClassDocReps.weights_from_ranking(relation_ranking)
        elif attention_mechanism == "mixture":
            weights = ClassDocReps.weights_from_ranking((significance_ranking,
                                            relation_ranking,
                                            significance_ranking_static,
                                            relation_ranking_static))
        else:
            assert False
        return np.average(contextualized_representations, weights=weights, axis=0)


    @staticmethod
    def weight_sentence(model, vocab, tokenization_info, class_representations, attention_mechanism, layer):
        tokenized_text, tokenized_to_id_indicies, tokenids_chunks = tokenization_info
        contextualized_word_representations = StaticReps.handle_sentence(
            model, layer, tokenized_text, tokenized_to_id_indicies, tokenids_chunks)
        document_representation = ClassDocReps.weight_sentence_with_attention(
            vocab, tokenized_text, contextualized_word_representations, class_representations, attention_mechanism)
        return document_representation


    @staticmethod
    def print_wordreps(classnames, class_words, cls):
        print('-'*60)
        logging.info('Word representations for class "%s" (label_id %s)...\n%s' % (
            classnames[cls], cls, ', '.join(class_words[cls])))


    @staticmethod
    def get_class_representations(args, classnames, static_word_representations, word_to_index, vocab_words):
        logging.info('Computing class representations...')
        finished_class = set()
        masked_words = set(classnames)
        cls_repr = [None for _ in range(len(classnames))]
        class_words = [[classnames[cls]] for cls in range(len(classnames))]
        class_words_representations = [[static_word_representations[word_to_index[classnames[cls]]]]
                                    for cls in range(len(classnames))]
        for t in range(1, args.T):
            class_representations = [ClassDocReps.average_with_harmonic_series(class_words_representation)
                                    for class_words_representation in class_words_representations]
            cosine_similarities = CDRU.cosine_similarity_embeddings(static_word_representations,
                                                            class_representations)
            nearest_class = cosine_similarities.argmax(axis=1)
            similarities = cosine_similarities.max(axis=1)
            for cls in range(len(classnames)):
                if cls in finished_class:
                    continue
                highest_similarity = -1.0
                highest_similarity_word_index = -1
                lowest_masked_words_similarity = 1.0
                existing_class_words = set(class_words[cls])
                stop_criterion = False
                for i, word in enumerate(vocab_words):
                    if nearest_class[i] == cls:
                        if word not in masked_words:
                            if similarities[i] > highest_similarity:
                                highest_similarity = similarities[i]
                                highest_similarity_word_index = i
                        else:
                            if word not in existing_class_words:
                                stop_criterion = True
                                break
                            lowest_masked_words_similarity = min(lowest_masked_words_similarity, similarities[i])
                    else:
                        if word in existing_class_words:
                            stop_criterion = True
                            break
                # the topmost t words are no longer the t words in class_words
                if lowest_masked_words_similarity < highest_similarity:
                    stop_criterion = True

                if stop_criterion:
                    finished_class.add(cls)
                    class_words[cls] = class_words[cls][:-1]
                    class_words_representations[cls] = class_words_representations[cls][:-1]
                    cls_repr[cls] = ClassDocReps.average_with_harmonic_series(class_words_representations[cls])
                    ClassDocReps.print_wordreps(classnames, class_words, cls)
                    break
                class_words[cls].append(vocab_words[highest_similarity_word_index])
                class_words_representations[cls].append(static_word_representations[highest_similarity_word_index])
                masked_words.add(vocab_words[highest_similarity_word_index])
                cls_repr[cls] = ClassDocReps.average_with_harmonic_series(class_words_representations[cls])
            if len(finished_class) == len(classnames):
                break
        print()
        return (class_words, cls_repr)


    @staticmethod
    def get_class_oriented_doc_representations(args, vocab, tokenization_info, cls_repr):
        logging.info('Computing the class-oriented representation of each document...')
        class_representations = np.array(cls_repr)
        model_class, tokenizer_class, pretrained_weights = Constants.MODEL
        model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
        model.eval()
        model.cuda()
        document_representations = []
        for i, _tokenization_info in tqdm.tqdm(enumerate(tokenization_info), total=len(tokenization_info)):
            document_representation = ClassDocReps.weight_sentence(
                model, vocab, _tokenization_info, class_representations, args.attention_mechanism, args.layer)
            document_representations.append(document_representation)
        document_representations = np.array(document_representations)
        return (class_representations, document_representations)
    
        
    @staticmethod
    def write_classdocreps(fname_classdocreps, class_words, class_representations, document_representations):
        logging.info('Caching class-oriented document representations to "%s"' % fname_classdocreps)
        os.makedirs(Constants.DPATH_CACHED) if not Constants.DPATH_CACHED.exists() else None
        with open(Constants.DPATH_CACHED/fname_classdocreps, 'wb') as f:
            pickle.dump({
                'class_words': class_words,
                'class_representations': class_representations,
                'document_representations': document_representations,
            }, f, protocol=4)


    # -------------------------------------------------------------------------
    def main(self, args, classnames):
        fname_classdocreps = 'document_repr_lm-%s-%s-%s-%s.pickle' % (args.lm_type, args.layer, args.attention_mechanism, args.T)
        if (Constants.DPATH_CACHED/fname_classdocreps).exists():
            logging.info('Class-Oriented Document Representations already computed. Using cached version.')
            with open(Constants.DPATH_CACHED/fname_classdocreps, 'rb') as f:
                classdocreps = pickle.load(f)
            for class_id, class_name in enumerate(classnames):
                self.print_wordreps(classnames, classdocreps['class_words'], class_id)
        else:
            logging.info('Computing Class-Oriented Document Representations...')
            vocab = self.read_staticreps(args)
            vocab_words, static_word_representations = vocab['vocab_words'], vocab['static_word_representations']
            word_to_index, vocab_occurrence = vocab['word_to_index'], vocab['vocab_occurrence']
            tokenization_info = self.read_tokenized(args)
            
            class_words, cls_repr = self.get_class_representations(
                args, classnames, static_word_representations, word_to_index, vocab_words)
            
            class_representations, document_representations = self.get_class_oriented_doc_representations(
                args, vocab, tokenization_info, cls_repr)

            self.write_classdocreps(fname_classdocreps, class_words, class_representations, document_representations)

            logging.info('Completed Class-Oriented Document Representation computations')

