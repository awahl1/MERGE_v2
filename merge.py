import itertools
import os
from collections import defaultdict, namedtuple, Counter

import numpy as np
import pandas as pd

NTword = namedtuple('NTword', 'wordstr position')
NTlexeme = namedtuple('NTlexeme','lex token_index')
NTlexemeWithPointer = namedtuple('NTlexemeWithPointer', 'lexeme pointer')
NTbigram = namedtuple('NTbigram','el1 el2 gapsize')
NTwinner = namedtuple('NTwinner','bigram log_likelihood table_num')
NTcontextpos = namedtuple("NTcontextpos", "context_pos satellite_pos gapsize merge_token_leftanchor")


DELIMITER_SET = set([" ",".",",","?",";",":","!","\r","\n"])


def tokenize(line):
    for index, char in enumerate(line):
        if char not in DELIMITER_SET:
            if 'left_edge' not in locals():
                left_edge=index
        else:
            if 'left_edge' in locals():
                yield line[left_edge:index]
                del left_edge
    if 'left_edge' in locals():
        yield line[left_edge:len(line)]




class Lexemes:

    def __init__(self):
        self.words2lexemes = {}
        self.lexemes2locs = defaultdict(set)
        self.locs2lexemes = defaultdict(dict)
        self.lexemes2freqs = Counter()

    def add_loc(self, word, utt_idx, word_idx):
        if word not in self.words2lexemes:
            self.words2lexemes[word] = NTlexeme(lex=(NTword(wordstr=word, position=0),), token_index=0)
        lexeme = self.words2lexemes[word]
        loc = (utt_idx, word_idx)
        self.lexemes2locs[lexeme].add(loc)
        self.locs2lexemes[utt_idx][word_idx] = lexeme


    def get_anchor_lexeme(self, utt_idx, word_idx):
        """
        Given an utterance index and position in the utterance, fetch the lexeme and its leftanchor position that contains
        the passed position
        :param line_idx:
        :param word_idx:
        :return:
        """
        placeholder_lexeme = self.locs2lexemes[utt_idx][word_idx]
        leftanchor_pos_in_utt = word_idx - placeholder_lexeme.token_index
        leftanchor_lexeme = self.locs2lexemes[utt_idx][leftanchor_pos_in_utt]
        return leftanchor_lexeme, leftanchor_pos_in_utt


    def update_freqs(self, merge_token, merge_token_count, winner, merged_satellite_positions):

        satel_lexemes = {pos: NTlexeme(lex=merge_token.lex, token_index=pos) for wordstr, pos in merge_token.lex if pos > 0}
        satel_lexemes[0] = merge_token

        # update all lexemes with merge tokens
        self.lexemes2freqs[merge_token] = merge_token_count
        for (utt_idx, satellite_pos), merge_token_leftanchor in merged_satellite_positions.items():
            satel_lexeme = satel_lexemes[satellite_pos - merge_token_leftanchor]
            self.lexemes2locs[satel_lexeme].add((utt_idx, satellite_pos))
            self.locs2lexemes[utt_idx][satellite_pos] = satel_lexeme

        for element in (winner.bigram.el1, winner.bigram.el2):
            self.lexemes2freqs[element] -= merge_token_count
            if self.lexemes2freqs[element] < 1:
                del self.lexemes2freqs[element]

        new_freqs = []
        for element in (winner.bigram.el1, winner.bigram.el2):
            if element not in self.lexemes2freqs:
                new_freqs.append(0)
            else:
                new_freqs.append(self.lexemes2freqs[element])

        return new_freqs




class Corpus:

    def __init__(self, corpus_files_dir=None, file_extension='txt', gap_sizes=None):

        self.corpus_files_dir = corpus_files_dir
        self.file_extension = file_extension
        self.lexemes = Lexemes()
        self.bigrams = Bigrams()
        self.size = 0
        self.utterance_lengths = Counter()
        self.gap_sizes = gap_sizes
        self.utt_idx2filename = {}
        self.utt_idx2utt = {}

        self._generate_initial_lexemes()
        self._count_initial_lexemes()
        self._generate_initial_bigrams()


    def _generate_initial_lexemes(self):
        utt_idx = 0
        for fn in os.listdir(self.corpus_files_dir):
            if fn.endswith(self.file_extension):
                self.utt_idx2utt[utt_idx] = fn
                with open(os.path.join(self.corpus_files_dir, fn), 'r') as f:
                    for line in f:
                        self.utt_idx2utt[utt_idx] = line
                        for word_idx, word in enumerate(tokenize(line)):
                            self.lexemes.add_loc(
                                word.lower(),
                                utt_idx,
                                word_idx
                            )
                            self.size += 1
                            self.utterance_lengths[utt_idx] += 1
                        utt_idx += 1

        del self.lexemes.words2lexemes

    def _count_initial_lexemes(self):
        for lexeme, locs in self.lexemes.lexemes2locs.items():
            self.lexemes.lexemes2freqs[lexeme] = len(locs)

    def _generate_initial_bigrams(self):
        cartesian_product = itertools.product(*(self.utterance_lengths.items(), self.gap_sizes))
        for (utt_idx, utt_len), gap_size in cartesian_product:
            rightmost_leftedge = utt_len - gap_size - 1
            for lex_idx in range(rightmost_leftedge):
                left_lexeme, _ = self.lexemes.get_anchor_lexeme(utt_idx, lex_idx)
                right_lexeme, _ = self.lexemes.get_anchor_lexeme(utt_idx, lex_idx + gap_size + 1)
                self.bigrams.save(left_lexeme, right_lexeme, gap_size, utt_idx, lex_idx)


    def generate_bigram_updates(self, merge_token, winner):

        new_bigrams = Bigrams(include_lexmaps=False)
        conflicting_bigrams = Bigrams(include_lexmaps=False)
        merge_token_count = 0
        merged_satellite_positions = {}
        for utt_idx, merge_token_leftanchor in self.bigrams.bigrams2locations[winner.bigram]:
            if not (utt_idx, merge_token_leftanchor) in conflicting_bigrams.bigrams2locations[winner.bigram]:
                merge_token_count += 1
                utterance_len = self.utterance_lengths[utt_idx]
                # utterance-level positions of each "satellite" wordstr in the merge token
                absolute_satel_positions = {satel_pos + merge_token_leftanchor for wordstr, satel_pos in merge_token.lex}

                # get every "context" position around each wordstr up to the permitted gap size
                context_positions = []
                for pos in absolute_satel_positions:
                    for gapsize in self.gap_sizes:
                        for context_pos in {'left': pos - gapsize - 1, 'right': pos + gapsize + 1}.values():
                            if 0 <= context_pos < utterance_len:
                                context_positions.append(
                                    NTcontextpos(
                                        context_pos=context_pos,
                                        satellite_pos=pos,
                                        gapsize=gapsize,
                                        merge_token_leftanchor=merge_token_leftanchor
                                    )
                                )

                for context_position in context_positions:
                    # make sure the context position isn't itself a satellite wordstr
                    if context_position.context_pos not in absolute_satel_positions:

                        # get the lexeme that the corresponding satellite wordstr belongs to (one of two lexemes making up the winning bigram)
                        premerge_lexeme, premerge_leftanchor = self.lexemes.get_anchor_lexeme(
                            utt_idx, context_position.satellite_pos
                        )

                        # if the context position is also a satellite position within an already-identified merge token within the same utterance
                        if (utt_idx, context_position.context_pos) in merged_satellite_positions:
                            context_lexeme = merge_token
                            context_leftanchor = merged_satellite_positions[(utt_idx, context_position.context_pos)]

                        # if not, get the lexeme the context position belongs to
                        else:
                            context_lexeme, context_leftanchor = self.lexemes.get_anchor_lexeme(
                                utt_idx, context_position.context_pos
                            )

                        # create new bigram
                        if context_leftanchor < context_position.merge_token_leftanchor:
                            gap_between_leftanchors = context_position.merge_token_leftanchor - context_leftanchor - 1
                            new_bigrams.save(context_lexeme, merge_token, gap_between_leftanchors, utt_idx, context_leftanchor)
                        else:
                            gap_between_leftanchors = context_leftanchor - context_position.merge_token_leftanchor - 1
                            new_bigrams.save(merge_token, context_lexeme, gap_between_leftanchors, utt_idx, context_position.merge_token_leftanchor)

                        # create conflicting bigram
                        if context_leftanchor < context_position.merge_token_leftanchor:
                            gap_between_leftanchors = premerge_leftanchor - context_leftanchor - 1
                            conflicting_bigrams.save(context_lexeme, premerge_lexeme, gap_between_leftanchors, utt_idx, context_leftanchor)
                        else:
                            gap_between_leftanchors = context_leftanchor - premerge_leftanchor - 1
                            conflicting_bigrams.save(premerge_lexeme, context_lexeme, gap_between_leftanchors, utt_idx, premerge_leftanchor)

                # create mapping from satellite positions in a merge token to that merge token's leftanchor position
                for satel_pos in absolute_satel_positions:
                    merged_satellite_positions[(utt_idx, satel_pos)] = merge_token_leftanchor

        return new_bigrams, conflicting_bigrams, merged_satellite_positions, merge_token_count





def get_bigrams_containing(self, lexeme=None, in_position=None, max_gapsize=None):

    self.hits = set()
    for gapsize in range(max_gapsize+1):
        if in_position==1 and (lexeme,gapsize) in self.left_lex_to_bigrams:
            self.hits = self.hits.union(self.left_lex_to_bigrams[(lexeme,gapsize)])
        if in_position==2 and (lexeme,gapsize) in self.right_lex_to_bigrams:
            self.hits = self.hits.union(self.right_lex_to_bigrams[(lexeme,gapsize)])
    return self.hits




class Bigrams:

    def __init__(self, include_lexmaps=True):
        self.bigrams2freqs = Counter()
        self.bigrams2locations = defaultdict(set)
        self.include_lexmaps = include_lexmaps
        if self.include_lexmaps:
            self.left_lex_to_bigrams = defaultdict(set)
            self.right_lex_to_bigrams = defaultdict(set)

    def save(self, left_lexeme, right_lexeme, gap_size, utt_idx, lex_idx):
        bigram = NTbigram(el1=left_lexeme, el2=right_lexeme, gapsize=gap_size)
        if self.include_lexmaps and bigram not in self.bigrams2freqs:
            self.left_lex_to_bigrams[(left_lexeme, gap_size)].add(bigram)
            self.right_lex_to_bigrams[(right_lexeme, gap_size)].add(bigram)
        self.bigrams2freqs[bigram] += 1
        self.bigrams2locations[bigram].add((utt_idx, lex_idx))

    def union(self, other_bigrams):

        for bigram, freq in other_bigrams.bigrams2freqs.items():

            self.bigrams2freqs[bigram] += freq
            self.bigrams2locations[bigram].update(other_bigrams.bigrams2locations[bigram])
            if self.include_lexmaps:
                self.left_lex_to_bigrams[(bigram.el1, bigram.gapsize)].update(bigram)
                self.right_lex_to_bigrams[(bigram.el2, bigram.gapsize)].update(bigram)

    def subtract_all(self, other_bigrams):

        for bigram, freq in other_bigrams.bigrams2freqs.items():
            self.bigrams2freqs[bigram] -= freq
            for loc in other_bigrams.bigrams2locations[bigram]:
                self.bigrams2locations[bigram].remove(loc)
            if self.bigrams2freqs[bigram] < 1:
                del self.bigrams2freqs[bigram]
                del self.bigrams2locations[bigram]
                if self.include_lexmaps:
                    self.left_lex_to_bigrams[(bigram.el1, bigram.gapsize)].remove(bigram)
                    self.right_lex_to_bigrams[(bigram.el2, bigram.gapsize)].remove(bigram)

    def remove(self, bigram):
        try:
            del self.bigrams2freqs[bigram]
            del self.bigrams2locations[bigram]
            if self.include_lexmaps:
                self.left_lex_to_bigrams[(bigram.el1, bigram.gapsize)].remove(bigram)
                self.right_lex_to_bigrams[(bigram.el2, bigram.gapsize)].remove(bigram)
        except KeyError:
            pass




def _get_column_collector():
    return {'row_indices': [], 'bigram_freqs': [], 'el1_freqs': [], 'el2_freqs': []}

class Tables:

    def __init__(self, max_row_count=50000):
        self.max_row_count = max_row_count
        self.column_collector = _get_column_collector()
        self._tables = {}
        self.bigrams2tablenum = {}

    def add_row(self, bigram=None, bigram_freq=None, el1_freq=None, el2_freq=None):
        self.column_collector['row_indices'].append(bigram)
        self.column_collector['bigram_freqs'].append(bigram_freq)
        self.column_collector['el1_freqs'].append(el1_freq)
        self.column_collector['el2_freqs'].append(el2_freq)
        if len(self.column_collector['row_indices']) == self.max_row_count:
            self.write_table()

    def write_table(self):
        if self.column_collector['row_indices']:
            row_indices = self.column_collector.pop('row_indices')
            df = pd.DataFrame(self.column_collector, index=row_indices)
            table_num = len(self._tables)
            self._tables[table_num] = df
            self.bigrams2tablenum.update({bigram: table_num for bigram in row_indices})
            self.column_collector = _get_column_collector()

    def get_winner(self, corpus_size):

        table_winners = []
        for table_num, df in self._tables.items():
            bf, el1, el2 = df['bigram_freqs'], df['el1_freqs'], df['el2_freqs']

            obs_exp = {}
            obs_exp['A'] = (bf,                             el2 / corpus_size * el1)
            obs_exp['B'] = (el1 - bf,                       (corpus_size + el2) / corpus_size * el1)
            obs_exp['C'] = (el2 - bf,                       el2 / corpus_size * (corpus_size - el1))
            obs_exp['D'] = (el1 + el2 - bf,                 (corpus_size - el2) / corpus_size * (corpus_size - el1))

            ll_components = {}
            for quadrant, (obs, exp) in obs_exp.items():
                real_vals = np.where(obs != 0)[0]
                component = np.zeros(len(df))
                component[real_vals] = get_log_likelihood(obs[real_vals], exp[real_vals])
                ll_components[quadrant] = component

            log_likelihood = 2 * sum([v for v in ll_components.values()])
            negs = np.where(ll_components['A'] < 0)[0]
            log_likelihood[negs] = log_likelihood[negs] * -1

            winner = log_likelihood.argmax()
            winner_ll = log_likelihood[winner]
            winner_bigram = df.index[winner]
            table_winners.append(NTwinner(bigram=winner_bigram, log_likelihood=winner_ll, table_num=table_num))

        return sorted(table_winners, key=lambda tup: tup[1], reverse=True)[0]

    def update(self, bigram=None, column=None, new_value=None):

        self._tables[self.bigrams2tablenum[bigram]].loc[bigram, column] = new_value

    def remove_zero_freqs(self):

        updated_tables = {}
        for table_num, df in self._tables.items():
            indexer = df['bigram_freqs'] > 0
            updated_df = df.loc[indexer]
            updated_tables[table_num] = updated_df
            for bigram in indexer[~indexer].index:
                del self.bigrams2tablenum[bigram]
        self._tables = updated_tables



def get_log_likelihood(obs, exp):
    return obs * np.log(obs / exp)

FILE_DIR = '/Users/alex.wahl/Repos/MERGE/TRN_cleaned'

class Merge:

    def __init__(self, corpus_files_dir=FILE_DIR, file_extension='txt', max_gap_size=0, max_row_count=50000):

        self.gap_sizes = range(max_gap_size + 1)
        self.corpus = Corpus(corpus_files_dir, file_extension, self.gap_sizes)
        self.tables = Tables(max_row_count)
        self.merge_tracker = {}

        # initial table setup
        for idx, (bigram, freq) in enumerate(self.corpus.bigrams.bigrams2freqs.items()):
            el1_freq = self.corpus.lexemes.lexemes2freqs[bigram.el1]
            el2_freq = self.corpus.lexemes.lexemes2freqs[bigram.el2]
            self.tables.add_row(bigram, freq, el1_freq, el2_freq)
        self.tables.write_table()

    def run(self, iterations=10000):

        for iteration in range(1, iterations + 1):
            winner = self.tables.get_winner(self.corpus.size)
            merge_token = self.merge(winner.bigram)
            self.merge_tracker[iteration] = (winner.bigram, winner.log_likelihood)
            new_bigrams, conflicting_bigrams, merged_satellite_positions, merge_token_count  = \
                self.corpus.generate_bigram_updates(merge_token, winner)

            self.corpus.size -= merge_token_count

            new_el1_freq, new_el2_freq = self.corpus.lexemes.update_freqs(
                merge_token, merge_token_count, winner, merged_satellite_positions
            )

            for element, column in ((winner.bigram.el1, 'el1_freqs'), (winner.bigram.el2, 'el2_freqs')):
                for mapping, freq in ((self.corpus.bigrams.left_lex_to_bigrams, new_el1_freq),
                                      (self.corpus.bigrams.right_lex_to_bigrams, new_el2_freq)):
                    for gapsize in self.gap_sizes:
                        if (element, gapsize) in mapping:
                            for bigram in mapping[(element, gapsize)]:
                                self.tables.update(bigram, column, freq)

            self.tables.remove_zero_freqs()
            self.corpus.bigrams.union(new_bigrams)
            for bigram in new_bigrams.bigrams2freqs.keys():
                bg_freq = self.corpus.bigrams.bigrams2freqs[bigram]
                el1_freq = self.corpus.lexemes.lexemes2freqs[bigram.el1]
                el2_freq = self.corpus.lexemes.lexemes2freqs[bigram.el2]
                self.tables.add_row(bigram, bg_freq, el1_freq, el2_freq)
            self.tables.write_table()

            self.corpus.bigrams.subtract_all(conflicting_bigrams)
            for bigram in conflicting_bigrams.bigrams2freqs.keys():
                new_freq = self.corpus.bigrams.bigrams2freqs[bigram]
                self.tables.update(bigram, 'bigram_freqs', new_freq)

            self.corpus.bigrams.remove(winner.bigram)
            self.tables.update(winner.bigram, 'bigram_freqs', 0)
            self.tables.remove_zero_freqs()


    @staticmethod
    def merge(bigram):
        el1_words = list(bigram.el1.lex)
        repositioned_el2_words = [
            NTword(wordstr=word, position=pos + bigram.gapsize + 1) for word, pos in bigram.el2.lex
        ]
        all_words = tuple(sorted(el1_words + repositioned_el2_words, key=lambda tup: tup[1]))
        return NTlexeme(lex=all_words, token_index=0)



