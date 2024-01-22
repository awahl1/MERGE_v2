
# MERGE algorithm

## Initial corpus representation

The user provides a path to an input corpus, which must comprise a set of text files containing sequences of newline-delimited utterances (i.e., sentences). At initialization, the algorithm reads in and parses the corpus into a sequence of "lexemes" and "bigrams." A lexeme is defined as a single item in the lexicon; initially, each word string will constitute one lexeme. A bigram is defined as a conjunct of two neighboring lexemes. The user also defines a maximum gap size parameter, such that all possible bigrams of neighboring lexemes at a distance up to and including the maximum gap size are calculated. The frequencies of all lexemes and bigrams are counted.

## Choosing a winner

For each bigram, a log likelihood word attraction score is calculated, which represents how much more or less likely the particuar combination of words is than might be expected by chance. The bigram with the highest log likelihood score is chosen as the "winner" and is merged into a new lexeme; this new lexeme will now contain the two word strings of the winning bigram, with each word string indexed for how far to the right it is from the leftmost wordstring in the lexeme (i.e., 0 or 1).

## Updating the corpus

For every given instance of the winning bigram in the corpus, both component lexemes are replaced by the merged lexeme. The righthand "satellite" lexeme instance is merely a placeholder to maintain the ordered structure of the corpus; it contains an index that indicates how far it is from the left edge of the true lefthand "anchor" lexeme.

New bigrams are then generated with respect to the instances of the newly minted lexeme, and the new bigram frequencies are counted. Similarly, conflicting bigrams are generated with respect to the component lexemes of the winning bigram -- that is, instances of bigrams that no longer exist as a result of the merge. The frequencies of these conflicting bigrams are decremented.

A new winner is calculated, merged, and the corpus is updated as above, and this process continues for some number of iterations specified by the user.



utterance index (utt_idx): the corpus-level index of the utterance

word index: the utterance-level index of a word



lexeme

- lex: sequence of wordstr's
- token position: *I think* that, given that each position in an utterance is occupied by one lexeme token, the token position indicates the relative position of each token from the leftmost one (in the case of multi-word lexemes)