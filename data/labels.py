'''
This file creates two dictionaries: `attr_idx` and `idx_attr`.

`attr_idx` is a dictionary of dictionaries; the keys are the five different
morphological properties/attributes considered for morphological analysis: part
of speech (POS), person, gender, number, tense, and definiteness. Each key maps
to a dictionary that maps possible values of the given attribute to an index
(e.g., 'adjective' > 0). These indices are later used for one-hot encoding
label vectors.

`idx_attr` is a reversal of `attr_idx`, used for identifying the value of an
attribute given the value's index (e.g., 0 > 'adjective'). In other words, each
inner dictionary for each attribute maps the indices to their corresponding
values.

Together, these two dictionaries define the gold standard for the morphological
analyzer. The comments below contain brief discussions as to where and why some
attribute value diverge from the HaAretz labels.
'''

attr_idx = {

    # part-of-speech ----------------------------------------------------------

    'pos': {  # 22 labels in COMPLEMENTARY distribution

        # CORPUS LABELS not included:
        #   - foreign > 1
        #   - unknown > 58

        #                             CORPUS COUNTS
        'adjective': 0,             # 5181
        'adverb': 1,                # 2282
        'conjunction': 2,           # 2017
        'copula': 3,                # 1021
        'existential': 4,           # 381
        'interjection': 5,          # 19
        'interrogative': 6,         # 291
        'modal': 7,                 # 458

        # multi-word expression
        'mwe': 8,                   # 4042

        'negation': 9,             # 679
        'noun': 10,                 # 20234
        'numberexpression': 11,     # 38
        'numeral': 12,              # 1896
        'participle': 13,           # 2526
        'preposition': 14,          # 5790
        'pronoun': 15,              # 1766
        'propername': 16,           # 4325
        'punctuation': 17,          # 11747
        'quantifier': 18,           # 673
        'title': 19,                # 56
        'verb': 20,                 # 5583
        'wprefix': 21,              # 120
        },

    # person ------------------------------------------------------------------

    'person': {  # 3 labels in OVERLAPPING distribution

        # n.b.:
        #   - The corpus label 'any' will map to '1', '2', AND '3'

        # CORPUS LABELS not included:
        #   - any > 2792
        #   - unspecified > 8
        #   - n/a > 61794

        #                     CORPUS COUNTS
        '1': 0,             # 400
        '2': 1,             # 125
        '3': 2,             # 6065
        },

    # gender ------------------------------------------------------------------

    'gender': {  # 2 labels in OVERLAPPING distribution

        # n.b.:
        #   - The corpus label 'masculine and feminine' will map to 'feminine'
        #     AND 'masculine'

        # CORPUS LABELS not included:
        #   - masculine and feminine > 23287
        #   - unspecified > 3
        #   - n/a > 32351

        #                     CORPUS COUNTS
        'feminine': 0,      # 13623
        'masculine': 1,     # 23287
        },

    # number ------------------------------------------------------------------

    'number': {  # 3 labels in OVERLAPPING distribution

        # n.b.:
        #   - Since the notion of 'dual' is inherently plural, I collapse the
        #     corpus labels 'dual' and 'dual and plural', such that they both
        #     will map to 'dual' AND 'plural'
        #   - The corpus label 'singular and plural' will map to 'singular' AND
        #     'plural'

        # CORPUS LABELS not included:
        #   - singular and plural > 7
        #   - dual and plural > 7
        #   - unspecified > 9
        #   - n/a > 32309

        #                     CORPUS COUNTS
        'dual': 0,          # 184
        'singular': 1,      # 28010
        'plural': 2,        # 10658
        },

    # tense -------------------------------------------------------------------

    'tense': {  # 6 labels in COMPLEMENTARY distribution

        # n.b.:
        #   - Since a bare infinitive is still an infinitive, I collapse the
        #     corpus labels for 'bareInfinitive' and 'infinitive', such that
        #     they both will map to 'infinitive'

        # CORPUS LABELS not included:
        #   - bareInfinitive > 8
        #   - unspecified > 1
        #   - n/a > 64327

        #                     CORPUS COUNTS
        'past': 0,          # 3776
        'beinoni': 1,       # 533
        'present': 2,       # 7
        'future': 3,        # 856
        'infinitive': 4,    # 1641
        'imperative': 5,    # 35
        },

    # definiteness ------------------------------------------------------------

    'definiteness': {  # 2 labels in COMPLEMENTARY distribution

        # n.b.:
        #   - IN THE FUTURE, it is probably better to have a single 'definite'
        #     label, in lieu of 'true' and 'false' for definiteness.

        # CORPUS LABELS not included:
        #   - unspecified > 3
        #   - n/a > 40055

        #                     CORPUS COUNTS
        'false': 0,         # 21139
        'true': 1,          # 9987
        },

    }

idx_attr = {}

for attr, values in attr_idx.items():
    idx_attr[attr] = {v: k for k, v in values.items()}
