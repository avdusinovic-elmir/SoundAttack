target = "Experience is important"
tests = ["", # Empty string
         "Knowledge is important", # 1 Error
         "Experience is not important", # 1 Errors
         "Experience was important", # 1 Error
         "Experience is important!", # 1 Error
         "Nothing happened here", # 3 Errors
         "Experience is important" # Target string repeated
         ]

# Edit distance
# The smaller the edit distance, the more similar the sentences are in terms of the number of insertions,
# deletions, or substitutions needed to transform one into the other.
# https://pypi.org/project/editdistance/
def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

# Cosine Similarity of Word Embeddings
# Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space.
# It is defined to equal the cosine of the angle between them, which is also the same as the inner product of
# the same vectors normalized to both have length 1.
# The cosine similarity of two strings is calculated by converting the strings into vectors of word embeddings
# and then calculating the cosine similarity of the vectors.
# The smaller the cosine similarity, the more similar the sentences are.
# 1 means the sentences are identical, 0 means the sentences are completely different.
# https://medium.com/@techclaw/cosine-similarity-between-two-arrays-for-word-embeddings-c8c1c98811b#:~:text=Cosine%20similarity%20measures%20the%20cosine,and%20%2D1%20indicating%20opposite%20vectors.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity_of_word_embeddings(target, test):
    vectorizer = CountVectorizer().fit([target, test])
    vectors = vectorizer.transform([target, test]).toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

# TF-IDF Similarity
# TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure used to evaluate how important
# a word is to a document in a collection or corpus. The importance increases proportionally to the number of
# times a word appears in the document but is offset by the frequency of the word in the corpus.
# https://goodboychan.github.io/python/datacamp/natural_language_processing/2020/07/17/04-TF-IDF-and-similarity-scores.html
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_similarity(target, test):
    vectorizer = TfidfVectorizer().fit([target, test])
    vectors = vectorizer.transform([target, test]).toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

# Word Overlap
# Word overlap is a simple measure of similarity between two strings based on the number of common words.
# The more words two strings have in common, the more similar they are.
# https://stackoverflow.com/questions/43808541/calculate-word-overlap-from-two-words-in-two-directions-in-python

def word_overlap(target, test):
    target_words = set(target.split())
    test_words = set(test.split())
    common_words = target_words.intersection(test_words)
    return len(common_words) / len(target_words)

# Semantic Textual Similarity Models
# Semantic textual similarity (STS) measures the degree of semantic equivalence between two sentences.
# STS models are trained on large datasets to predict the similarity between sentences.
# They can provide a more nuanced measure of similarity than simple word overlap or edit distance.
# Some popular STS models include BERT, RoBERTa, and Universal Sentence Encoder.
# These models can be used to compute the similarity between sentences based on their semantic content.
# Value ranges from 0 to 5, where 5 means the sentences are semantically equivalent.
# https://pypi.org/project/semantic-text-similarity/

from semantic_text_similarity.models import WebBertSimilarity
from semantic_text_similarity.models import ClinicalBertSimilarity

web_model = WebBertSimilarity(device='cpu', batch_size=10) #defaults to GPU prediction

clinical_model = ClinicalBertSimilarity(device='cuda', batch_size=10) #defaults to GPU prediction

def semantic_similarity(model, target, test):
    similarity = model.predict([(target, test)])
    return similarity[0]

# Sequence Alignment Scores
# Sequence alignment scores are used to measure the similarity between two sequences of characters.
# These scores are commonly used in bioinformatics to compare DNA or protein sequences.
# One popular sequence alignment algorithm is the Needleman-Wunsch algorithm, which computes the optimal
# alignment between two sequences based on a scoring matrix.
# The alignment score is then used to measure the similarity between the sequences.
# The higher the alignment score, the more similar the sequences are.
# The range of the alignment score depends on the scoring matrix used.
# In this example, we use the Needleman-Wunsch algorithm to compute the alignment score between two strings.
# The alignment score is the negative of the edit distance between the strings.
# https://pypi.org/project/minineedle/#:~:text=The%20Needleman%E2%80%93Wunsch%20algorithm%20is,programming%20to%20compare%20biological%20sequences.
# from minineedle import needle, smith, core
# def sequence_alignment_score(target, test):
#     # Perform sequence alignment using the Needleman-Wunsch algorithm
#     alignment_score = 0
#     # Create the instance
#     alignment: needle.NeedlemanWunsch[str] = needle.NeedlemanWunsch(target, test)
#
#     # Make the alignment
#     alignment.align()
#
#     # Get the score
#     alignment_score = alignment.get_score()
#     return alignment_score




# Results for each test saved as table in results.xlsx
import pandas as pd
results = []
for test in tests:
    results.append({
        "Test": test,
        "Edit Distance": levenshtein(target, test),
        "Cosine Similarity of Word Embeddings": cosine_similarity_of_word_embeddings(target, test),
        "TF-IDF Similarity": tfidf_similarity(target, test),
        "Word Overlap": word_overlap(target, test),
        "Semantic Similarity (Web BERT)": semantic_similarity(web_model, target, test),
        # "Semantic Similarity (Clinical BERT)": semantic_similarity(clinical_model, target, test),
        "Sequence Alignment Score": sequence_alignment_score(target, test)
    })

df = pd.DataFrame(results)
df.to_excel("results.xlsx", index=False)

for test in tests:
    print("Target:", target)
    print("Test:", test)
    # print("Edit distance:", levenshtein(target, test))
    # print("Cosine similarity of word embeddings:", cosine_similarity_of_word_embeddings(target, test))
    # print("TF-IDF similarity:", tfidf_similarity(target, test))
    # print("Word overlap:", word_overlap(target, test))
    print("Semantic similarity (Web BERT):", semantic_similarity(web_model, target, test))
    # print("Semantic similarity (Clinical BERT):", semantic_similarity(clinical_model, target, test))
    # print("Sequence alignment score:", sequence_alignment_score(target, test))
    print()