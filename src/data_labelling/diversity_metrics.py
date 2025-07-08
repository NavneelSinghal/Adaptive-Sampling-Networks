import numpy as np
from sacrebleu.metrics import BLEU
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

def calculate_self_bleu(generations: List[str]) -> float:
    """
    Calculates the Self-BLEU score for a set of generations.
    A lower score indicates higher diversity.

    Args:
        generations: A list of generated sentences.

    Returns:
        The average BLEU score, indicating inter-text similarity.
    """
    if len(generations) < 2:
        return 0.0
    total_bleu_score = 0.0
    bleu = BLEU(effective_order=True)
    for i in range(len(generations)):
        hypothesis = generations[i]
        references = generations[:i] + generations[i+1:]
        formatted_references = [references]
        score = bleu.sentence_score(hypothesis, formatted_references)
        total_bleu_score += score.score
    return total_bleu_score / len(generations)

def calculate_embedding_entropy(
    generations: List[str], 
    model: SentenceTransformer
) -> float:
    """
    Calculates the Von Neumann entropy of the similarity matrix of sentence embeddings.
    A higher score indicates higher diversity.

    Args:
        generations: A list of generated sentences.
        model: A pre-loaded SentenceTransformer model.

    Returns:
        The Von Neumann entropy score.
    """
    if not generations:
        return 0.0

    embeddings = model.encode(generations, convert_to_numpy=True)
    similarity_matrix = cosine_similarity(embeddings)
    eigenvalues = np.linalg.eigvalsh(similarity_matrix)
    non_zero_eigenvalues = eigenvalues[eigenvalues > 1e-9] 
    entropy = -np.sum(non_zero_eigenvalues * np.log2(non_zero_eigenvalues))
    return entropy
