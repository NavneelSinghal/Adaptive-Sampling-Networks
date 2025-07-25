from src.data_labelling.verifiers import GSM8KVerifier, MathVerifier, IFEvalVerifierOld
from typing import List

verifiers = {
    "gsm8k": GSM8KVerifier(),
    "MATH": MathVerifier(),
    "ifeval": IFEvalVerifierOld(),
}

def compute_unnormalized_verifiable_reward(generations: List[dict]) -> List[dict]:
    global verifiers
    for x in generations:
        verifiable_reward = None
        dataset_type = x.get('dataset')
        if dataset_type in verifiers:
            verifier = verifiers[dataset_type]
            ground_truth = x.get('ground_truth')
            if ground_truth is not None:
                try:
                    result = verifier(tokenized_prediction=None, prediction=x.get('generation', ''), label=ground_truth, query=x.get('prompt'))
                    reward = result.score
                except:
                    reward = 0
                x['unnormalized_verifiable_reward'] = reward
    return generations
