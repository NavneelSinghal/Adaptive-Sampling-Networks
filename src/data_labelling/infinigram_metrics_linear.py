import requests
import time
from typing import List

def simple_tokenizer(text: str) -> List[str]:
    """
    A simple space-based tokenizer.
    NOTE: The Infini-gram API uses its own specific tokenizer (e.g., Llama-2).
    This local tokenizer is just for splitting the text into chunks to query.
    The API itself handles the true tokenization of the query string.
    """
    return text.split()

class InfiniGramApiLabeller:
    """
    Labels text by querying the Infini-gram API to find n-grams
    that exist in a large reference corpus.
    """
    API_URL = "https://api.infini-gram.io/"

    def __init__(self, index_name: str, max_n: int = 10):
        """
        Initializes the labeller.

        Args:
            index_name: The name of the Infini-gram index to use (e.g., 'v4_rpj_llama_s4').
            max_n: The maximum n-gram length to check.
        """
        self.index_name = index_name
        self.max_n = max_n
        self.session = requests.Session()

    def _check_ngram_exists(self, ngram: List[str]) -> bool:
        """Queries the API to see if an n-gram exists in the corpus."""
        if not ngram:
            return False
        payload = {
            'index': self.index_name,
            'query_type': 'count',
            'query': ' '.join(ngram),
        }
        #print(payload)
        # Retry logic. FIXME
        for _ in range(3):
            try:
                #print('REQUESTING')
                response = self.session.post(self.API_URL, json=payload, timeout=10)
                response.raise_for_status()
                result = response.json()
                if "error" in result:
                    return False
                return result.get("count", 0) > 0
            except requests.exceptions.RequestException as e:
                #print(f'WARNING: GOT EXCEPTION {e}')
                time.sleep(1)
        #print('WARNING: UNABLE TO REQUEST')
        return False

    def label(self, generated_text: str) -> dict:
        """
        Labels the generated text to calculate total overlap and the number of breaks.

        Args:
            generated_text: The text to label.

        Returns:
            A dictionary with 'infinigram_total_coverage' and 'infinigram_breaks' scores.
        """
        tokens = simple_tokenizer(generated_text)
        num_tokens = len(tokens)
        total_overlap = 0
        breaks = 0
        i = 0
        was_in_match = False
        while i < num_tokens:
            #print('ENTERING LOOP')
            # Find the longest matching n-gram starting at position i
            # TODO: Check whether this or a simple left-to-right search amortizes better.
            longest_match_len = 0
            for n in range(self.max_n, 0, -1):
                if i + n > num_tokens:
                    continue
                ngram_to_check = tokens[i : i + n]
                if self._check_ngram_exists(ngram_to_check):
                    longest_match_len = n
                    break
            if longest_match_len > 0: # TODO: Can also look at distribution of this value.
                total_overlap += longest_match_len
                i += longest_match_len
                was_in_match = True
            else:
                if was_in_match:
                    breaks += 1
                i += 1
                was_in_match = False
        return {
            "infinigram_total_coverage": total_overlap,
            "infinigram_breaks": breaks,
        }
