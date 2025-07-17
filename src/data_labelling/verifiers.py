# taken from the implementation in https://github.com/allenai/open-instruct

import ast
import json
import logging
import re
import string
import signal
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional, Union

import sympy
from sympy.parsing.latex import parse_latex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
eval_logger = logging.getLogger("math_utils")

def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def get_unnormalized_answer(text: str) -> str:
    INVALID_ANSWER = "[invalidanswer]"
    end_seq = "I hope it is correct."
    text += end_seq
    match = re.search(r"Final Answer: The final answer is(.*?). I hope it is correct.", text)
    if match:
        return match.group(1).strip()
    else:
        return INVALID_ANSWER


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_equiv(x1: str, x2: str) -> bool:
    """
    x1 and x2 are normalized latex string
    """
    try:
        with timeout(seconds=5):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (sympy.parsing.latex.errors.LaTeXParsingError, sympy.SympifyError, TypeError):
                eval_logger.debug(f"couldn't parse one of {x1} or {x2}")
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                eval_logger.debug(f"couldn't subtract {x1} and {x2}")
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                eval_logger.debug(f"Had some trouble simplifying when comparing {x1} and {x2}")
    except TimeoutError:
        eval_logger.debug(f"Timed out comparing {x1} and {x2}")
        return False
    except ImportError as e:
        eval_logger.error(e)
        raise
    except Exception as e:
        eval_logger.debug(f"Failed comparing {x1} and {x2} with {e}")
        return False


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)
    return string


def hendrycks_is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2

def verify_keywords(text, keyword_list):
    """
    Verify if the response contains all the specified keywords.

    Args:
        response (str): The response text to check
        keyword_list (list): A list of keywords to check for

    Returns:
        bool: True if all keywords are present in the response, False otherwise
    """
    response_lower = text.lower()
    return all(keyword.lower() in response_lower for keyword in keyword_list)


def verify_keyword_frequency(text, word, N):
    """
    Verifies if a keyword appears exactly N times in the given text.

    Args:
        text (str): The text to analyze
        keyword (str): The keyword to count
        expected_count (int): The expected number of occurrences

    Returns:
        tuple: (bool, int) - (Whether constraint is met, actual count found)
    """
    text = text.lower()
    keyword = word.lower()
    import re
    words = re.findall(r"\b\w+\b", text)
    actual_count = sum(1 for word in words if word == keyword)
    constraint_met = actual_count == N
    return constraint_met


def validate_forbidden_words(text, forbidden_words):
    """
    Validates that the text does not contain any of the specified forbidden words.

    Args:
        text (str): The text to check for forbidden words
        forbidden_words (list[str]): A list of forbidden words

    Returns:
        tuple[bool, list[str]]: A tuple containing:
            - Boolean indicating if any forbidden words are present
            - List of forbidden words found in the text

    Example:
        text = "This is a message that should not contain any bad words"
        forbidden_words = ["bad", "evil", "harmful"]
        result = validate_forbidden_words(text, forbidden_words)
    """
    text_lower = text.lower()
    found_words = [word for word in forbidden_words if word.lower() in text_lower]
    return len(found_words) == 0


def verify_letter_frequency(text: str, letter: str, N: int) -> bool:
    """
    Verifies if a given letter appears exactly the specified number of times in the text.

    Args:
        text (str): The text to check
        letter (str): The letter to count (case-sensitive)
        target_count (int): The expected number of occurrences

    Returns:
        bool: True if the constraint is met, False otherwise

    Example:
        >>> verify_letter_frequency("hello world", "l", 3)
        True
        >>> verify_letter_frequency("hello world", "o", 2)
        True
        >>> verify_letter_frequency("hello world", "x", 0)
        True
    """
    if len(letter) != 1:
        raise ValueError("Letter parameter must be a single character")
    actual_count = text.count(letter)
    return actual_count == N


def validate_response_language(text, language):
    """
    Validates that the entire response is in the specified language.

    Args:
        text (str): The text to check
        language (str): The language code (e.g., 'en' for English)

    Returns:
        bool: True if the response is entirely in the specified language, False otherwise

    Example:
        text = "This is an English sentence"
        language = "en"
        result = validate_response_language(text, language)
    """
    from langdetect import detect
    detected_language = detect(text)
    return detected_language == language


def verify_paragraph_count(text: str, N: int) -> bool:
    """
    Verifies that a text contains the expected number of paragraphs,
    where paragraphs are separated by markdown dividers '* * *'

    Args:
        text (str): The text to analyze
        expected_count (int): Expected number of paragraphs

    Returns:
        bool: True if the text contains exactly the expected number of paragraphs,
              False otherwise

    Example:
         text = "First paragraph\n* * *\nSecond paragraph"
         verify_paragraph_count(text, 2)
        True
    """

    def clean_text(text: str) -> str:
        """Remove extra whitespace and normalize line endings"""
        return "\n".join(line.strip() for line in text.splitlines()).strip()
    text = clean_text(text)
    paragraphs = text.split("* * *")
    actual_count = len(paragraphs)
    valid_paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if len(valid_paragraphs) != actual_count:
        return False

    return actual_count == N


def validate_word_constraint(text: str, N: int, quantifier: str) -> bool:
    """
    Validates if a text meets specified word count constraints.

    Args:
        text (str): The text to check
        count (int): The target word count
        qualifier (str): The type of constraint ('at least', 'around', 'at most')

    Returns:
        bool: True if the constraint is met, False otherwise

    Raises:
        ValueError: If an invalid qualifier is provided
    """
    words = text.strip().split()
    actual_count = len(words)
    tolerance = max(round(N * 0.1), 1)
    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "at most":
        return actual_count <= N
    elif quantifier == "around":
        return abs(actual_count - N) <= tolerance
    else:
        return False


def verify_sentence_constraint(text: str, N: int, quantifier: str) -> bool:
    """
    Verifies if a text contains the expected number of sentences.

    Args:
        text (str): The text to analyze
        N (int): The expected number of sentences
        quantifier (str): The quantifier ('at least', 'around', 'at most')

    Returns:
        bool: True if the text contains the expected number of sentences, False otherwise
    """
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)
    actual_count = len(sentences)
    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "around":
        return abs(actual_count - N) <= 1
    elif quantifier == "at most":
        return actual_count <= N
    else:
        return False


def validate_paragraphs(text, N, first_word, i):
    """
    Validates that a text contains the expected number of paragraphs and that the i-th paragraph starts with a specific
    word.

    Args:
        text (str): The text to analyze
        N (int): The expected number of paragraphs
        first_word (str): The expected first word of the i-th paragraph
        i (int): The index of the paragraph to check (1-indexed)

    Returns:
        bool: True if the text meets the paragraph and first word requirements, False otherwise
    """
    paragraphs = text.split("\n\n")
    if len(paragraphs) != N:
        return False
    if paragraphs[i - 1].strip().startswith(first_word):
        return True
    return False


def verify_postscript(text, postscript_marker):
    """
    Verifies if a text contains a postscript starting with '{postscript marker}'

    Args:
        text (str): The text to verify

    Returns:
        bool: True if the text contains a valid postscript, False otherwise
    """
    if postscript_marker in text:
        marker_index = text.find(postscript_marker)
        remaining_text = text[marker_index:].strip()
        return len(remaining_text) > len(postscript_marker)
    return False


def validate_placeholders(text: str, N: int) -> tuple[bool, List[str]]:
    """
    Validates if a text contains at least the specified number of placeholders in square brackets.

    Args:
        text (str): The text to check for placeholders
        min_placeholders (int): Minimum number of placeholders required

    Returns:
        tuple[bool, List[str]]: A tuple containing:
            - Boolean indicating if the text meets the placeholder requirement
            - List of found placeholders

    Example:
        >>> text = "Hello [name], your [item] will be delivered to [address]"
        >>> validate_placeholders(text, 2)
        (True, ['name', 'item', 'address'])
    """
    pattern = r"\[(.*?)\]"
    placeholders = re.findall(pattern, text)
    has_enough = len(placeholders) >= N
    return has_enough


def verify_bullet_points(text: str, N: int) -> tuple[bool, str]:
    """
    Verifies if a text contains exactly N bullet points in markdown format.
    Returns a tuple of (is_valid, message).

    Args:
        text (str): The text to check
        expected_count (int): The expected number of bullet points

    Returns:
        tuple[bool, str]: (True if constraint is met, explanation message)
    """
    lines = text.split("\n")
    bullet_points = [line.strip() for line in lines if line.strip().startswith(("*", "-"))]
    actual_count = len(bullet_points)
    if actual_count == N:
        return True
    else:
        return False


def validate_title(text: str) -> bool:
    pattern = r"<<(.*?)>>"
    matches = re.findall(pattern, text)
    if len(matches) > 0:
        return True
    else:
        return False


def validate_choice(text: str, options: list) -> bool:
    for option in options:
        if text in option:
            return True
    return False


def validate_highlighted_sections(text: str, N: int) -> bool:
    pattern = r"\*(.*?)\*"
    matches = re.findall(pattern, text)
    if len(matches) >= N:
        return True
    else:
        return False


def validate_sections(text: str, N: int, section_splitter: str) -> bool:
    sections = text.split(section_splitter)
    if sections[0] == "":
        sections.pop(0)
    if len(sections) == N:
        return True
    else:
        return False

def validate_json_format(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True

def validate_repeat_prompt(text: str, original_prompt: str) -> bool:
    if text.startswith(original_prompt):
        return True
    else:
        return False

def validate_two_responses(text: str) -> bool:
    if text.count("******") == 1:
        response_list = text.split("******")
        first_response = response_list[0].strip()
        second_response = response_list[1].strip()
        if first_response != second_response:
            return True
    return False


def validate_uppercase(text: str) -> bool:
    if text == text.upper():
        return True
    else:
        return False


def validate_lowercase(text: str) -> bool:
    if text == text.lower():
        return True
    else:
        return False


def validate_frequency_capital_words(text: str, N: int, quantifier: str) -> bool:
    words = re.findall(r"\b[A-Z]+\b", text)
    if quantifier == "at least":
        return len(words) >= N
    elif quantifier == "around":
        return len(words) == N
    elif quantifier == "at most":
        return len(words) <= N
    else:
        return False


def validate_end(text: str, end_phrase: str) -> bool:
    if text.endswith(end_phrase):
        return True
    else:
        return False


def validate_quotation(text: str) -> bool:
    if text.startswith('"') and text.endswith('"'):
        return True
    else:
        return False


def validate_no_commas(text: str) -> bool:
    if "," not in text:
        return True
    else:
        return False


IF_FUNCTIONS_MAP = {
    "verify_keywords": verify_keywords,
    "verify_keyword_frequency": verify_keyword_frequency,
    "validate_forbidden_words": validate_forbidden_words,
    "verify_letter_frequency": verify_letter_frequency,
    "validate_response_language": validate_response_language,
    "verify_paragraph_count": verify_paragraph_count,
    "validate_word_constraint": validate_word_constraint,
    "verify_sentence_constraint": verify_sentence_constraint,
    "validate_paragraphs": validate_paragraphs,
    "verify_postscript": verify_postscript,
    "validate_placeholders": validate_placeholders,
    "verify_bullet_points": verify_bullet_points,
    "validate_title": validate_title,
    "validate_choice": validate_choice,
    "validate_highlighted_sections": validate_highlighted_sections,
    "validate_sections": validate_sections,
    "validate_json_format": validate_json_format,
    "validate_repeat_prompt": validate_repeat_prompt,
    "validate_two_responses": validate_two_responses,
    "validate_uppercase": validate_uppercase,
    "validate_lowercase": validate_lowercase,
    "validate_frequency_capital_words": validate_frequency_capital_words,
    "validate_end": validate_end,
    "validate_quotation": validate_quotation,
    "validate_no_commas": validate_no_commas,
}

def extract_final_answer(prediction: str) -> str:
    """
    Extract the substring between <answer> and </answer>.
    If no match is found, extract the substring after </think>.
    If neither condition matches, clean the prediction by removing the <|assistant|> tag.
    If none of the above applies, return the original string.

    Args:
        prediction (str): The input string.

    Returns:
        str: The extracted substring or the cleaned/original string.
    """
    answer_match = re.search(r"<answer>(.*?)</answer>", prediction, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    think_match = re.search(r"</think>(.*)", prediction, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    cleaned = re.sub(r"<\|assistant\|>", "", prediction)
    if cleaned != prediction:
        return cleaned.strip()
    return prediction

class VerificationResult:
    score: float
    cost: float = 0.0
    reasoning: Optional[str] = None

    def __init__(self, score, cost=0.0, reasoning=None):
        self.score = score
        self.cost = cost
        self.reasoning = reasoning

class VerifierFunction(ABC):
    def __init__(self, name: str, weight: float = 1.0, verifier_config: Optional[Any] = None) -> None:
        self.name = name
        self.weight = weight
        self.verifier_config = verifier_config

    @abstractmethod
    def __call__(
        self, tokenized_prediction: List[int], prediction: str, label: Any, query: Optional[str] = None
    ) -> VerificationResult:
        raise NotImplementedError

def remove_thinking_section(prediction: str) -> str:
    prediction = prediction.replace("<|assistant|>", "").strip()
    prediction = prediction.split("</think>")[-1]
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")
    return prediction.strip()

class GSM8KVerifier(VerifierFunction):
    def __init__(self, verifier_config: Optional[Any] = None) -> None:
        super().__init__("gsm8k", verifier_config=verifier_config, weight=1.0)

    def __call__(self, tokenized_prediction: List[int], prediction: str, label: str, query: Optional[str] = None) -> VerificationResult:
        response = re.sub(r"(\d),(\d)", r"\1\2", prediction)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response)
        extracted = numbers[-1] if numbers else response
        score = float(str(extracted).lower() == str(label).lower())
        return VerificationResult(score=score)

class MathVerifier(VerifierFunction):
    def __init__(self, verifier_config: Optional[Any] = None) -> None:
        super().__init__("math", verifier_config=verifier_config, weight=1.0)

    def __call__(self, tokenized_prediction: List[int], prediction: str, label: str, query: Optional[str] = None) -> VerificationResult:
        raw_answer = prediction
        all_answers = []
        boxed_answer = last_boxed_only_string(raw_answer)
        if boxed_answer is not None:
            try:
                boxed_answer = remove_boxed(boxed_answer)
            except AssertionError:
                boxed_answer = None
        if boxed_answer is not None:
            all_answers.append(boxed_answer)
        minerva_answer = normalize_final_answer(get_unnormalized_answer(raw_answer))
        if minerva_answer is not None and minerva_answer != "[invalidanswer]":
            all_answers.append(minerva_answer)
        if not all_answers:
            dollars = [m.start() for m in re.finditer(r"\$", raw_answer)]
            if len(dollars) > 1:
                answer = normalize_final_answer(raw_answer[dollars[-2] + 1 : dollars[-1]])
                all_answers.append(answer)
        if not all_answers:
            all_answers.append(normalize_final_answer(prediction))
            all_answers.append(prediction)
        for answer in all_answers:
            if is_equiv(answer, label) or hendrycks_is_equiv(answer, label):
                return VerificationResult(score=1.0)
        return VerificationResult(score=0.0)

class IFEvalVerifierOld(VerifierFunction):
    def __init__(self, verifier_config: Optional[Any] = None) -> None:
        super().__init__("ifeval_old", verifier_config=verifier_config, weight=1.0)

    def __call__(self, tokenized_prediction: List[int], prediction: str, label: Union[str, Dict], query: Optional[str] = None) -> VerificationResult:
        constraint = label
        answer = remove_thinking_section(prediction)
        if isinstance(constraint, str):
            try:
                constraint = json.loads(constraint)
            except json.JSONDecodeError:
                logging.warning(f"Could not parse IFEval ground_truth JSON: {constraint}")
                return VerificationResult(score=0.0)

        if "func_name" not in constraint:
            logging.warning(f"Constraint missing 'func_name': {constraint}")
            return VerificationResult(score=0.0)

        func_name = constraint.pop("func_name")
        if func_name not in IF_FUNCTIONS_MAP:
            logging.warning(f"Function '{func_name}' not in IF_FUNCTIONS_MAP.")
            return VerificationResult(score=0.0)

        func = IF_FUNCTIONS_MAP[func_name]
        non_none_args = {k: v for k, v in constraint.items() if v is not None}
        try:
            if not constraint:
                score = float(func(prediction))
            else:
                score = float(func(answer, **non_none_args))
        except Exception as e:
            logging.error(f"Error executing IFEval function '{func_name}': {e}")
            score = 0.0
        return VerificationResult(score=score)
