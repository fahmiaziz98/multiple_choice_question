import re
import string
from typing import List


def normalize_item(item) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(item))))


def remove_duplicates(items: List[str]) -> List[str]:
    unique_items = []
    normalized_unique_items = []

    for item in items:
        normalized_item = normalize_item(item)

        if normalized_item not in normalized_unique_items:
            unique_items.append(item)
            normalized_unique_items.append(normalized_item)

    return unique_items
    
def remove_distractors_duplicate_with_correct_answer(correct: str, distractors: List[str]) -> List[str]:
    normalized_correct = normalize_item(correct)

    filtered_distractors = []

    for distractor in distractors:
        if normalize_item(distractor) != normalized_correct:
            filtered_distractors.append(distractor)

    return filtered_distractors

def clean_text(text: str) -> str:
    # remove brackets
    cleaned_text = re.sub(r"\((.*?)\)", lambda L: "", text)
    # remove square bracket
    cleaned_text = re.sub(r"\[(.*?)\]", lambda L: "", cleaned_text)
    # remove multiple space
    cleaned_text = re.sub(" +", " ", cleaned_text)
    # replace weird hypen
    cleaned_text = cleaned_text.replace('–', '-')

    return cleaned_text
