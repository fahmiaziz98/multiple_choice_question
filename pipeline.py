from typing import List
from nltk.tokenize import sent_tokenize
import toolz

from models.models import Question
from utils.cleaned_text import clean_text, remove_duplicates, remove_distractors_duplicate_with_correct_answer
from ml_models.distractor import DistractorGenerator
from ml_models.qagenerator import QuestionAnswerGenerator


class Pipeline:

    def __init__(self):

        self.question_generator = QuestionAnswerGenerator()
        self.distractor_generator = DistractorGenerator()
        
    #  <======================= Main Function =============================>
    def generate_mcqs(self, context: str, desired_count: int) -> List[Question]:

        cleaned_text =  clean_text(context)
        questions = self._generate_question_answer_pairs(cleaned_text, desired_count)
        questions = self._generate_distractors(cleaned_text, questions)

        return questions
    # <====================================================>


    # number: 1
    def _generate_question_answer_pairs(self, context: str, desired_count: int) -> List[Question]:
        context_splits = self._split_context_according_to_desired_count(context, desired_count)

        questions = []

        for split in context_splits:
            answer, question = self.question_generator.generate_qna(split)
            questions.append(Question(answer.capitalize(), question))

        questions = list(toolz.unique(questions, key=lambda x: x.answerText))

        return questions

    # number: 2
    def _generate_distractors(self, context: str, questions: List[Question]) -> List[Question]:
        for question in questions:
            t5_distractors =  self.distractor_generator.generate(5, question.answerText, question.questionText, context)

            distractors = remove_duplicates(t5_distractors)
            distractors = remove_distractors_duplicate_with_correct_answer(question.answerText, distractors)

            #TODO - filter distractors having a similar bleu score with another distractor
            # filter_distractors = []
            # for dist in distractors:
            #     bleu_score = self._calculate_nltk_bleu([dist], question.answerText)
            #     if bleu_score > 0.1:
            #         filter_distractors.append(dist)
            # <=================Need Improve Model=================>

            question.distractors = distractors
        return questions

    # Helper functions
    def _split_context_according_to_desired_count(self, context: str, desired_count: int) -> List[str]:
        sents = sent_tokenize(context)
        total_sents = len(sents)

        if total_sents <= desired_count:
            return sents  # No need to split if the desired count is greater than or equal to the total sentences.

        sentences_per_split = total_sents // desired_count
        remainder = total_sents % desired_count  # Handle the remaining sentences.

        context_splits = []
        start_sent_index = 0

        for i in range(desired_count):
            end_sent_index = start_sent_index + sentences_per_split + (1 if i < remainder else 0)
            context_split = ' '.join(sents[start_sent_index:end_sent_index])
            context_splits.append(context_split)
            start_sent_index = end_sent_index

        return context_splits
