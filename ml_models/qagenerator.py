from typing import List, Dict, Tuple
from transformers import T5TokenizerFast, T5ForConditionalGeneration
import string
from typing import List

# Constants
MODEL_NAME = 't5-small'
SOURCE_MAX_TOKEN_LEN = 300
TARGET_MAX_TOKEN_LEN = 80
SEP_TOKEN = '<sep>'
TOKENIZER_LEN = 32101


class QuestionAnswerGenerator():

    def __init__(self):
        self.tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
        self.tokenizer.add_tokens(SEP_TOKEN)
        self.tokenizer_len = len(self.tokenizer)
        self.model = T5ForConditionalGeneration.from_pretrained("fahmiaziz/QAModel")

    def generate(self, answer: str, context: str) -> str:

        model_output = self._model_predict(answer, context)
        generated_answer, generated_question = model_output.split(SEP_TOKEN)
        return generated_question

    def generate_qna(self, context: str) -> Tuple[str, str]:

        answer_mask = '[MASK]'
        model_output = self._model_predict(answer_mask, context)
        
        qna_pair = model_output.split(SEP_TOKEN)

        if len(qna_pair) < 2:
            generated_answer = ''
            generated_question = qna_pair[0]
        else:
            generated_answer = qna_pair[0]
            generated_question = qna_pair[1]

        return generated_answer, generated_question

    def _model_predict(self, answer: str, context: str) -> str:
        source_encoding = self.tokenizer(
            '{} {} {}'.format(answer, SEP_TOKEN, context),
            max_length=SOURCE_MAX_TOKEN_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        generated_ids = self.model.generate(
            input_ids=source_encoding['input_ids'],
            attention_mask=source_encoding['attention_mask'],
            num_beams=16,
            max_length=TARGET_MAX_TOKEN_LEN,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True
        )

        preds = {
            self.tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for generated_id in generated_ids
        }

        return ''.join(preds)
