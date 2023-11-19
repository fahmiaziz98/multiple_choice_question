from transformers import T5TokenizerFast, T5ForConditionalGeneration
import string
from typing import List


SOURCE_MAX_TOKEN_LEN = 512
TARGET_MAX_TOKEN_LEN = 50
SEP_TOKEN = "[SEP]"  
MODEL_NAME = "t5-small"  

# Definisi kelas DistractorGenerator
class DistractorGenerator:
    def __init__(self):
        self.tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
        self.tokenizer.add_tokens(SEP_TOKEN)
        self.tokenizer_len = len(self.tokenizer)
        self.model = T5ForConditionalGeneration.from_pretrained("fahmiaziz/QDModel")

    def generate(self, generate_count: int, correct: str, question: str, context: str) -> List[str]:
        model_output = self._model_predict(generate_count, correct, question, context)

        cleaned_result = model_output.replace('<pad>', '').replace('</s>', ',')
        cleaned_result = self._replace_all_extra_id(cleaned_result)
        distractors = cleaned_result.split(",")[:-1]
        distractors = [x.translate(str.maketrans('', '', string.punctuation)) for x in distractors]
        distractors = list(map(lambda x: x.strip(), distractors))

        return distractors

    def _model_predict(self, generate_count: int, correct: str, question: str, context: str) -> str:
        source_encoding = self.tokenizer(
            '{} {} {} {} {}'.format(correct, SEP_TOKEN, question, SEP_TOKEN, context),
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
            num_beams=generate_count,
            num_return_sequences=generate_count,
            max_length=TARGET_MAX_TOKEN_LEN,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True
        )

        preds = {
            self.tokenizer.decode(generated_id, skip_special_tokens=False, clean_up_tokenization_spaces=True)
            for generated_id in generated_ids
        }

        return ''.join(preds)

    def _correct_index_of(self, text: str, substring: str, start_index: int = 0):
        try:
            index = text.index(substring, start_index)
        except ValueError:
            index = -1

        return index

    def _replace_all_extra_id(self, text: str):
        new_text = text
        start_index_of_extra_id = 0

        while (self._correct_index_of(new_text, '<extra_id_') >= 0):
            start_index_of_extra_id = self._correct_index_of(new_text, '<extra_id_', start_index_of_extra_id)
            end_index_of_extra_id = self._correct_index_of(new_text, '>', start_index_of_extra_id)

            new_text = new_text[:start_index_of_extra_id] + '[SEP]' + new_text[end_index_of_extra_id + 1:]

        return new_text
