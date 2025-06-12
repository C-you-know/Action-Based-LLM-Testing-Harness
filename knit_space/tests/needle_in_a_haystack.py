import random
import uuid
import re
from typing import Any, Dict, Iterator, List, Optional

from .base import AbstractQATest, QAItem, register_test

@register_test('search', 'long_context', 'attention', 'numerical_search')
class FindUniqueNumberIndexQATest(AbstractQATest):
    DEFAULT_ARRAY_LENGTHS = [1000, 5000, 10000]
    DEFAULT_NUMBER_MIN_VAL = -100000
    DEFAULT_NUMBER_MAX_VAL = 100000

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.array_lengths_options = self.config.get(
            "array_lengths", self.DEFAULT_ARRAY_LENGTHS
        )
        self.number_min_val = self.config.get(
            "number_min_value", self.DEFAULT_NUMBER_MIN_VAL
        )
        self.number_max_val = self.config.get(
            "number_max_value", self.DEFAULT_NUMBER_MAX_VAL
        )

        if not isinstance(self.array_lengths_options, list) or not self.array_lengths_options:
            self.logger.warning(
                f"Invalid array_lengths_options: {self.array_lengths_options}. "
                f"Defaulting to {self.DEFAULT_ARRAY_LENGTHS}."
            )
            self.array_lengths_options = self.DEFAULT_ARRAY_LENGTHS
        if not isinstance(self.number_min_val, int) or not isinstance(self.number_max_val, int) or self.number_min_val >= self.number_max_val:
            self.logger.warning(
                f"Invalid number range ({self.number_min_val} - {self.number_max_val}). "
                f"Defaulting to {self.DEFAULT_NUMBER_MIN_VAL} - {self.DEFAULT_NUMBER_MAX_VAL}."
            )
            self.number_min_val = self.DEFAULT_NUMBER_MIN_VAL
            self.number_max_val = self.DEFAULT_NUMBER_MAX_VAL

    def generate(self, count: int = 4, **kwargs) -> Iterator[QAItem]:
        for _ in range(count):

            array_length = random.choice(self.array_lengths_options)

            target_number_k = random.randint(self.number_min_val, self.number_max_val)

            k_index = random.randint(0, array_length - 1)

            number_array = [0] * array_length 
            number_array[k_index] = target_number_k

            for i in range(array_length):
                if i == k_index:
                    continue 

                random_num = random.randint(self.number_min_val, self.number_max_val)
                while random_num == target_number_k:
                    random_num = random.randint(self.number_min_val, self.number_max_val)
                number_array[i] = random_num

            array_string_representation = ", ".join(map(str, number_array))
            if len(array_string_representation) > 15000 and array_length > 1000: 

                preview_array_str = (", ".join(map(str, number_array[:20])) +
                                     f"... (total {array_length} numbers) ..." +
                                     ", ".join(map(str, number_array[-20:])))
                self.logger.debug(f"Generated a long array of {array_length} numbers. Question will use full list. Preview: {preview_array_str}")

            question_text = (
                f"You are given a list of {array_length} integers. "
                f"The target number to find is: {target_number_k}\n"
                f"This target number appears exactly once in the list.\n\n"
                f"The list of numbers is:\n[{array_string_representation}]\n\n"
                f"What is the 0-based index of the number {target_number_k} in the list?\n"
                "Provide your answer as a single integer within <answer></answer> tags. For example: <answer>42</answer>"
            )

            item_id = f"{self.name}-{uuid.uuid4().hex[:8]}"

            yield QAItem(
                id=item_id,
                question=question_text,
                answer=k_index, 
                skill_coefficient = 4,
                modality='text',
                metadata={
                    'array_length': array_length,
                    'number_range_min': self.number_min_val,
                    'number_range_max': self.number_max_val,
                    'target_number_K': target_number_k,
                    'correct_index': k_index, 
                    'output_format_instruction': "<answer>INDEX</answer>"
                },
                verification_fn=self._verify_index_answer
            )

    @staticmethod
    def _verify_index_answer(expected_index: int, provided_answer_str: str, qa_item: QAItem) -> bool:

        match = re.fullmatch(r'<answer>(\d+)</answer>', provided_answer_str.strip())
        if not match:

            return False 

        try:
            provided_index_int = int(match.group(1))
        except ValueError:

            return False 

        is_correct = (provided_index_int == expected_index)

        return is_correct