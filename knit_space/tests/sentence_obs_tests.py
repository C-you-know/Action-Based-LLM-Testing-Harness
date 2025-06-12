# knit_space/tests/sentence_obs_test.py

from ..obscurers.bijective_char_mapper import BijectiveCharacterMapper 
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union, Callable
import re

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


from .base import AbstractQATest, QAItem, register_test 

import random
import string
import uuid
from typing import Iterator, Optional, Dict, Any

@register_test('text_processing', 'obfuscation', 'decoding')
class RandomSentenceObfuscationTest(AbstractQATest):
    """
    Generates a QA item involving decoding an obfuscated sentence.
    A random sentence is generated, encoded using BijectiveCharacterMapper,
    and the task is to decode it given the character map.
    """

    DEFAULT_MAP_FILENAME = "sentence_obfuscation_char_map.txt"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        map_file = self.config.get("char_map_file", self.DEFAULT_MAP_FILENAME)
        self.char_mapper = BijectiveCharacterMapper(mapping_file_path=map_file)
        self.min_words = self.config.get("min_words", 10)
        self.max_words = self.config.get("max_words", 25)
        self.min_word_len = self.config.get("min_word_len", 3)
        self.max_word_len = self.config.get("max_word_len", 10)


    @property
    def supported_modalities(self) -> List[str]:
        return ['text']

    def _generate_random_word(self) -> str:
        """Generates a random word made of lowercase English letters."""
        length = random.randint(self.min_word_len, self.max_word_len)
        return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))

    def _generate_random_sentence(self) -> str:
        """Generates a sentence with a random number of random words."""
        num_words = random.randint(self.min_words, self.max_words)
        words = [self._generate_random_word() for _ in range(num_words)]
        return " ".join(words)

    def _format_mapping_table_for_prompt(self, mapping_table: Dict[str, str]) -> str:
        """Formats the character mapping table into a readable string for the question."""
        relevant_chars = sorted([char for char in mapping_table.keys() if char in string.ascii_lowercase])

        formatted_mappings = ["Character Mappings:"]
        for char in relevant_chars:
            if char in self.char_mapper.SOURCE_CHARS:
                 formatted_mappings.append(f"  '{char}' maps to '{mapping_table[char]}'")
        return "\n".join(formatted_mappings)

    @staticmethod
    def _verify_decoded_sentence(expected_sentence: str, provided_llm_output: str, self) -> bool:
        if not isinstance(provided_llm_output, str):
            return False

        content_to_search = provided_llm_output.strip()

        code_block_match = re.search(r"```(?:[a-zA-Z0-9]*\n)?(.*?)```", content_to_search, re.DOTALL)
        if code_block_match:
            content_to_search = code_block_match.group(1).strip()

        answer_match = re.search(r"<answer>(.*?)</answer>", content_to_search, re.IGNORECASE | re.DOTALL)

        if not answer_match:
            return False

        extracted_llm_answer_raw = answer_match.group(1).strip()
        
        processed_llm_answer = re.sub(r"\s+", "", extracted_llm_answer_raw)
        processed_expected_sentence = re.sub(r"\s+", "", expected_sentence)

        return processed_expected_sentence == processed_llm_answer

    def generate(self,
                 count: int = 4,
                 difficulty: Optional[str] = None,
                 prefix: Optional[str] = None,
                 suffix: Optional[str] = None,
                 **kwargs) -> Iterator[QAItem]:

        for i in range(count):
            original_sentence = self._generate_random_sentence()
            encoded_sentence = self.char_mapper.encode(original_sentence)
            
            mapping_table_str = self._format_mapping_table_for_prompt(self.char_mapper.get_mapping_table())

            question_base = (
                f"Given the following character mappings from english alphabets to a secret language. The encoding-decoding mappings are given below. Decode the sentence back to english. %%TOK%% means next letter. %%TOK%% %%TOK%% meanins white space. Give the entire translations in <answer> </answer>.\n\n"
                f"{mapping_table_str}\n\n"
                f"Encoded Sentence: {encoded_sentence}"
            )
            
            final_question_str = self.build_question(
                base_question=question_base,
                prefix=prefix,
                suffix=suffix
            )
            expected_answer = original_sentence

            yield QAItem(
                id=f"{self.name}-{uuid.uuid4().hex[:8]}",
                question=final_question_str,
                answer=expected_answer,
                skill_coefficient = 1,
                modality='text',
                metadata={
                    'difficulty': difficulty or 'medium',
                    'sentence_word_count': len(original_sentence.split()),
                    'mapper_file': self.char_mapper.mapping_file_path
                },
                verification_fn=self._verify_decoded_sentence
            )
