# knit_space/tests/mmlu_obfuscated_tests.py
import random
import uuid
import re
import string
import os # For standalone path if needed
from typing import Any, Dict, Iterator, List, Optional, Tuple
import logging
try:
    from datasets import load_dataset
except ImportError:
    # This is a critical dependency for this test.
    print("CRITICAL ERROR: `datasets` library not found. MMLUObfuscatedQATest cannot function.")
    print("Please install it: pip install datasets")
    # Allow the script to define the class structure but it won't generate.
    pass


# Relative import for the obfuscator
from ..obscurers.char_obfuscator import CharObfuscator
# Relative import for base classes
from .base import AbstractQATest, QAItem, register_test


@register_test('mmlu', 'obfuscated_reasoning', 'symbol_manipulation', 'instruction_following')
class MMLUObfuscatedQATest(AbstractQATest):
    # MMLU subjects (can be 'all' or specific subjects)
    # Using 'all' might be slow for loading; consider specific subjects for faster init.
    # For a robust test, 'dev' split of 'all' is good as it's smaller than train.
    # 'test' split is also an option.
    # Using 'auxiliary_train' as it often contains the dev/validation questions for 'all'
    DEFAULT_MMLU_CONFIG = "all"
    DEFAULT_MMLU_SPLIT = "auxiliary_train" # Or "test", "validation"

    # Pool of symbols for obfuscation, passed to CharObfuscator
    # Ensuring a diverse set of 48 Greek characters (24 lower, 24 upper)
    DEFAULT_SYMBOLS_POOL = "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"


    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.mmlu_config_name = self.config.get("mmlu_config", self.DEFAULT_MMLU_CONFIG)
        self.mmlu_split = self.config.get("mmlu_split", self.DEFAULT_MMLU_SPLIT)
        
        self.dataset = None
        self._load_mmlu_dataset()

        symbols_pool_override = self.config.get("obfuscation_symbols_pool")
        try:
            self.obfuscator = CharObfuscator(symbols_pool=symbols_pool_override or self.DEFAULT_SYMBOLS_POOL)
        except ValueError as e:
            self.logger.critical(f"Failed to initialize CharObfuscator: {e}. This test will not function.")
            self.obfuscator = None # Mark as unusable
        except NameError: # CharObfuscator class not found
            self.logger.critical("CharObfuscator class not found. This test will not function.")
            self.obfuscator = None


    def _load_mmlu_dataset(self):
        if 'load_dataset' not in globals():
            self.logger.critical("`datasets` library not loaded. Cannot load MMLU dataset.")
            return

        try:
            self.logger.info(f"Loading MMLU dataset: config='{self.mmlu_config_name}', split='{self.mmlu_split}'...")
            # ds = load_dataset("cais/mmlu", self.mmlu_config_name, split=self.mmlu_split)
            # The MMLU dataset on Hugging Face Hub now suggests this structure for 'all':
            ds = load_dataset("cais/mmlu", name=self.mmlu_config_name) # Loads all subjects
            self.dataset = ds[self.mmlu_split].to_list() # Convert to list for easier random sampling
            self.logger.info(f"Loaded {len(self.dataset)} examples from MMLU '{self.mmlu_config_name}' - '{self.mmlu_split}'.")
            if not self.dataset:
                 self.logger.error("MMLU dataset is empty after loading. Check config/split.")
        except Exception as e:
            self.logger.error(f"Failed to load MMLU dataset: {e}", exc_info=True)
            self.dataset = None

    def generate(self, count: int = 1, **kwargs) -> Iterator[QAItem]:
        if not self.dataset or not self.obfuscator:
            self.logger.error(f"{self.name}: MMLU dataset not loaded or obfuscator not ready. Skipping generation.")
            return

        for i in range(count):
            # 1. Select MMLU Item
            mmlu_item = random.choice(self.dataset)
            
            original_question = mmlu_item['question']
            original_choices = mmlu_item['choices'] # This is a list of strings
            correct_choice_idx = mmlu_item['answer'] # This is an integer index (0-3)
            
            if not (isinstance(original_choices, list) and len(original_choices) == 4 and isinstance(correct_choice_idx, int) and 0 <= correct_choice_idx < 4):
                self.logger.warning(f"Skipping malformed MMLU item (idx {i}): {mmlu_item}")
                continue

            # 2. Generate Character Mapping for this item
            current_char_map = self.obfuscator.create_mapping()
            mapping_str_for_prompt = self.obfuscator.format_map_for_prompt(current_char_map)

            # 3. Obfuscate Content
            obfuscated_q_text = self.obfuscator.obfuscate_text(original_question, current_char_map)
            
            obfuscated_options_formatted = []
            option_labels = ['A', 'B', 'C', 'D']
            for idx, opt_text in enumerate(original_choices):
                obfuscated_opt_text = self.obfuscator.obfuscate_text(opt_text, current_char_map)
                obfuscated_options_formatted.append(f"{option_labels[idx]}) {obfuscated_opt_text}")
            
            obfuscated_options_str = "\n".join(obfuscated_options_formatted)

            # 4. Determine the target obfuscated answer
            correct_option_original_text = original_choices[correct_choice_idx]
            correct_option_original_text_lower = correct_option_original_text.lower()
            target_obfuscated_answer = self.obfuscator.obfuscate_text(correct_option_original_text_lower, current_char_map)

            # 5. Format the Full Question for LLM
            instruction_text = (
                "Instruction: The following multiple-choice question and its options have had their lowercase English letters "
                "replaced by other symbols according to the character map provided below. "
                "Uppercase letters, numbers, punctuation, and spaces in the original text remain unchanged. "
                "The option labels (A, B, C, D) also remain unchanged.\n\n"
                "Your task is to:\n"
                "1. Understand the question and options using the character map.\n"
                "2. Identify the correct option (A, B, C, or D).\n"
                "3. Take the *original English text* of that single correct option.\n"
                "4. Convert this original English text of the correct option *entirely to lowercase*.\n"
                "5. Re-encode this lowercased text using the provided character map.\n"
                "6. Provide this final re-encoded string as your answer, enclosed in <answer></answer> tags.\n"
            )

            question_text_for_llm = (
                f"{instruction_text}\n"
                f"Character Mapping Used:\n{mapping_str_for_prompt}\n\n"
                f"Obfuscated Question:\n{obfuscated_q_text}\n\n"
                f"Options:\n{obfuscated_options_str}\n\n"
                f"What is the re-encoded lowercase text of the correct option?"
            )
            
            item_id = f"{self.name}-mmlu-{mmlu_item.get('subject','unknown')}-{uuid.uuid4().hex[:6]}"

            yield QAItem(
                id=item_id,
                question=question_text_for_llm,
                answer=target_obfuscated_answer, # This is the obfuscated, lowercased text of the correct option
                skill_coefficient = 4,
                modality='text',
                metadata={
                    'mmlu_subject': mmlu_item.get('subject', 'N/A'), # MMLU 'all' might not have subject per item
                    'original_question': original_question,
                    'original_options': original_choices,
                    'original_correct_choice_index': correct_choice_idx,
                    'original_correct_option_text_lower': correct_option_original_text_lower,
                    'char_map_used': current_char_map, # For debugging/analysis
                    'output_format_instruction': "<answer>OBFUSCATED_CORRECT_OPTION_TEXT</answer>"
                },
                verification_fn=self._verify_obfuscated_text_answer
            )

    @staticmethod
    def _verify_obfuscated_text_answer(expected_obfuscated_answer: str, provided_llm_output: str, qa_item: QAItem) -> bool:
        logger = getattr(qa_item, 'logger', logging.getLogger("MMLUObfuscatedQATest.VerificationFn"))
        
        match = re.search(r"<answer>(.*?)</answer>", provided_llm_output.strip(), re.DOTALL | re.IGNORECASE)
        if not match:
            logger.warning(f"VFY {qa_item.id}: No <answer> tags found. Raw LLM: '{provided_llm_output[:150]}'")
            return False
        
        extracted_llm_answer = match.group(1).strip()
        
        is_correct = (extracted_llm_answer == expected_obfuscated_answer)
        
        log_level = logging.INFO if is_correct else logging.WARNING
        logger.log(log_level,
                   f"MMLU Obfuscated Text VFY {('PASSED' if is_correct else 'FAILED')} for {qa_item.id}. "
                   f"Exp: '{expected_obfuscated_answer}', LLM: '{extracted_llm_answer}'.")
        if not is_correct and len(expected_obfuscated_answer) == len(extracted_llm_answer):
             diff_count = sum(1 for c1, c2 in zip(expected_obfuscated_answer, extracted_llm_answer) if c1 != c2)
             logger.debug(f"Length match, but {diff_count} char difference(s).")
        elif not is_correct:
             logger.debug(f"Length mismatch. Exp len: {len(expected_obfuscated_answer)}, LLM len: {len(extracted_llm_answer)}.")

        return is_correct

if __name__ == '__main__':
    # Ensure AbstractQATest and QAItem are defined for standalone execution
    # (Your provided snippet has stubs for these)

    print("Running MMLUObfuscatedQATest standalone...")
    
    # Configure for a quick standalone test
    # NOTE: Downloading MMLU 'all' can be large (several GB).
    # For a quicker local test, you might want to use a specific, smaller MMLU subject.
    # e.g., "cais/mmlu", "high_school_mathematics"
    # However, the 'name' parameter for load_dataset with "cais/mmlu" is usually the subject.
    # 'all' means it will load data for all subjects.
    test_config = {
        "mmlu_config": "all",             # This is the 'name' parameter for load_dataset
        "mmlu_split": "dev", # 'dev' split is usually smaller and suitable for quick tests.
                                          # The 'all' config has 'dev', 'test', 'validation', 'train'
                                          # 'auxiliary_train' might be what was previously referred to as dev for 'all'.
                                          # Let's try 'dev' for the 'all' configuration.
        # "obfuscation_symbols_pool": "αβγδεζηθικλμνξοπρστυφχψω" # Example of overriding symbol pool (needs 26 unique)
    }
    
    # If 'dev' split of 'all' is too large or doesn't exist, try a specific subject's dev split
    # test_config = {
    #     "mmlu_config": "high_school_mathematics", 
    #     "mmlu_split": "test" # Or "dev" if available for specific subject
    # }

    mmlu_test_generator = MMLUObfuscatedQATest(config=test_config)
    mmlu_test_generator.logger.setLevel(logging.DEBUG)
    if hasattr(mmlu_test_generator, 'obfuscator') and mmlu_test_generator.obfuscator:
        # If obfuscator has its own logger, you could set its level too.
        pass

    num_items_to_generate = 6 # Generate a few items for testing
    print(f"\n--- Requesting count={num_items_to_generate} items ---")

    generated_items_list = []
    if mmlu_test_generator.dataset and mmlu_test_generator.obfuscator:
        for i, item in enumerate(mmlu_test_generator.generate(count=num_items_to_generate)):
            generated_items_list.append(item)
            print(f"\n--- Generated QAItem {i+1} (ID: {item.id}) ---")
            print(f"  MMLU Subject: {item.metadata.get('mmlu_subject')}")
            print(f"  Expected Obfuscated Answer (first 50 chars): {item.answer[:50]}...")
            # print(f"  Original Question (for our ref): {item.metadata['original_question'][:100]}...")
            # print(f"  Char Map (sample): {list(item.metadata['char_map_used'].items())[:5]}")
            # print(f"  Full Question to LLM (first 500 chars):\n{item.question[:500]}...")
            
            # Example of manual verification test
            if i == 0:
                simulated_correct_llm_output = f"<answer>{item.answer}</answer>"
                simulated_wrong_llm_output = f"<answer>{item.answer[:-1]}X</answer>" # one char wrong
                simulated_bad_format_output = f"I think the answer is {item.answer}"

                is_verified_correct = MMLUObfuscatedQATest._verify_obfuscated_text_answer(item.answer, simulated_correct_llm_output, item)
                is_verified_wrong = MMLUObfuscatedQATest._verify_obfuscated_text_answer(item.answer, simulated_wrong_llm_output, item)
                is_verified_bad_format = MMLUObfuscatedQATest._verify_obfuscated_text_answer(item.answer, simulated_bad_format_output, item)
                print(f"  Simulated Verification (Correct LLM Output): {is_verified_correct}")
                print(f"  Simulated Verification (Wrong LLM Output): {is_verified_wrong}")
                print(f"  Simulated Verification (Bad Format LLM Output): {is_verified_bad_format}")
    else:
        print("MMLU dataset not loaded or obfuscator not ready. No items generated.")

    if not generated_items_list:
        print(f"\nNo items generated. Check MMLU dataset loading and paths/configs. Ensure `datasets` library is installed.")
    else:
        print(f"\nSuccessfully generated {len(generated_items_list)} item(s).")