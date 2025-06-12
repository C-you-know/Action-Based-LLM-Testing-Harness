# knit_space/tests/mmlu_obfuscated_choice_tests.py
import random
import uuid
import re
import string
import os
import logging # <-- IMPORT LOGGING HERE
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    from datasets import load_dataset
except ImportError:
    logging.critical("CRITICAL ERROR: `datasets` library not found. MMLUObfuscatedChoiceQATest cannot function.")
    logging.critical("Please install it: pip install datasets")
    # Allow the script to define the class structure but it won't generate.
    pass

# Relative import for the obfuscator (assuming it's in the same obscurers directory)
# This will work when run as part of the package.
try:
    from ..obscurers.char_obfuscator import CharObfuscator
except ImportError:
    # This path might be tried if the above fails, e.g., when running the file directly
    # and the knit_space package isn't "active" in the Python path in the right way.
    # For __main__ block, we'll handle it more explicitly.
    logging.debug("Could not import CharObfuscator via relative path, will try direct if in __main__.")
    pass

# Relative import for base classes
try:
    from .base import AbstractQATest, QAItem, register_test
except ImportError:
    logging.debug("Could not import base classes via relative path, will use stubs if in __main__.")
    pass


@register_test('mmlu', 'obfuscated_reasoning', 'symbol_manipulation', 'multiple_choice')
class MMLUObfuscatedChoiceQATest(AbstractQATest):
    DEFAULT_MMLU_CONFIG = "all"
    DEFAULT_MMLU_SPLIT = "dev"
    DEFAULT_SYMBOLS_POOL = "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
    OPTION_LABELS = ['A', 'B', 'C', 'D']

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config) # Initializes self.logger
        self.mmlu_config_name = self.config.get("mmlu_config", self.DEFAULT_MMLU_CONFIG)
        self.mmlu_split = self.config.get("mmlu_split", self.DEFAULT_MMLU_SPLIT)
        
        self.dataset = None
        self._load_mmlu_dataset()

        symbols_pool_override = self.config.get("obfuscation_symbols_pool")
        
        # Attempt to instantiate CharObfuscator
        # CharObfuscator_class = None # Initialize to None
        if 'CharObfuscator' in globals():
            CharObfuscator_class = globals()['CharObfuscator']
        else:
            # Attempt to dynamically import if not found globally (e.g. if relative import failed initially but path is now ok)
            try:
                from ..obscurers.char_obfuscator import CharObfuscator as CO_from_import
                CharObfuscator_class = CO_from_import
                logging.debug("CharObfuscator loaded dynamically in __init__.")
            except ImportError:
                self.logger.critical("CharObfuscator class could not be imported or found globally.")
                CharObfuscator_class = None

        if CharObfuscator_class:
            try:
                self.obfuscator = CharObfuscator_class(symbols_pool=symbols_pool_override or self.DEFAULT_SYMBOLS_POOL)
            except ValueError as e:
                self.logger.critical(f"Failed to initialize CharObfuscator: {e}. This test will not function.")
                self.obfuscator = None
        else:
            self.logger.critical("CharObfuscator class definition is missing. This test will not function.")
            self.obfuscator = None


    def _load_mmlu_dataset(self):
        if 'load_dataset' not in globals():
            self.logger.critical("`datasets` library not loaded. Cannot load MMLU dataset.")
            return
        try:
            self.logger.info(f"Loading MMLU dataset: config='{self.mmlu_config_name}', split='{self.mmlu_split}'...")
            ds_full = load_dataset("cais/mmlu", name=self.mmlu_config_name)
            if self.mmlu_split not in ds_full:
                self.logger.error(f"Split '{self.mmlu_split}' not found in MMLU config '{self.mmlu_config_name}'. Available: {list(ds_full.keys())}.")
                available_splits = list(ds_full.keys())
                if 'dev' in available_splits:
                    self.mmlu_split_to_use = 'dev'
                elif 'validation' in available_splits:
                    self.mmlu_split_to_use = 'validation'
                elif 'test' in available_splits:
                    self.mmlu_split_to_use = 'test'
                elif available_splits:
                    self.mmlu_split_to_use = available_splits[0]
                else:
                    self.logger.error("No splits found in the loaded dataset.")
                    self.dataset = None
                    return
                self.logger.warning(f"Using fallback MMLU split: '{self.mmlu_split_to_use}'")
                self.dataset = ds_full[self.mmlu_split_to_use].to_list()
            else:
                self.dataset = ds_full[self.mmlu_split].to_list()

            if self.dataset:
                self.logger.info(f"Loaded {len(self.dataset)} examples from MMLU '{self.mmlu_config_name}' - '{self.mmlu_split if self.mmlu_split in ds_full else self.mmlu_split_to_use}'.")
            else:
                self.logger.error("MMLU dataset is empty after loading attempts.")
        except Exception as e:
            self.logger.error(f"Failed to load MMLU dataset: {e}", exc_info=True)
            self.dataset = None

    # ... (generate method and _verify_obfuscated_choice_label method remain the same) ...
    def generate(self, count: int = 8, **kwargs) -> Iterator[QAItem]:
        if not self.dataset or not self.obfuscator:
            self.logger.error(f"{self.name}: MMLU dataset not loaded or obfuscator not ready. Skipping generation.")
            return

        for i in range(count):
            mmlu_item = random.choice(self.dataset)

            original_question = mmlu_item['question']
            original_choices_text = mmlu_item['choices'] # List of 4 strings
            correct_choice_idx = mmlu_item['answer']   # Integer index 0-3

            if not (isinstance(original_choices_text, list) and
                    len(original_choices_text) == 4 and
                    isinstance(correct_choice_idx, int) and
                    0 <= correct_choice_idx < 4):
                self.logger.warning(f"Skipping malformed MMLU item (idx {i}): Choices or answer index invalid. Item: {mmlu_item}")
                continue

            current_char_map = self.obfuscator.create_mapping()
            mapping_str_for_prompt = self.obfuscator.format_map_for_prompt(current_char_map)

            obfuscated_q_text = self.obfuscator.obfuscate_text(original_question, current_char_map)

            obfuscated_options_formatted = []
            for idx, opt_text in enumerate(original_choices_text):
                obfuscated_opt_text = self.obfuscator.obfuscate_text(opt_text, current_char_map)
                obfuscated_options_formatted.append(f"{self.OPTION_LABELS[idx]}) {obfuscated_opt_text}")
            obfuscated_options_str = "\n".join(obfuscated_options_formatted)

            correct_answer_label = self.OPTION_LABELS[correct_choice_idx]

            instruction_text = (
                "Instruction: The following multiple-choice question and its options have had their lowercase English letters "
                "replaced by other symbols according to the character map provided below. "
                "Uppercase letters, numbers, punctuation, and spaces in the original text remain unchanged. "
                "The option labels (A, B, C, D) also remain unchanged.\n\n"
                "Your task is to:\n"
                "1. Understand the question and options using the character map.\n"
                "2. Identify the correct option (A, B, C, or D).\n"
                "3. Provide the single capital letter corresponding to your chosen option as your answer, enclosed in <answer></answer> tags.\n"
            )

            question_text_for_llm = (
                f"{instruction_text}\n"
                f"Character Mapping Used:\n{mapping_str_for_prompt}\n\n"
                f"Obfuscated Question:\n{obfuscated_q_text}\n\n"
                f"Options:\n{obfuscated_options_str}\n\n"
                f"Which option is correct (A, B, C, or D)?"
            )

            item_id_subject_field = mmlu_item.get('subject') # MMLU 'all' items might not have 'subject'
            if item_id_subject_field is None: # try 'topic' or 'category' if they exist in your MMLU variant
                item_id_subject_field = mmlu_item.get('topic', mmlu_item.get('category', 'unknown_subject'))

            item_id_subject = str(item_id_subject_field).replace(" ", "_").lower()
            item_id = f"{self.name}-mmlu-{item_id_subject}-{uuid.uuid4().hex[:6]}"

            yield QAItem(
                id=item_id,
                question=question_text_for_llm,
                answer=correct_answer_label,
                modality='text',
                skill_coefficient = 3,
                metadata={
                    'mmlu_subject': item_id_subject,
                    'original_question': original_question,
                    'original_options': original_choices_text,
                    'original_correct_choice_index': correct_choice_idx,
                    'char_map_used': current_char_map,
                    'output_format_instruction': "<answer>CHOICE_LETTER</answer>"
                },
                verification_fn=self._verify_obfuscated_choice_label
            )

    @staticmethod
    def _verify_obfuscated_choice_label(expected_label: str, provided_llm_output: str, qa_item: QAItem) -> bool:
        logger = getattr(qa_item, 'logger', logging.getLogger("MMLUObfuscatedChoiceQATest.VerificationFn"))

        match = re.search(r"<answer>\s*([A-D])\s*</answer>", provided_llm_output.strip(), re.IGNORECASE)
        if not match:
            logger.warning(f"VFY {qa_item.id}: No/Invalid <answer> tags or invalid choice format. Raw LLM: '{provided_llm_output[:100]}'")
            return False

        extracted_llm_label = match.group(1).upper()

        is_correct = (extracted_llm_label == expected_label.upper())

        log_level = logging.INFO if is_correct else logging.WARNING
        logger.log(log_level,
                   f"MMLU Obfuscated Choice VFY {('PASSED' if is_correct else 'FAILED')} for {qa_item.id}. "
                   f"Exp: '{expected_label}', LLM: '{extracted_llm_label}'.")
        return is_correct


if __name__ == '__main__':
    # --- Standalone setup for AbstractQATest, QAItem, register_test ---
    # This block is crucial for making the file runnable directly for testing.
    if 'AbstractQATest' not in globals():
        # Setup basic logging for standalone run
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        # Add knit_space parent to sys.path to allow 'from ..obscurers import' if run directly from tests dir
        import sys
        knit_space_parent_dir = os.path.dirname(current_script_dir) # Should be knit_space
        project_root_dir = os.path.dirname(knit_space_parent_dir) # Should be Test-Minor-Project
        if knit_space_parent_dir not in sys.path:
            sys.path.insert(0, knit_space_parent_dir)
        if project_root_dir not in sys.path: # To find knit_space itself
             sys.path.insert(0, project_root_dir)


        print("Running MMLUObfuscatedChoiceQATest in standalone mode with stubs and path adjustments.")
        
        # Redefine base stubs if they weren't imported
        class AbstractQATest:
            def __init__(self, config=None):
                self.config = config or {}; self.logger = logging.getLogger(self.__class__.__name__)
                if not self.logger.handlers: # Check actual handlers, not just hasHandlers
                    h = logging.StreamHandler(); h.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                    self.logger.addHandler(h); self.logger.propagate = False
                self.logger.setLevel(logging.DEBUG) # Ensure level is set
            @property
            def name(self): return self.__class__.__name__
        class QAItem:
            def __init__(self, id, question, answer, modality, metadata, verification_fn=None):
                self.id=id;self.question=question;self.answer=answer;self.modality=modality;self.metadata=metadata;self.verification_fn=verification_fn
                self.logger = logging.getLogger(f"QAItemStandalone_{id[:8]}")
                if not self.logger.handlers:
                    h = logging.StreamHandler(); h.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                    self.logger.addHandler(h); self.logger.propagate = False
                self.logger.setLevel(logging.DEBUG)
            def __repr__(self): return (f"QAItem(id='{self.id}', question='{self.question[:50]}...', answer='{self.answer}', metadata_keys={list(self.metadata.keys())})")
        def register_test(*args, **kwargs):
            def decorator(cls): return cls
            return decorator
        
        # Explicitly import CharObfuscator for standalone __main__
        # This assumes char_obfuscator.py is in knit_space/obscurers/
        try:
            from ..obscurers.char_obfuscator import CharObfuscator
            # Make it globally available in this __main__ scope for the test class __init__
            globals()['CharObfuscator'] = CharObfuscator
            print("CharObfuscator loaded successfully for standalone __main__.")
        except ImportError as e:
            print(f"CRITICAL: Failed to import CharObfuscator for standalone __main__: {e}")
            print("Ensure char_obfuscator.py is in knit_space/obscurers/ and paths are correct.")
            # Set CharObfuscator to None or a placeholder if you want the script to run further but fail gracefully
            globals()['CharObfuscator'] = None


    # --- End Standalone setup ---
    
    print("Running MMLUObfuscatedChoiceQATest standalone...")
    test_config = {
        "mmlu_config": "all",
        "mmlu_split": "dev",
    }
    
    mmlu_choice_generator = MMLUObfuscatedChoiceQATest(config=test_config)
    # Logger level was set in stub AbstractQATest __init__

    num_items_to_generate = 2
    print(f"\n--- Requesting count={num_items_to_generate} items ---")

    generated_items_list = []
    if mmlu_choice_generator.dataset and mmlu_choice_generator.obfuscator:
        for i, item in enumerate(mmlu_choice_generator.generate(count=num_items_to_generate)):
            generated_items_list.append(item)
            print(f"\n--- Generated QAItem {i+1} (ID: {item.id}) ---")
            print(f"  MMLU Subject: {item.metadata.get('mmlu_subject')}")
            print(f"  Expected Answer Label: {item.answer}")
            # ... (rest of your print statements and verification simulation) ...
            if i == 0: # Test verification for the first item
                simulated_correct_llm_output = f"<answer>{item.answer}</answer>"
                print(f"  Simulated Verification (Correct LLM Output '{simulated_correct_llm_output}'): {MMLUObfuscatedChoiceQATest._verify_obfuscated_choice_label(item.answer, simulated_correct_llm_output, item)}")


    else:
        print("MMLU dataset not loaded or obfuscator not ready. No items generated.")

    if not generated_items_list:
        print(f"\nNo items generated. Check MMLU dataset loading and paths/configs.")
    else:
        print(f"\nSuccessfully generated {len(generated_items_list)} item(s).")