# knit_space/tests/medmcqa_obfuscated_choice_tests.py
import random
import uuid
import re
import string
import os
import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    from datasets import load_dataset
except ImportError:
    logging.critical("CRITICAL ERROR: `datasets` library not found. MedMCQAObfuscatedChoiceQATest cannot function.")
    logging.critical("Please install it: pip install datasets")
    pass

# Relative import for the obfuscator
try:
    from ..obscurers.char_obfuscator import CharObfuscator
except ImportError:
    logging.debug("Could not import CharObfuscator via relative path, will try direct if in __main__.")
    pass

# Relative import for base classes
try:
    from .base import AbstractQATest, QAItem, register_test
except ImportError:
    logging.debug("Could not import base classes via relative path, will use stubs if in __main__.")
    pass


@register_test('medmcqa', 'obfuscated_reasoning', 'medical_qa', 'symbol_manipulation', 'multiple_choice')
class MedMCQAObfuscatedChoiceQATest(AbstractQATest):
    DEFAULT_MEDMCQA_SPLIT = "train" # Or "validation", "test"
    DEFAULT_SYMBOLS_POOL = "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ" # Same pool as MMLU
    OPTION_LABELS = ['A', 'B', 'C', 'D'] # 0-3 indexed

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config) # Initializes self.logger
        self.medmcqa_split = self.config.get("medmcqa_split", self.DEFAULT_MEDMCQA_SPLIT)
        
        self.dataset = None
        self._load_medmcqa_dataset()

        symbols_pool_override = self.config.get("obfuscation_symbols_pool")
        
        CharObfuscator_class = None
        if 'CharObfuscator' in globals():
            CharObfuscator_class = globals()['CharObfuscator']
        else:
            try:
                from ..obscurers.char_obfuscator import CharObfuscator as CO_from_import
                CharObfuscator_class = CO_from_import
                logging.debug("CharObfuscator loaded dynamically in MedMCQA __init__.")
            except ImportError:
                self.logger.critical("CharObfuscator class could not be imported or found globally for MedMCQA.")

        if CharObfuscator_class:
            try:
                self.obfuscator = CharObfuscator_class(symbols_pool=symbols_pool_override or self.DEFAULT_SYMBOLS_POOL)
            except ValueError as e:
                self.logger.critical(f"Failed to initialize CharObfuscator for MedMCQA: {e}. This test will not function.")
                self.obfuscator = None
        else:
            self.logger.critical("CharObfuscator class definition is missing for MedMCQA. This test will not function.")
            self.obfuscator = None

    def _load_medmcqa_dataset(self):
        if 'load_dataset' not in globals():
            self.logger.critical("`datasets` library not loaded. Cannot load MedMCQA dataset.")
            return
        try:
            self.logger.info(f"Loading MedMCQA dataset: split='{self.medmcqa_split}'...")
            # MedMCQA does not use a 'name' or 'config' like MMLU for its main variants
            ds_full = load_dataset("openlifescienceai/medmcqa")
            if self.medmcqa_split not in ds_full:
                self.logger.error(f"Split '{self.medmcqa_split}' not found in MedMCQA dataset. Available: {list(ds_full.keys())}.")
                available_splits = list(ds_full.keys())
                # Fallback logic
                if 'train' in available_splits: self.medmcqa_split_to_use = 'train'
                elif 'validation' in available_splits: self.medmcqa_split_to_use = 'validation'
                elif 'test' in available_splits: self.medmcqa_split_to_use = 'test'
                elif available_splits: self.medmcqa_split_to_use = available_splits[0]
                else:
                    self.logger.error("No splits found in the loaded MedMCQA dataset.")
                    self.dataset = None
                    return
                self.logger.warning(f"Using fallback MedMCQA split: '{self.medmcqa_split_to_use}'")
                self.dataset = ds_full[self.medmcqa_split_to_use].to_list()
            else:
                self.dataset = ds_full[self.medmcqa_split].to_list()

            if self.dataset:
                self.logger.info(f"Loaded {len(self.dataset)} examples from MedMCQA - '{self.medmcqa_split if self.medmcqa_split in ds_full else self.medmcqa_split_to_use}'.")
            else:
                self.logger.error("MedMCQA dataset is empty after loading attempts.")
        except Exception as e:
            self.logger.error(f"Failed to load MedMCQA dataset: {e}", exc_info=True)
            self.dataset = None

    def generate(self, count: int = 1, **kwargs) -> Iterator[QAItem]:
        if not self.dataset or not self.obfuscator:
            self.logger.error(f"{self.name}: MedMCQA dataset not loaded or obfuscator not ready. Skipping generation.")
            return

        for i in range(count):
            medmcqa_item = random.choice(self.dataset)
            
            original_question = medmcqa_item.get('question', '')
            # Options are in opa, opb, opc, opd
            original_choices_text = [
                medmcqa_item.get('opa', ''), medmcqa_item.get('opb', ''),
                medmcqa_item.get('opc', ''), medmcqa_item.get('opd', '')
            ]
            correct_choice_cop = medmcqa_item.get('cop') # This is 1, 2, 3, or 4

            # Validate fetched item
            if not original_question or not all(original_choices_text) or correct_choice_cop is None:
                self.logger.warning(f"Skipping malformed MedMCQA item (idx {i}): Missing question, options, or correct answer. Item: {medmcqa_item}")
                continue
            try:
                correct_choice_idx_0_based = int(correct_choice_cop) - 1 # Convert 1-4 to 0-3
                if not (0 <= correct_choice_idx_0_based < 4):
                    raise ValueError("Correct option index out of bounds.")
            except ValueError:
                self.logger.warning(f"Skipping malformed MedMCQA item (idx {i}): Invalid 'cop' value '{correct_choice_cop}'.")
                continue

            current_char_map = self.obfuscator.create_mapping()
            mapping_str_for_prompt = self.obfuscator.format_map_for_prompt(current_char_map)

            obfuscated_q_text = self.obfuscator.obfuscate_text(original_question, current_char_map)
            
            obfuscated_options_formatted = []
            for idx, opt_text in enumerate(original_choices_text):
                obfuscated_opt_text = self.obfuscator.obfuscate_text(opt_text, current_char_map)
                obfuscated_options_formatted.append(f"{self.OPTION_LABELS[idx]}) {obfuscated_opt_text}")
            obfuscated_options_str = "\n".join(obfuscated_options_formatted)

            correct_answer_label = self.OPTION_LABELS[correct_choice_idx_0_based]

            instruction_text = ( # Same instruction as MMLU choice version
                "Instruction: The following multiple-choice question and its options have had their lowercase English letters "
                "replaced by other symbols according to the character map provided below. "
                "Uppercase letters, numbers, punctuation, and spaces in the original text remain unchanged. "
                "The option labels (A, B, C, D) also remain unchanged.\n\n"
                "Your task is to:\n"
                "1. Understand the question and options using the character map.\n"
                "2. Identify the correct option (A, B, C, or D).\n"
                "3. Provide the single capital letter corresponding to your chosen option as your answer, enclosed in <answer></answer> tags.e.g(<answer>E</answer>)\n"
            )

            question_text_for_llm = (
                f"{instruction_text}\n"
                f"Character Mapping Used:\n{mapping_str_for_prompt}\n\n"
                f"Obfuscated Question:\n{obfuscated_q_text}\n\n"
                f"Options:\n{obfuscated_options_str}\n\n"
                f"Which option is correct (A, B, C, or D)?"
            )
            
            subject_name = str(medmcqa_item.get('subject_name', 'unknown_subject')).replace(" ", "_").lower()
            exp_id = str(medmcqa_item.get('exp', 'no_exp')).replace(" ", "_") # Add explanation if available
            item_id = f"{self.name}-{subject_name}-{uuid.uuid4().hex[:6]}"

            yield QAItem(
                id=item_id,
                question=question_text_for_llm,
                answer=correct_answer_label,
                skill_coefficient=3,
                modality='text',
                metadata={
                    'medmcqa_subject_name': subject_name,
                    'medmcqa_exp': exp_id, # Explanation for the answer if provided by dataset
                    'original_question': original_question,
                    'original_options': original_choices_text,
                    'original_correct_choice_index_1_based': correct_choice_cop,
                    'char_map_used': current_char_map,
                    'output_format_instruction': "<answer>CHOICE_LETTER</answer>"
                },
                verification_fn=self._verify_obfuscated_choice_label # Can reuse the same verification
            )

    @staticmethod
    def _verify_obfuscated_choice_label(expected_label: str, provided_llm_output: str, qa_item: QAItem) -> bool:
        # This static method can be shared with MMLUObfuscatedChoiceQATest
        # or copied here if you want separate logging contexts per test type.
        # For simplicity, let's assume it's defined here (or imported if put in a common util).
        logger = getattr(qa_item, 'logger', logging.getLogger("MedMCQAObfuscatedChoiceQATest.VerificationFn"))
        
        match = re.fullmatch(r"<answer>\s*([A-D])\s*</answer>", provided_llm_output.strip(), re.IGNORECASE)
        if not match:
            logger.warning(f"VFY {qa_item.id}: No/Invalid <answer> tags or invalid choice format. Raw LLM: '{provided_llm_output[:100]}'")
            return False
        
        extracted_llm_label = match.group(1).upper()
        
        is_correct = (extracted_llm_label == expected_label.upper())
        
        log_level = logging.INFO if is_correct else logging.WARNING
        logger.log(log_level,
                   f"MedMCQA Obfuscated Choice VFY {('PASSED' if is_correct else 'FAILED')} for {qa_item.id}. "
                   f"Exp: '{expected_label}', LLM: '{extracted_llm_label}'.")
        return is_correct

if __name__ == '__main__':
    # --- Standalone setup ---
    if 'AbstractQATest' not in globals():
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        import sys
        knit_space_parent_dir = os.path.dirname(current_script_dir)
        project_root_dir = os.path.dirname(knit_space_parent_dir)
        if knit_space_parent_dir not in sys.path: sys.path.insert(0, knit_space_parent_dir)
        if project_root_dir not in sys.path: sys.path.insert(0, project_root_dir)
        print("Running MedMCQAObfuscatedChoiceQATest in standalone mode with stubs and path adjustments.")
        class AbstractQATest:
            def __init__(self, config=None):
                self.config=config or {}; self.logger=logging.getLogger(self.__class__.__name__)
                if not self.logger.handlers:
                    h=logging.StreamHandler(); h.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                    self.logger.addHandler(h); self.logger.propagate=False
                self.logger.setLevel(logging.DEBUG)
            @property
            def name(self): return self.__class__.__name__
        class QAItem:
            def __init__(self,id,question,answer,modality,metadata,verification_fn=None):
                self.id=id;self.question=question;self.answer=answer;self.modality=modality;self.metadata=metadata;self.verification_fn=verification_fn
                self.logger = logging.getLogger(f"QAItemStandalone_{id[:8]}")
                if not self.logger.handlers:
                    h=logging.StreamHandler();h.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                    self.logger.addHandler(h);self.logger.propagate=False
                self.logger.setLevel(logging.DEBUG)
            def __repr__(self): return (f"QAItem(id='{self.id}', question='{self.question[:50]}...', answer='{self.answer}', metadata_keys={list(self.metadata.keys())})")
        def register_test(*args,**kwargs):
            def decorator(cls): return cls
            return decorator
        try:
            from ..obscurers.char_obfuscator import CharObfuscator
            globals()['CharObfuscator'] = CharObfuscator
            print("CharObfuscator loaded successfully for MedMCQA standalone __main__.")
        except ImportError as e:
            print(f"CRITICAL: Failed to import CharObfuscator for MedMCQA standalone __main__: {e}")
            globals()['CharObfuscator'] = None
    # --- End Standalone setup ---

    print("Running MedMCQAObfuscatedChoiceQATest standalone...")
    test_config = {"medmcqa_split": "validation"} # 'validation' split is often a good choice for testing
    
    medmcqa_generator = MedMCQAObfuscatedChoiceQATest(config=test_config)
    medmcqa_generator.logger.setLevel(logging.DEBUG)

    num_items_to_generate = 2
    print(f"\n--- Requesting count={num_items_to_generate} items ---")

    generated_items_list = []
    if medmcqa_generator.dataset and medmcqa_generator.obfuscator:
        for i, item in enumerate(medmcqa_generator.generate(count=num_items_to_generate)):
            generated_items_list.append(item)
            print(f"\n--- Generated QAItem {i+1} (ID: {item.id}) ---")
            print(f"  MedMCQA Subject: {item.metadata.get('medmcqa_subject_name')}")
            print(f"  Expected Answer Label: {item.answer}")
            # print(f"  Full Question to LLM (first 500 chars):\n{item.question[:500]}...")
            
            if i == 0:
                simulated_correct_llm_output = f"<answer>{item.answer}</answer>"
                is_verified = MedMCQAObfuscatedChoiceQATest._verify_obfuscated_choice_label(item.answer, simulated_correct_llm_output, item)
                print(f"  Simulated Verification (Correct LLM Output '{simulated_correct_llm_output}'): {is_verified}")
    else:
        print("MedMCQA dataset not loaded or obfuscator not ready. No items generated.")

    if not generated_items_list:
        print(f"\nNo items generated. Check MedMCQA dataset loading and paths/configs.")
    else:
        print(f"\nSuccessfully generated {len(generated_items_list)} item(s).")