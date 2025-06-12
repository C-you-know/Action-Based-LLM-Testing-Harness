import random
import uuid
import re
import math 
from typing import Any, Dict, Iterator, List, Optional, Union, Callable 
from decimal import Decimal, InvalidOperation 


from .base import AbstractQATest, QAItem, register_test 

def _is_prime(n: int) -> bool:
    if n < 2: return False
    if n == 2 or n == 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _fibonacci(n: int) -> int:
    if n <= 0: return 0 # Or raise error
    if n == 1: return 0 # Or 1 depending on definition (0, 1, 1, 2... or 1, 1, 2...)
    if n == 2: return 1
    a, b = 0, 1 # Assuming F0=0, F1=1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


@register_test('n_rules_vector', 'constraints', 'mixed_math')
class NRulesVectorQATest(AbstractQATest):
    DEFAULT_N_RULES = 50
    MAX_INT_PARAM = 1000 # Max value for random integer parameters for rules
    MAX_FLOAT_PARAM = 500.0
    FLOAT_TOLERANCE = Decimal('0.001') # General tolerance for float comparisons


    _prime_cache = {}
    _non_prime_cache = {}


    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.n_rules = self.config.get("num_rules", self.DEFAULT_N_RULES)
        if not isinstance(self.n_rules, int) or self.n_rules <= 0:
            self.logger.warning(f"Invalid num_rules: {self.n_rules}. Defaulting to {self.DEFAULT_N_RULES}.")
            self.n_rules = self.DEFAULT_N_RULES

        self.rule_generator_functions = [
            self._gen_positive_integer_range_rule,
            self._gen_negative_integer_range_rule,
            self._gen_modular_rule,
            self._gen_divisibility_rule,
            # self._gen_digit_sum_rule, # Requires int parsing in validator
            self._gen_algebraic_addition_rule, # New, based on our earlier discussion
            self._gen_algebraic_subtraction_rule, # New
            self._gen_algebraic_multiplication_rule, # New
            # self._gen_quadratic_form_rule, # From your code
            # self._gen_fibonacci_rule,      # From your code
            # self._gen_factorial_rule,      # From your code
            # self._gen_prime_in_range_rule, # From your code
            # self._gen_perfect_square_rule, # From your code
            # self._gen_triangular_rule,     # From your code
        ]
        # You can add more from your list if they primarily deal with integers or
        # if we decide to handle float rules more explicitly now.

    # --- Rule Generation Methods (adapted from your _range_rule etc.) ---
    def _gen_positive_integer_range_rule(self, index: int) -> Dict[str, Any]:
        min_val = random.randint(1, self.MAX_INT_PARAM - 100)
        max_val = random.randint(min_val + 10, min_val + 100)
        desc = f"Rule {index}: The value for ans[{index}] must be an integer between {min_val} and {max_val} (inclusive)."
        return {'type': 'positive_integer_range', 'index': index, 'min': min_val, 'max': max_val, 'description': desc}

    def _gen_negative_integer_range_rule(self, index: int) -> Dict[str, Any]:
        min_val = random.randint(1, self.MAX_INT_PARAM - 100)
        max_val = random.randint(min_val + 5, min_val + 50)
        desc = f"Rule {index}: The value for ans[{index}] must be an integer NOT between {min_val} and {max_val} (inclusive)."
        return {'type': 'negative_integer_range', 'index': index, 'min': min_val, 'max': max_val, 'description': desc}

    def _gen_modular_rule(self, index: int) -> Dict[str, Any]:
        # Your _modular_rule is good, but let's remove the extra ans[i] <= max_range for simplicity,
        # unless it's crucial for solvability/bounding.
        modulus = random.randint(3, 20)
        remainder = random.randint(0, modulus - 1)
        # max_val_for_rule = random.randint(50, self.MAX_INT_PARAM // 2) # Example upper bound
        desc = f"Rule {index}: The value for ans[{index}] must be an integer such that ans[{index}] % {modulus} = {remainder}."
        return {'type': 'modular_arithmetic', 'index': index, 'modulus': modulus, 'remainder': remainder, 'description': desc} # 'max_val': max_val_for_rule

    def _gen_divisibility_rule(self, index: int) -> Dict[str, Any]:
        divisor = random.randint(2, 25)
        # max_val_for_rule = random.randint(100, self.MAX_INT_PARAM // 2)
        desc = f"Rule {index}: The value for ans[{index}] must be an integer divisible by {divisor}."
        return {'type': 'divisibility', 'index': index, 'divisor': divisor, 'description': desc} # 'max_val': max_val_for_rule

    def _gen_algebraic_addition_rule(self, index: int) -> Dict[str, Any]:
        a = random.randint(-self.MAX_INT_PARAM // 2, self.MAX_INT_PARAM // 2)
        b = random.randint(-self.MAX_INT_PARAM // 2, self.MAX_INT_PARAM // 2)
        result = a + b
        desc = f"Rule {index}: The value for ans[{index}] must be an integer equal to {a} + {b})."
        # Storing a and b allows verification without pre-calculating result in verification_fn
        return {'type': 'algebraic_addition', 'index': index, 'a': a, 'b': b, 'expected_result': result, 'description': desc}

    def _gen_algebraic_subtraction_rule(self, index: int) -> Dict[str, Any]:
        a = random.randint(-self.MAX_INT_PARAM // 2, self.MAX_INT_PARAM // 2)
        b = random.randint(-self.MAX_INT_PARAM // 2, self.MAX_INT_PARAM // 2)
        result = a - b
        desc = f"Rule {index}: The value for ans[{index}] must be an integer equal to {a} - {b})."
        return {'type': 'algebraic_subtraction', 'index': index, 'a': a, 'b': b, 'expected_result': result, 'description': desc}

    def _gen_algebraic_multiplication_rule(self, index: int) -> Dict[str, Any]:
        # Keep numbers small to avoid very large results
        a = random.randint(-self.MAX_INT_PARAM // 100, self.MAX_INT_PARAM // 100)
        b = random.randint(-self.MAX_INT_PARAM // 100, self.MAX_INT_PARAM // 100)
        if a == 0: a = 1 # Avoid trivial 0 * x
        if b == 0: b = 1
        result = a * b
        desc = f"Rule {index}: The value for ans[{index}] must be an integer equal to {a} * {b}."
        return {'type': 'algebraic_multiplication', 'index': index, 'a': a, 'b': b, 'expected_result': result, 'description': desc}

    # ... (Add more rule generator methods here, adapting from your code)
    # For example, _gen_digit_sum_rule, _gen_fibonacci_rule etc. ensuring they return the dict structure.

    def generate(self, count: int = 3, **kwargs) -> Iterator[QAItem]:
        # This is largely as we planned previously
        for _ in range(count):
            all_rule_descriptions = []
            rules_details_for_metadata = []

            current_n_rules = kwargs.get('num_rules', self.n_rules) # Allow overriding N for this generation
            if not isinstance(current_n_rules, int) or current_n_rules <= 0:
                current_n_rules = self.n_rules


            for rule_idx in range(current_n_rules):
                if not self.rule_generator_functions:
                    self.logger.error("No rule generator functions defined!")
                    # Potentially raise an error or yield no items
                    return

                rule_generator_func = random.choice(self.rule_generator_functions)
                rule_detail = rule_generator_func(rule_idx)
                
                all_rule_descriptions.append(rule_detail['description'])
                rules_details_for_metadata.append(rule_detail)

            question_text = (
                f"Consider a vector of {current_n_rules} values, ans[0] through ans[{current_n_rules - 1}]. "
                f"Each value is expected to be an integer unless specified otherwise by the rule. "
                f"Generate a valid vector that satisfies all the following rules:\n\n"
                + "\n".join(all_rule_descriptions)
                + f"\n\nProvide your answer as a comma-separated list of {current_n_rules} values within <answer></answer> tags. "
                f"For example: <answer>[value0, value1, ..., value{current_n_rules - 1}]</answer>"
            )
            item_id = f"{self.name}-{current_n_rules}_rules-{uuid.uuid4().hex[:8]}"

            yield QAItem(
                id=item_id,
                question=question_text,
                answer=True,
                skill_coefficient = 5,
                modality='text',
                metadata={
                    'num_rules': current_n_rules,
                    'rules_details': rules_details_for_metadata,
                },
                verification_fn=NRulesVectorQATest._verify_n_rules_vector # Assign static method
            )

    @staticmethod
    def _parse_llm_answer_vector(text: str, num_expected_values: int, logger) -> Optional[List[Any]]: # logger can be passed if not static, or use print
        """
        Extract answer vector from text using multiple parsing strategies.
        Returns a list of numbers (int or Decimal) or None if parsing fails.
        """
        # Strategy 1: Look for <answer>[...]</answer> tags (preferred)
        match1 = re.search(r'<answer>\s*\[(.*?)\]\s*</answer>', text, re.DOTALL | re.IGNORECASE)
        raw_numbers_str = None
        if match1:
            raw_numbers_str = match1.group(1)
        else:
            # Strategy 2: Look for any list-like structure [a, b, c, ...] as a fallback
            # This is less strict, might pick up other lists.
            match3 = re.search(r'\[([\d\.,\s\-]+)\]', text) # Find first list-like
            if match3:
                raw_numbers_str = match3.group(1)
            else:
                # Strategy 3: Extract all numbers and take first N as a last resort
                # This is very broad and might be error-prone.
                # Consider if this strategy is too lenient.
                all_numbers_found = re.findall(r'[\-]?\d+\.?\d*', text)
                if len(all_numbers_found) >= num_expected_values:
                    raw_numbers_str = ",".join(all_numbers_found[:num_expected_values])
                else:
                    # logger.info("Could not parse answer vector: No recognizable format or insufficient numbers.")
                    print(f"DEBUG: Could not parse answer vector: No recognizable format for {num_expected_values} numbers.")
                    return None
        
        if raw_numbers_str is None:
            print(f"DEBUG: raw_numbers_str is None after parsing attempts.")
            return None

        # Parse the extracted number string
        # Clean up: remove anything not a digit, dot, minus, comma, or whitespace
        cleaned = re.sub(r'[^\d\.\-,\s]', '', raw_numbers_str)
        parts = re.split(r'[,\s]+', cleaned.strip()) # Split by comma or one/more whitespace
        
        parsed_values = []
        for part in parts:
            part = part.strip()
            if part: # Ensure part is not empty after strip
                try:
                    # Attempt to parse as int first. If it has a decimal, parse as Decimal.
                    if '.' in part:
                        parsed_values.append(Decimal(part))
                    else:
                        parsed_values.append(int(part))
                except ValueError: # For int conversion
                    # logger.info(f"Could not parse value '{part}' as int/Decimal.")
                    print(f"DEBUG: Could not parse value '{part}' as int/Decimal.")
                    return None # Invalid number format in list
                except InvalidOperation: # For Decimal conversion
                    # logger.info(f"Could not parse value '{part}' as Decimal.")
                    print(f"DEBUG: Could not parse value '{part}' as Decimal.")
                    return None

        if len(parsed_values) != num_expected_values:
            # logger.info(f"Parsed {len(parsed_values)} values, but expected {num_expected_values}.")
            print(f"DEBUG: Parsed {len(parsed_values)} values, but expected {num_expected_values}.")
            return None
            
        return parsed_values


    @staticmethod
    def _verify_n_rules_vector(expected_answer: bool, provided_answer_str: str, qa_item: QAItem) -> bool:
        # `expected_answer` is True, not used for logic.
        # Accessing logger: Since this is static, we can't use self.logger.
        # We could pass a logger instance or use print for debugging.
        # For AbstractQATest, the framework might handle logging around this.
        # For now, using print for debug messages.

        rules_details = qa_item.metadata.get('rules_details', [])
        num_expected_rules = qa_item.metadata.get('num_rules', 0)

        if not rules_details or num_expected_rules == 0:
            print(f"ERROR: Verification for {qa_item.id}: Missing rule metadata.")
            return False

        # Use the adapted parser
        # If we had access to qa_item.logger (e.g., if this wasn't static or logger passed):
        # parsed_answer_vector = NRulesVectorQATest._parse_llm_answer_vector(provided_answer_str, num_expected_rules, qa_item.logger)
        parsed_answer_vector = NRulesVectorQATest._parse_llm_answer_vector(provided_answer_str, num_expected_rules, None)


        if parsed_answer_vector is None:
            print(f"DEBUG: Verification for {qa_item.id}: Parsing LLM answer failed or wrong count.")
            return False
        
        for i, value_from_llm in enumerate(parsed_answer_vector):
            rule = rules_details[i]
            rule_type = rule['type']
            is_valid_for_rule = False
            
            # Ensure value_from_llm is int for rules expecting int
            # For now, let's assume rules we've implemented expect integers.
            # If value_from_llm is Decimal but rule expects int, try converting.
            current_value = None
            if isinstance(value_from_llm, Decimal):
                if value_from_llm.as_tuple().exponent == 0: # It's a whole number
                    current_value = int(value_from_llm)
                else:
                    # Rule expects int, but LLM gave float.
                    # This could be an error depending on rule strictness.
                    # For now, if rule is int-specific, this will likely fail type checks or logic.
                    # Or we can be strict:
                    print(f"DEBUG: Rule {i} ({rule_type}) likely expects int, got Decimal {value_from_llm} for {qa_item.id}.")
                    # For now, let it proceed; specific rule logic will handle type.
                    current_value = value_from_llm # Keep as Decimal if rule might handle it
            else: # It's already an int
                current_value = value_from_llm

            try:
                if not isinstance(current_value, int) and rule_type in [
                    'positive_integer_range', 'negative_integer_range', 
                    'modular_arithmetic', 'divisibility',
                    'algebraic_addition', 'algebraic_subtraction', 'algebraic_multiplication'
                    # Add other int-specific rules here
                ]:
                    print(f"DEBUG: Rule {i} ({rule_type}) expects int, but value is {current_value} (type {type(current_value)}) for {qa_item.id}.")
                    return False # Type mismatch for integer rules

                if rule_type == 'positive_integer_range':
                    is_valid_for_rule = (rule['min'] <= current_value <= rule['max'])
                elif rule_type == 'negative_integer_range':
                    is_valid_for_rule = not (rule['min'] <= current_value <= rule['max'])
                elif rule_type == 'modular_arithmetic':
                    is_valid_for_rule = (current_value % rule['modulus'] == rule['remainder'])
                elif rule_type == 'divisibility':
                    is_valid_for_rule = (current_value % rule['divisor'] == 0)
                elif rule_type == 'algebraic_addition':
                    is_valid_for_rule = (current_value == rule['expected_result'])
                elif rule_type == 'algebraic_subtraction':
                    is_valid_for_rule = (current_value == rule['expected_result'])
                elif rule_type == 'algebraic_multiplication':
                    is_valid_for_rule = (current_value == rule['expected_result'])
                # Add more rule type checks here
                else:
                    print(f"WARNING: Verification for {qa_item.id}: Unknown rule type '{rule_type}' for rule {i}.")
                    return False 
            except TypeError as te:
                # Catch type errors, e.g., Decimal % int
                print(f"DEBUG: TypeError during validation for rule {i} ({rule_type}): {te}. Value: {current_value} for {qa_item.id}")
                return False
            except Exception as e:
                print(f"ERROR: Exception during rule validation for {qa_item.id}, rule {i} ({rule_type}): {e}")
                return False

            if not is_valid_for_rule:
                print(f"INFO: Verification failed for {qa_item.id}: Rule {i} ('{rule['description']}') violated by value {value_from_llm}.")
                return False
        
        return True