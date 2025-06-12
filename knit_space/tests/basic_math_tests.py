# knit_space/tests/basic_math_tests.py

from .base  import AbstractQATest, QAItem, create_test_cases, register_test
import random
import uuid
from typing import Any, Dict, Iterator, Optional, List, Union
from decimal import Decimal, ROUND_HALF_UP, getcontext, ROUND_UP, ROUND_DOWN, InvalidOperation # Added InvalidOperation
import re


@register_test('math', 'basic_arithmetic')
class MathQATestClass(AbstractQATest):
    """
    Generates basic math questions involving addition, subtraction,
    multiplication, and division.

    Features:
    - Operations: Addition, Subtraction (rational numbers, 3 dp answers),
                  Multiplication (integers, integer answers),
                  Division (integers, aiming for >2 dp float answers, formatted to 3-5 dp).
    - Output: 3 test cases per operation for each `count` cycle in `generate()`.
    - Prompting: Instructs answer format: "%> answer <&".
    - `obs` mapping: Appends text transformation rules if `obs` dict is provided.
    - Verification: Includes a sample numerical verification function.
    """

    @property
    def supported_modalities(self) -> List[str]:
        return ['text']

    def _format_decimal_to_str(self, num: Union[Decimal, float, str], precision: int) -> str:
        """
        Formats a number (Decimal, float, or string representation of a number)
        to a string with a specific number of decimal places, using ROUND_HALF_UP.
        """
        if not isinstance(num, Decimal):
            try:
                num = Decimal(str(num))
            except InvalidOperation:
                # Handle cases where conversion to Decimal might fail for unexpected input
                raise ValueError(f"Cannot convert '{num}' (type: {type(num)}) to Decimal for formatting.")

        if precision < 0:
            # Or raise ValueError("Precision cannot be negative")
            precision = 0 # Default to integer if negative precision is given

        if precision == 0:
            quantizer = Decimal('1')
        else:
            # Correctly create a quantizer for n decimal places
            # e.g., for precision=3, quantizer is Decimal('0.001')
            quantizer = Decimal('1e-' + str(precision))

        return str(num.quantize(quantizer, rounding=ROUND_HALF_UP))


    # In knit_space/tests/basic_math_tests.py
# Inside the MathQATestClass class:

    # In knit_space/tests/basic_math_tests.py
# Inside the MathQATestClass class:

    # In knit_space/tests/basic_math_tests.py
# Inside MathQATestClass:

    # In knit_space/tests/basic_math_tests.py
# Inside MathQATestClass:

    @staticmethod
    def _verify_numerical_response(expected_answer_str: str,
                                   provided_response: Any, self) -> bool:
        # print(f"\n--- Verifying Math Response ---")
        # print(f"DEBUG: Expected Answer (str): '{expected_answer_str}'")
        # print(f"DEBUG: Provided Response (raw): '{provided_response}'")

        _hardcoded_float_tolerance = Decimal('0.001')

        try:
            expected_num = Decimal(expected_answer_str)
            # print(f"DEBUG: Expected_num (Decimal): {expected_num}")

            extracted_provided_num_str = None
            if isinstance(provided_response, str):
                # Attempt 1: Strict format "%> NUMBER <&" (allow for trailing "...")
                # Regex captures the number part, optionally followed by "..."
                # We then strip "..." if it was captured with the number.
                match_strict = re.search(r"\%>\s*([-\d.]+(?:\.\.\.)?)\s*<\&", provided_response)
                if match_strict:
                    temp_extracted = match_strict.group(1)
                    if temp_extracted.endswith("..."): # If "..." is part of it, remove it
                        extracted_provided_num_str = temp_extracted[:-3]
                    else:
                        extracted_provided_num_str = temp_extracted
                    # print(f"DEBUG: Extracted from strict format (post ... strip): '{extracted_provided_num_str}'")

                # If strict match failed or resulted in empty after stripping "...", try other methods
                if not extracted_provided_num_str:
                    match_alt_format = re.search(r"\%>\s*answer\s*<\&\s*([-\d.]+)", provided_response, re.IGNORECASE)
                    if match_alt_format:
                        extracted_provided_num_str = match_alt_format.group(1)
                        # print(f"DEBUG: Extracted from alt format: '{extracted_provided_num_str}'")

                if not extracted_provided_num_str:
                    # Fallback: find the most "complete" number in the whole string.
                    # This regex tries to find a number that is not part of a "version-like" string if possible.
                    # It prioritizes numbers that look like standalone values.
                    all_numbers = re.findall(r"([-\d.]+)", provided_response)
                    if all_numbers:
                        # Prefer a number that is not immediately followed by non-space and non-digit,
                        # unless it's the only one. This is a heuristic.
                        # For "7. 3131313131", it would find "7" and "3131313131".
                        # We want the longest one or the one that seems most like the answer.
                        # Let's try to pick the one with a decimal if expected has one, or longest.
                        if '.' in expected_answer_str:
                            best_match = None
                            for num_str in all_numbers:
                                if '.' in num_str:
                                    if best_match is None or len(num_str) > len(best_match):
                                        best_match = num_str
                            if best_match:
                                extracted_provided_num_str = best_match
                            elif all_numbers: # If no float found, take the first one as a last resort
                                extracted_provided_num_str = all_numbers[0]
                        elif all_numbers: # If expecting integer, take the first one
                             extracted_provided_num_str = all_numbers[0]
                        # print(f"DEBUG: Extracted from fallback (all_numbers): '{extracted_provided_num_str}' from {all_numbers}")


            elif isinstance(provided_response, (int, float, Decimal)):
                extracted_provided_num_str = str(provided_response)
                # print(f"DEBUG: Provided response was already numeric: '{extracted_provided_num_str}'")


            if not extracted_provided_num_str: # Check if it's None or empty string
                # print(f"DEBUG: Could not extract number. Returning False.")
                return False

            # Final check to ensure extracted_provided_num_str is not just "..."
            if extracted_provided_num_str == "...":
                # print(f"DEBUG: Extracted string is just '...'. Returning False.")
                return False

            provided_num = Decimal(extracted_provided_num_str)
            # print(f"DEBUG: Provided_num (Decimal): {provided_num}")

            is_expected_integer_type = '.' not in expected_answer_str

            if is_expected_integer_type:
                # print(f"DEBUG: Comparing as INTEGER type.")
                is_provided_an_integer_value = (provided_num == provided_num.to_integral_value(rounding=ROUND_DOWN))
                if not is_provided_an_integer_value:
                    # print(f"DEBUG: Expected integer, but provided is not whole. Returning False.")
                    return False
                result = (expected_num == provided_num)
                # print(f"DEBUG: Integer comparison result: {result}")
                return result
            else:
                # print(f"DEBUG: Comparing as REAL NUMBER type.")
                diff = abs(expected_num - provided_num)
                # print(f"DEBUG: Difference (abs): {diff}")
                # print(f"DEBUG: Using hardcoded float tolerance: {_hardcoded_float_tolerance}")
                result = (diff <= _hardcoded_float_tolerance)
                # print(f"DEBUG: Real number comparison result: {result}")
                # if not result and diff <= Decimal('0.01'): # Temp debug for near misses
                #    print(f"!!!! NEAR MISS: Expected={expected_num}, Got={provided_num}, Diff={diff}, Tol={_hardcoded_float_tolerance}")
                return result

        except InvalidOperation as e:
            # print(f"DEBUG: InvalidOperation during Decimal conversion: {e}. Extracted str: '{extracted_provided_num_str if 'extracted_provided_num_str' in locals() else 'N/A'}'. Returning False.")
            return False
        except Exception as e:
            # print(f"DEBUG: An unexpected error occurred: {e}. Returning False.")
            # import traceback
            # print(traceback.format_exc())
            return False

    def generate(self,
                 count: int = 1,
                 difficulty: Optional[str] = None,
                 prefix: Optional[str] = None,
                 suffix: Optional[str] = None, # This is the default suffix
                 **kwargs) -> Iterator[QAItem]:

        obs_mapping: Optional[Dict[str, str]] = kwargs.get('obs')
        
        operations = ['addition', 'subtraction', 'multiplication', 'division']
        num_cases_per_op = 3 

        for _ in range(count):
            for op_type in operations:
                for i in range(num_cases_per_op):
                    q_math_part = ""
                    expected_answer_val_str = ""
                    # Use a local suffix for each question to avoid carry-over from division
                    current_suffix_for_question = suffix
                    
                    question_metadata = {
                        'operation': op_type,
                        'difficulty': difficulty or 'medium',
                        'case_index': i + 1 
                    }

                    if op_type == 'addition' or op_type == 'subtraction':
                        # 1. Generate raw numbers (can be float)
                        raw_val_a = random.uniform(1, 1000)
                        raw_val_b = random.uniform(1, 1000)

                        # 2. Decide precision for numbers in the question string (e.g., 0-3 dp)
                        q_precision_a = random.randint(0, 3)
                        q_precision_b = random.randint(0, 3)

                        # 3. Format these raw numbers to get the strings that will appear in the question
                        #    This uses the *corrected* _format_decimal_to_str
                        a_str_q = self._format_decimal_to_str(raw_val_a, q_precision_a)
                        b_str_q = self._format_decimal_to_str(raw_val_b, q_precision_b)
                        
                        # 4. The calculation for the expected answer MUST use these formatted numbers
                        a_q_dec = Decimal(a_str_q)
                        b_q_dec = Decimal(b_str_q)
                        
                        if op_type == 'addition':
                            res_dec = a_q_dec + b_q_dec
                            q_math_part = f"What is {a_str_q} + {b_str_q}?"
                        else: # Subtraction
                            # Ensure a mix of positive and negative results if a_q_dec < b_q_dec
                            # (optional: can also be achieved by random range of raw_val_a and raw_val_b)
                            if random.choice([True, False]) and a_q_dec < b_q_dec:
                               a_q_dec, b_q_dec = b_q_dec, a_q_dec # Swap Decimal versions
                               a_str_q, b_str_q = b_str_q, a_str_q # Also swap string versions for question

                            res_dec = a_q_dec - b_q_dec
                            q_math_part = f"What is {a_str_q} - {b_str_q}?"
                        
                        # 5. Format the final expected answer to 3 decimal places as per spec
                        expected_answer_val_str = self._format_decimal_to_str(res_dec, 3)

                    elif op_type == 'multiplication':
                        val_a_int = random.randint(1, 100)
                        val_b_int = random.randint(1, 100)
                        res_int = val_a_int * val_b_int
                        expected_answer_val_str = str(res_int)
                        q_math_part = f"What is {val_a_int} * {val_b_int}?"

                    elif op_type == 'division':
                        # Loop to ensure a valid division (non-zero denominator)
                        # and potentially to meet other criteria if added later
                        for _attempt in range(100): # Max attempts to find suitable numbers
                            numerator = random.randint(1, 1000)
                            denominator = random.randint(1, 1000)
                            if denominator == 0:
                                continue # Avoid division by zero

                            res_decimal = Decimal(numerator) / Decimal(denominator)
                            
                            # As per original logic: "formatted to 3-5 dp", using ROUND_UP to 3dp here.
                            # The docstring said 3-5 dp. Let's stick to 3dp for consistency, or make it variable.
                            # Using 3 decimal places with ROUND_UP as in the original snippet for division.
                            ans_precision = 3
                            quantizer = Decimal('1e-' + str(ans_precision))
                            rounded_result = res_decimal.quantize(quantizer, rounding=ROUND_UP)

                            expected_answer_val_str = str(rounded_result)
                            q_math_part = f"What is {numerator} / {denominator}?"
                            question_metadata['numerator'] = numerator
                            question_metadata['denominator'] = denominator
                            question_metadata['result_precision'] = ans_precision
                            current_suffix_for_question = "The answer should be in decimals" # Override default suffix
                            break # Found suitable division
                        else:
                            # Fallback if no suitable division found after many attempts (should be rare)
                            # Or raise an error, or skip this test case
                            continue


                    question_core_content = f"{q_math_part}\nFormat your answer as: %> answer <&"
                    
                    obs_text_addon = ""
                    if obs_mapping:
                        obs_text_parts = [f"'{key}' maps to '{value}'" for key, value in obs_mapping.items()]
                        obs_text_addon = "\nAdditionally, apply the following text transformations based on this mapping: " + ", ".join(obs_text_parts) + "."
                    
                    question_body_with_obs = question_core_content + obs_text_addon

                    final_question_str = self.build_question(
                        base_question=question_body_with_obs,
                        prefix=prefix,
                        suffix=current_suffix_for_question # Use the locally determined suffix
                    )
                    
                    yield QAItem(
                        id=f"{self.name}-{op_type}-{uuid.uuid4().hex[:8]}",
                        question=final_question_str,
                        answer=expected_answer_val_str,
                        skill_coefficient=1,
                        modality='text',
                        metadata=question_metadata,
                        verification_fn=self._verify_numerical_response
                    )