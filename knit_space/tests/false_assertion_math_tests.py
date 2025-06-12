# knit_space/tests/implicit_correction_math_tests.py

from .base import AbstractQATest, QAItem, register_test
import random
import uuid
from typing import Any, Dict, Iterator, Optional, List, Union
from decimal import Decimal, ROUND_HALF_UP, getcontext, InvalidOperation, ROUND_DOWN
import re
import math # For isclose if needed, but direct Decimal comparison with tolerance is better
import logging # Ensure logging is imported

# getcontext().prec = 30 # Set a good default precision for Decimal calculations

@register_test('reasoning', 'numerical_correction', 'implicit_verification', 'arithmetic')
class ImplicitCorrectionMathQATest(AbstractQATest):
    """
    Presents a false numerical statement (Fact: A op B = C_false).
    Then asks the LLM to perform a subsequent operation on "the result" from that statement
    (e.g., subtract 1 from "the sum").
    The test is whether the LLM uses the *actual correct result* of (A op B)
    for the subsequent operation, or if it blindly uses C_false.
    The QAItem.answer is based on the *actual correct result*.
    """

    DEFAULT_FLOAT_PRECISION_ASSERTION = 4 # Decimal places for numbers shown in the "Fact"
    DEFAULT_FINAL_ANSWER_PRECISION = 4 # DP for the final answer LLM should provide
    DEFAULT_FALSE_RESULT_MIN_DEVIATION = Decimal('0.0001')
    VERIFICATION_TOLERANCE = Decimal('0.0001') # Tolerance for comparing LLM's final answer

    @property
    def supported_modalities(self) -> List[str]:
        return ['text']

    def _format_decimal_for_display(self, num: Union[Decimal, float, str], precision: int) -> str:
        if not isinstance(num, Decimal):
            try: num = Decimal(str(num))
            except InvalidOperation:
                self.logger.error(f"Cannot convert '{num}' to Decimal for formatting.")
                raise ValueError(f"Invalid number for display: {num}")

        if precision < 0: precision = 0
        
        # Avoid trailing .0000 if it's effectively an integer after rounding to target precision
        quantizer_check = Decimal('1e-' + str(precision))
        num_rounded_for_check = num.quantize(quantizer_check, rounding=ROUND_HALF_UP)

        if num_rounded_for_check == num_rounded_for_check.to_integral_value(rounding=ROUND_DOWN):
             # If it's a whole number after rounding to target precision, format as integer string
            return str(num_rounded_for_check.to_integral_value(rounding=ROUND_DOWN))
        else:
            # Otherwise, format to the specified precision
            quantizer_format = Decimal('1e-' + str(precision))
            return str(num.quantize(quantizer_format, rounding=ROUND_HALF_UP))


    def _generate_operands_and_base_results(self, op_type: str) -> Optional[Dict[str, Any]]:
        # This is largely the same as in FalseAssertionConditionalQATest
        # It generates op1, op2, actual_result_dec, and claimed_false_result_dec
        op1_val, op2_val = Decimal(0), Decimal(0)
        actual_result_dec = Decimal(0)
        
        int_min, int_max = -1000, 1000 
        float_min, float_max = -1000.0, 1000.0
        operand_display_precision = self.DEFAULT_FLOAT_PRECISION_ASSERTION # For numbers in the initial "Fact"
        
        # Set precision for calculations
        # Python Decimals handle precision well; explicit context change rarely needed if operands are Decimal
        
        # Generate operands
        if random.choice([True, False]):
            op1_val = Decimal(random.randint(int_min, int_max))
        else:
            raw_float1 = random.uniform(float_min, float_max)
            # Quantize raw float to a working precision slightly higher than display to avoid pre-rounding issues
            op1_val = Decimal(str(raw_float1)).quantize(Decimal('1e-' + str(operand_display_precision + 2)), rounding=ROUND_HALF_UP)

        if random.choice([True, False]):
            op2_val = Decimal(random.randint(int_min, int_max))
        else:
            raw_float2 = random.uniform(float_min, float_max)
            op2_val = Decimal(str(raw_float2)).quantize(Decimal('1e-' + str(operand_display_precision + 2)), rounding=ROUND_HALF_UP)

        op_symbol_map = {'addition': '+', 'subtraction': '-', 'multiplication': '*', 'division': '/'}
        op_name_map = {'addition': 'sum', 'subtraction': 'difference', 'multiplication': 'product', 'division': 'quotient'}
        op_symbol = op_symbol_map.get(op_type)
        operation_name = op_name_map.get(op_type)

        if op_type == 'addition': actual_result_dec = op1_val + op2_val
        elif op_type == 'subtraction': actual_result_dec = op1_val - op2_val
        elif op_type == 'multiplication':
            if ('.' in str(op1_val) and '.' in str(op2_val) and (abs(op1_val) > 10 or abs(op2_val) > 10)): # Heuristic to scale
                op1_val /= Decimal('10')
                op2_val /= Decimal('10')
            actual_result_dec = op1_val * op2_val
        elif op_type == 'division':
            max_attempts = 20
            for _ in range(max_attempts):
                if op2_val == Decimal(0) or (_ > 0) : # Ensure op2_val is not zero, regenerate if it was or if retrying
                    op2_val = Decimal(str(random.uniform(0.1, 100.0))).quantize(Decimal('1e-' + str(operand_display_precision + 2)), rounding=ROUND_HALF_UP)
                    if random.choice([True,False]): op2_val *= -1 # allow negative denominators
                    if op2_val == Decimal(0): continue
                
                # Context precision for division needs to be high enough
                current_prec = getcontext().prec
                try:
                    # Ensure precision is high enough for intermediate division
                    getcontext().prec = max(current_prec, self.DEFAULT_FINAL_ANSWER_PRECISION + operand_display_precision + 10)
                    temp_res = op1_val / op2_val
                     # Avoid extreme results that are hard to format or reason about
                    if abs(temp_res) > Decimal('1e9') or (abs(temp_res) < Decimal('1e-7') and temp_res != 0):
                        if _ < max_attempts -1: continue
                        else: # Last attempt, try to make op1 smaller if op2 is small
                            op1_val = op1_val / Decimal('100')
                            temp_res = op1_val / op2_val # Recalculate

                    actual_result_dec = temp_res
                    break
                finally:
                    getcontext().prec = current_prec # Restore
            else:
                self.logger.warning(f"Division generation failed for {op1_val}/{op2_val}. Returning None.")
                return None # Failed to generate valid division
        else:
            self.logger.error(f"Unknown operation type: {op_type}")
            return None

        # Generate Claimed False Result (similar to FalseAssertionConditionalQATest)
        claimed_false_result_dec = Decimal(0)
        attempts = 0
        min_dev_abs = self.DEFAULT_FALSE_RESULT_MIN_DEVIATION
        
        # Scale minimum deviation if actual result is non-zero float
        scaled_min_dev = max(min_dev_abs, abs(actual_result_dec * Decimal('0.005'))) if actual_result_dec != 0 and '.' in str(actual_result_dec) else max(min_dev_abs, Decimal('0.1'))


        while attempts < 50:
            error_magnitude = max(scaled_min_dev, abs(actual_result_dec * Decimal(random.uniform(0.05, 0.3))) if actual_result_dec !=0 else scaled_min_dev * Decimal(random.uniform(1,5)))
            offset = error_magnitude * Decimal(random.choice([-1, 1]))
            claimed_false_result_dec = actual_result_dec + offset
            
            # Ensure formatted strings are different and values are different enough
            formatted_actual = self._format_decimal_for_display(actual_result_dec, self.DEFAULT_FLOAT_PRECISION_ASSERTION)
            formatted_false = self._format_decimal_for_display(claimed_false_result_dec, self.DEFAULT_FLOAT_PRECISION_ASSERTION)

            if formatted_false != formatted_actual and abs(claimed_false_result_dec - actual_result_dec) >= min_dev_abs:
                break
            attempts += 1
        else: # Fallback
            self.logger.debug("Using fallback for distinct false result.")
            claimed_false_result_dec = actual_result_dec + (scaled_min_dev * Decimal('1.5') * Decimal(random.choice([-1,1])))
            if self._format_decimal_for_display(claimed_false_result_dec, self.DEFAULT_FLOAT_PRECISION_ASSERTION) == self._format_decimal_for_display(actual_result_dec, self.DEFAULT_FLOAT_PRECISION_ASSERTION):
                 claimed_false_result_dec = actual_result_dec + Decimal('0.7654') # Another fallback

        return {
            "op1_val": op1_val, "op2_val": op2_val, "op_symbol": op_symbol, "operation_name": operation_name,
            "actual_result_dec": actual_result_dec,
            "claimed_false_result_dec": claimed_false_result_dec
        }

    def generate(self, count: int = 1, **kwargs) -> Iterator[QAItem]:
        operations = ['addition', 'subtraction', 'multiplication', 'division']
        for _ in range(count):
            op_type = random.choice(operations)
            
            calc_data = self._generate_operands_and_base_results(op_type)
            if calc_data is None: continue

            op1_str = self._format_decimal_for_display(calc_data['op1_val'], self.DEFAULT_FLOAT_PRECISION_ASSERTION)
            op2_str = self._format_decimal_for_display(calc_data['op2_val'], self.DEFAULT_FLOAT_PRECISION_ASSERTION)
            op_symbol = calc_data['op_symbol']
            operation_name = calc_data['operation_name']
            claimed_false_result_str = self._format_decimal_for_display(calc_data['claimed_false_result_dec'], self.DEFAULT_FLOAT_PRECISION_ASSERTION)
            
            actual_correct_intermediate_result_dec = calc_data['actual_result_dec']

            # Secondary operation: Subtract 1 from the *actual correct intermediate result*
            final_true_answer_dec = actual_correct_intermediate_result_dec - Decimal('1')
            # Format this final true answer to the desired precision for QAItem.answer
            final_true_answer_str = self._format_decimal_for_display(final_true_answer_dec, self.DEFAULT_FINAL_ANSWER_PRECISION)

            false_fact_statement = f"Statement: {op1_str} {op_symbol} {op2_str} = {claimed_false_result_str}."

            question_text = (
                f"You are given the following information:\n"
                f"{false_fact_statement}\n\n" # The "Fact"
                f"Using the result from the statement, what is its value after subtracting 1?\n" # "result from the statement"
                "Provide your final numerical answer within <answer></answer> tags. "
                f"Format your answer to {self.DEFAULT_FINAL_ANSWER_PRECISION} decimal places if it is not an integer."
            )
            
            item_id = f"{self.name}-{op_type}-{uuid.uuid4().hex[:8]}"
            
            yield QAItem(
                id=item_id,
                question=question_text,
                answer=final_true_answer_str, # LLM should arrive at this if it corrects the "Fact"
                skill_coefficient = 3,
                modality='text',
                metadata={
                    'operation': op_type,
                    'operand1_str': op1_str,
                    'operand2_str': op2_str,
                    'operator_symbol': op_symbol,
                    'claimed_false_result_in_statement': claimed_false_result_str,
                    'actual_intermediate_result_str': self._format_decimal_for_display(actual_correct_intermediate_result_dec, self.DEFAULT_FINAL_ANSWER_PRECISION),
                    'secondary_operation_desc': f"Subtract 1 from the true {operation_name}",
                    'expected_final_answer_precision': self.DEFAULT_FINAL_ANSWER_PRECISION,
                    'output_format_instruction': "<answer>NUMBER</answer>"
                },
                verification_fn=self._verify_numerical_calculation_answer 
            )

    @staticmethod
    def _verify_numerical_calculation_answer(expected_answer_str: str,
                                             provided_llm_output: str,
                                             qa_item: QAItem) -> bool:
        logger = getattr(qa_item, 'logger', logging.getLogger("ImplicitCorrectionMathQATest.VerificationFn"))
        
        match = re.search(r"<answer>([-\d.]+)</answer>", provided_llm_output.strip(), re.IGNORECASE)
        if not match:
            # Try a more lenient regex if the strict one fails, just in case of minor LLM format variations
            match_lenient = re.search(r"([-\d.]+)", provided_llm_output)
            if match_lenient:
                 extracted_num_str = match_lenient.group(1)
                 logger.debug(f"VFY {qa_item.id}: Used lenient regex. Extracted '{extracted_num_str}'. Raw LLM: '{provided_llm_output[:100]}'")
            else:
                logger.warning(f"VFY {qa_item.id}: Could not extract number even with lenient regex. Raw LLM: '{provided_llm_output[:100]}'")
                return False
        else:
            extracted_num_str = match.group(1)

        try:
            llm_answer_dec = Decimal(extracted_num_str)
            expected_answer_dec = Decimal(expected_answer_str)
        except InvalidOperation:
            logger.warning(f"VFY {qa_item.id}: Invalid number format in extracted LLM answer ('{extracted_num_str}') or expected ('{expected_answer_str}').")
            return False

        tolerance = getattr(ImplicitCorrectionMathQATest, 'VERIFICATION_TOLERANCE', Decimal('0.0001'))
        
        is_correct = abs(llm_answer_dec - expected_answer_dec) <= tolerance
        
        if is_correct:
            logger.debug( # Log successful verification at DEBUG level
                       f"Implicit Correction VFY PASSED for {qa_item.id}. "
                       f"Exp: '{expected_answer_dec}', LLM: '{llm_answer_dec}'. "
                       f"Diff: {abs(llm_answer_dec - expected_answer_dec):.4f}. Tol: {tolerance}.") # Using .4f for float formatting
        else:
            # Log failed verification at INFO level, clearly indicating failure
            diff_val = abs(llm_answer_dec - expected_answer_dec)
            logger.info( # Changed from WARNING to INFO
                       f"Implicit Correction VFY FAILED for {qa_item.id}. "
                       f"Exp: '{expected_answer_dec}', LLM: '{llm_answer_dec}'. "
                       f"Diff: {diff_val:.4f}. Tol: {tolerance}. " # Using .4f for float formatting
                       f"Metadata: Operation='{qa_item.metadata.get('operation')}', "
                       f"ClaimedFalse='{qa_item.metadata.get('claimed_false_result_in_statement')}', "
                       f"ActualIntermediate='{qa_item.metadata.get('actual_intermediate_result_str')}'"
            )
        return is_correct