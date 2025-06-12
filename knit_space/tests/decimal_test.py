import random
import uuid
import re # For parsing the answer
import math # For math.gcd

try:
    import mpmath
except ImportError:
    print("ERROR: mpmath library is not installed. NthDecimalDigitQATest cannot function.")
    pass

from typing import Any, Dict, Iterator, Optional, Tuple

# Assuming base.py is in the same directory or accessible in Python path
from .base import AbstractQATest, QAItem, register_test

@register_test('math_high_precision', 'real_numbers', 'nth_digit') # Removed 'constants' tag
class NthDecimalDigitQATest(AbstractQATest):
    DEFAULT_MAX_N = 8000
    MPMATH_PRECISION_BUFFER = 15

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_n = self.config.get("max_n_digit", self.DEFAULT_MAX_N)

        if 'mpmath' not in globals():
            self.logger.error("mpmath library is not available. NthDecimalDigitQATest cannot generate items.")
            self._generators_active = False
            return
        self._generators_active = True

        # Generators:
        # Path A: Direct non-terminating fraction
        # Path B: Sqrt of a positive non-terminating fraction
        # Path C: Sqrt of a positive integer (non-perfect square)
        self.number_generators = [
            self._generate_direct_fraction_data,
            self._generate_sqrt_of_fraction_data,
            self._generate_sqrt_of_integer_data,
        ]


    def _get_denominator_for_non_terminating_fraction(self) -> int:
        # (Same as before)
        while True:
            q_val = random.randint(3, 1000)
            temp_q = q_val
            while temp_q > 0 and temp_q % 2 == 0:
                temp_q //= 2
            while temp_q > 0 and temp_q % 5 == 0:
                temp_q //= 5
            if temp_q > 1:
                return q_val

    def _generate_non_terminating_fraction_params(self) -> Dict:
        # (Same as before)
        q_original = self._get_denominator_for_non_terminating_fraction()
        p_original = random.randint(3, 1000)
        while p_original % q_original == 0 and p_original >= q_original:
            p_original = random.randint(3, 1000)

        common_divisor = math.gcd(p_original, q_original)
        p_reduced = p_original // common_divisor
        q_reduced = q_original // common_divisor

        return {
            'original_p': p_original, 'original_q': q_original,
            'reduced_p': p_reduced, 'reduced_q': q_reduced
        }

    def _generate_direct_fraction_data(self) -> Dict:
        # (Same as before)
        fraction_params = self._generate_non_terminating_fraction_params()
        p_orig, q_orig = fraction_params['original_p'], fraction_params['original_q']
        p_red, q_red = fraction_params['reduced_p'], fraction_params['reduced_q']

        display_desc = f"the fraction {p_orig}/{q_orig}"
        calc_desc = display_desc
        if p_orig != p_red or q_orig != q_red:
            calc_desc += f" (which simplifies to {p_red}/{q_red})"

        return {
            "type": "fraction",
            "params": {'numerator': p_red, 'denominator': q_red},
            "display_description": display_desc,
            "calculation_value_description": calc_desc,
        }

    def _generate_sqrt_of_fraction_data(self) -> Dict:
        # (Same as before)
        fraction_params = self._generate_non_terminating_fraction_params()
        p_orig, q_orig = fraction_params['original_p'], fraction_params['original_q']
        p_red, q_red = fraction_params['reduced_p'], fraction_params['reduced_q']

        display_desc = f"the square root of ({p_orig}/{q_orig})"
        calc_desc_base = f"the fraction {p_orig}/{q_orig}"
        if p_orig != p_red or q_orig != q_red:
            calc_desc_base += f" (simplifies to {p_red}/{q_red})"

        return {
            "type": "sqrt_of_fraction",
            "params": {'numerator': p_red, 'denominator': q_red},
            "display_description": display_desc,
            "calculation_value_description": f"the square root of ({calc_desc_base})",
        }

    def _generate_sqrt_of_integer_data(self) -> Dict:
        radicand = random.randint(2, 2000)
        if 'mpmath' in globals():
            with mpmath.workprec(50): # Temp precision for check
                sqrt_val = mpmath.sqrt(radicand)
                # Correct way to check if mpf is integer:
                while sqrt_val % 1 == 0: # It's a perfect square
                    radicand = random.randint(2, 2000)
                    sqrt_val = mpmath.sqrt(radicand) # Re-calculate for the new radicand

        description = f"the square root of {radicand}"
        return {
            "type": "sqrt_integer",
            "params": {'radicand': radicand},
            "display_description": description,
            "calculation_value_description": description,
        }

    def _get_nth_digit_with_mpmath(self, number_type: str, params: Dict, n_digit_pos: int) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if 'mpmath' not in globals(): return None, None, "mpmath library not loaded."

        integer_part_len_estimate = 10
        val_for_int_part_est = None

        # Temporarily increase precision for accurate integer part estimation
        with mpmath.workdps(n_digit_pos + integer_part_len_estimate + self.MPMATH_PRECISION_BUFFER + 20): # Extra buffer for estimation
            if number_type == "fraction":
                val_for_int_part_est = mpmath.mpf(params['numerator']) / mpmath.mpf(params['denominator'])
            elif number_type == "sqrt_integer":
                val_for_int_part_est = mpmath.sqrt(params['radicand'])
            elif number_type == "sqrt_of_fraction":
                num = mpmath.mpf(params['numerator'])
                den = mpmath.mpf(params['denominator'])
                if den == 0: return None, None, "Denominator zero in sqrt_of_fraction param estimate."
                frac_val = num / den
                if frac_val < 0: return None, None, "Negative fraction in sqrt_of_fraction param estimate."
                val_for_int_part_est = mpmath.sqrt(frac_val)
            # Pi and e removed

        if val_for_int_part_est is not None:
            try:
                # Ensure enough precision for nstr to correctly represent the integer part
                with mpmath.workdps(max(20, integer_part_len_estimate + 10)): # Ensure good precision for nstr itself
                    temp_str = mpmath.nstr(val_for_int_part_est, n=mpmath.inf if val_for_int_part_est != 0 else 1).split('.')[0]
                if 'e' in temp_str.lower():
                    parts_sci = temp_str.lower().split('e')
                    if len(parts_sci) == 2:
                        try:
                            exponent = int(parts_sci[1])
                            integer_part_len_estimate = max(1, exponent + 1)
                        except ValueError: integer_part_len_estimate = 50
                    else: integer_part_len_estimate = 50
                else:
                    integer_part_len_estimate = len(temp_str.lstrip('-'))
            except Exception as e_est:
                self.logger.debug(f"Error during integer part length estimation: {e_est}")
                # Keep default integer_part_len_estimate

        required_dps = n_digit_pos + integer_part_len_estimate + self.MPMATH_PRECISION_BUFFER

        try:
            with mpmath.workdps(required_dps):
                value = None
                if number_type == "fraction":
                    value = mpmath.mpf(params['numerator']) / mpmath.mpf(params['denominator'])
                elif number_type == "sqrt_integer":
                    radicand_mpf = mpmath.mpf(params['radicand'])
                    if radicand_mpf < 0: return None, None, "Cannot take square root of a negative integer."
                    value = mpmath.sqrt(radicand_mpf)
                elif number_type == "sqrt_of_fraction":
                    num = mpmath.mpf(params['numerator'])
                    den = mpmath.mpf(params['denominator'])
                    if den == 0: return None, None, "Denominator cannot be zero in sqrt_of_fraction calculation."
                    frac_val = num / den
                    if frac_val < 0: return None, None, "Cannot take square root of a negative fraction value."
                    value = mpmath.sqrt(frac_val)
                # Pi and e cases removed
                else:
                    return None, None, f"Unknown number type: {number_type}"

                string_output_decimal_places = n_digit_pos + 5
                value_str = mpmath.nstr(value, n=string_output_decimal_places, strip_zeros=False)

                parts = value_str.split('.')
                if len(parts) < 2:
                    if n_digit_pos > 0: return value_str, '0', None
                    return value_str, None, "Value is integer-like, no decimal part to extract from as expected."

                decimal_part = parts[1]
                if n_digit_pos > len(decimal_part):
                    self.logger.debug(
                        f"Nth digit ({n_digit_pos}) requested beyond generated decimal string ('{decimal_part}', len {len(decimal_part)}) "
                        f"for {number_type} {params}. Assuming '0'."
                    )
                    return value_str, '0', None

                nth_digit = decimal_part[n_digit_pos - 1]
                return value_str, nth_digit, None

        except Exception as e:
            self.logger.error(f"Error in _get_nth_digit_with_mpmath for {number_type}, params {params}, N={n_digit_pos}: {e}", exc_info=True)
            return None, None, str(e)

    @staticmethod
    def _get_ordinal(n: int) -> str:
        # (Same as before)
        if 11 <= n <= 13: return f"{n}th"
        suffixes = {1: 'st', 2: 'nd', 3: 'rd'}
        return f"{n}{suffixes.get(n % 10, 'th')}"

    def generate(self, count: int = 2, **kwargs) -> Iterator[QAItem]:
        if not self._generators_active or not self.number_generators:
            self.logger.error("NthDecimalDigitQATest cannot generate: mpmath likely missing or no generators configured.")
            return

        for _ in range(count):
            # Randomly select one of the available generator functions
            generator_func = random.choice(self.number_generators)
            number_data_template = generator_func()

            n_digit_pos = random.randint(1, self.max_n)

            full_decimal_debug, correct_digit_str, error_msg = self._get_nth_digit_with_mpmath(
                number_data_template['type'],
                number_data_template['params'],
                n_digit_pos
            )

            if error_msg or correct_digit_str is None:
                self.logger.warning(
                    f"Could not generate QAItem for {number_data_template.get('calculation_value_description', 'N/A')} "
                    f"(N={n_digit_pos}, Type={number_data_template['type']}): {error_msg}. Skipping."
                )
                continue

            ordinal_n = self._get_ordinal(n_digit_pos)
            question_text = (
                f"What is the {ordinal_n} digit after the decimal point of {number_data_template['display_description']}?\n"
                "Provide your answer as a single digit within <answer></answer> tags. For example: <answer>7</answer>"
            )
            item_id = f"{self.name}-{uuid.uuid4().hex[:8]}"

            yield QAItem(
                id=item_id,
                question=question_text,
                answer=str(correct_digit_str),
                modality='text',
                skill_coefficient = 2,
                metadata={
                    'number_display_description': number_data_template['display_description'],
                    'number_calculation_description': number_data_template.get('calculation_value_description', number_data_template['display_description']),
                    'number_type': number_data_template['type'],
                    'number_params': number_data_template['params'],
                    'N_digit_position': n_digit_pos,
                    'max_n_setting': self.max_n,
                    '_debug_mpmath_computed_value_snippet': full_decimal_debug[:100] if full_decimal_debug else None
                },
                verification_fn=self._verify_digit_answer
            )

    @staticmethod
    def _verify_digit_answer(expected_answer_digit: str, provided_answer_str: str, qa_item: QAItem) -> bool:
        # (Same as before)
        match = re.fullmatch(r'<answer>(\d)</answer>', provided_answer_str.strip())
        if not match:
            return False
        provided_digit = match.group(1)
        return provided_digit == expected_answer_digit