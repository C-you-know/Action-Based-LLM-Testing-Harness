# knit_space/tests/__init__.py

from .base import create_test_cases, QAItem, AbstractQATest, register_test, test_registry
from .basic_math_tests import MathQATestClass
from .sentence_obs_tests import RandomSentenceObfuscationTest
from .coding_tests import CodingQATestClass

from . import long_context_tests
from .long_context_tests import LongContextWikiBookTest

from .sudoku_test import SudokuValidationQATest

from .chess_memory_tests import ChessMemoryQATest

from .rule_tests import NRulesVectorQATest

from .rule_tests_fake_guide import NRulesVectorFakeGuidanceQATest

from .decimal_test import NthDecimalDigitQATest

from .needle_in_a_haystack import FindUniqueNumberIndexQATest

from .cardinality_test import WikiCharCountQATest

from .real_time_tests import LiveStockPriceQATest

from .mmlu_obfuscated_tests import MMLUObfuscatedQATest

from .mmlu_obfuscated_choice_tests import MMLUObfuscatedChoiceQATest

from .medmcqa_obfuscated_choice_tests import MedMCQAObfuscatedChoiceQATest

from .false_assertion_math_tests import ImplicitCorrectionMathQATest

__all__ = [
    "create_test_cases",
    "QAItem",
    "AbstractQATest",
    "register_test",
    "test_registry",
    "MathQATestClass",
    "RandomSentenceObfuscationTest",
    "CodingQATestClass",
    "LongContextWikiBookTest",
    "SudokuValidationQATest",
    "ChessMemoryQATest",
    "NRulesVectorQATest",
    "NRulesVectorFakeGuidanceQATest",
    "NthDecimalDigitQATest",
    "FindUniqueNumberIndexQATest",
    "WikiCharCountQATest",
    "LiveStockPriceQATest",
    "MMLUObfuscatedQATest",
    "MMLUObfuscatedChoiceQATest",
    "MedMCQAObfuscatedChoiceQATest",
    "ImplicitCorrectionMathQATest",
]