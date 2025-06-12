# knit_space/tests/coding_problem_defs.py
from typing import List, Dict, Any, NamedTuple, Optional

class TestCase(NamedTuple):
    input: Any
    expected_output: Any
    description: Optional[str] = None

class CodingProblem(NamedTuple):
    id: str
    description: str
    hints: Dict[str, Dict[str, Any]] # e.g., {"python": {"function_name": "solve"}}
    test_cases: List[TestCase]
    supported_languages: List[str]

ANAGRAM_PROBLEM = CodingProblem(
    id="anagram_checker",
    description="Write a function that takes two strings, s1 and s2, and returns true if s2 is an anagram of s1, and false otherwise.",
    hints={
        "python": {"function_name": "is_anagram"},
        "javascript": {"function_name": "isAnagram"},
    },
    test_cases=[
        TestCase(input=("listen", "silent"), expected_output=True),
        TestCase(input=("rat", "car"), expected_output=False),
        TestCase(input=("", ""), expected_output=True),
    ],
    supported_languages=["python", "javascript"]
)

FIBONACCI_PROBLEM = CodingProblem(
    id="fibonacci_sequence",
    description="Write a function that takes an integer n and returns the nth Fibonacci number. Assume F(0)=0 and F(1)=1.",
    hints={
        "python": {"function_name": "fibonacci"},
        "javascript": {"function_name": "getFibonacci"},
        "java": {"function_name": "fib", "class_name": "Solution"}
    },
    test_cases=[
        TestCase(input=(0,), expected_output=0),
        TestCase(input=(1,), expected_output=1),
        TestCase(input=(2,), expected_output=1),
        TestCase(input=(10,), expected_output=55),
    ],
    supported_languages=["python", "javascript"] # Add "java" once its runner is ready
)

ALL_CODING_PROBLEMS = [
    ANAGRAM_PROBLEM,
    FIBONACCI_PROBLEM,
]