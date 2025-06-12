# File: sudoku_tests.py (or your chosen filename)

import random
import uuid
import re # Import regular expressions
from typing import List, Iterator, Optional, Dict, Any, Tuple, Callable

# Assuming base.py is in the same directory or accessible in PYTHONPATH
from .base import AbstractQATest, QAItem, register_test

@register_test('puzzle', 'sudoku', 'logic', 'validation')
class SudokuValidationQATest(AbstractQATest):
    """
    Generates Sudoku validation tasks.
    One task will be a valid Sudoku grid, the other an invalid one.
    The LLM should determine if the presented Sudoku grid is valid.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.grid_size = 9

    @property
    def supported_modalities(self) -> List[str]:
        return ['text']

    @staticmethod
    def _extract_answer_content(answer_str: str) -> Optional[str]:
        """Extracts content from <answer>...</answer> tags."""
        if not isinstance(answer_str, str):
            return None
        match = re.search(r"<answer>(.*?)</answer>", answer_str, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _verify_sudoku_answer(expected_answer_full: str,
                              provided_answer_str: str,
                              qa_item: QAItem) -> bool:
        """
        Verifies Sudoku answer by extracting content from <answer> tags.
        """
        expected_content = SudokuValidationQATest._extract_answer_content(expected_answer_full)
        provided_content = SudokuValidationQATest._extract_answer_content(provided_answer_str)

        if expected_content is None:
            # This shouldn't happen if QAItem.answer is set correctly
            qa_item.metadata['verification_error'] = "Could not parse expected_answer_full"
            return False

        if provided_content is None:
            qa_item.metadata['verification_error'] = "Could not parse provided_answer_str for <answer> tag"
            return False
        
        # Case-insensitive comparison
        is_correct = expected_content.lower() == provided_content.lower()
        if not is_correct:
            qa_item.metadata['verification_details'] = f"Expected: '{expected_content}', Got: '{provided_content}'"
        return is_correct

    def _find_empty_cell(self, grid: List[List[int]]) -> Optional[Tuple[int, int]]:
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if grid[r][c] == 0:
                    return (r, c)
        return None

    def _is_safe(self, grid: List[List[int]], row: int, col: int, num: int) -> bool:
        # Check row
        if num in grid[row]:
            return False
        # Check column
        if num in [grid[r][col] for r in range(self.grid_size)]:
            return False
        # Check 3x3 box
        box_start_row, box_start_col = row - row % 3, col - col % 3
        for r in range(box_start_row, box_start_row + 3):
            for c in range(box_start_col, box_start_col + 3):
                if grid[r][c] == num:
                    return False
        return True

    def _solve_sudoku_recursive(self, grid: List[List[int]]) -> bool:
        empty_cell = self._find_empty_cell(grid)
        if not empty_cell:
            return True

        row, col = empty_cell
        nums = list(range(1, self.grid_size + 1))
        random.shuffle(nums)

        for num in nums:
            if self._is_safe(grid, row, col, num):
                grid[row][col] = num
                if self._solve_sudoku_recursive(grid):
                    return True
                grid[row][col] = 0
        return False

    def _generate_filled_sudoku_grid(self) -> Optional[List[List[int]]]:
        grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        if self._solve_sudoku_recursive(grid):
            return grid
        self.logger.error("Failed to generate a fully solved Sudoku grid.")
        return None

    def _make_grid_invalid(self, original_grid: List[List[int]]) -> List[List[int]]:
        grid = [row[:] for row in original_grid]
        attempts = 0
        while attempts < 50:
            choice = random.choice(['row', 'col', 'box'])
            if choice == 'row':
                r = random.randint(0, self.grid_size - 1)
                c1, c2 = random.sample(range(self.grid_size), 2)
                if grid[r][c1] != grid[r][c2]:
                    grid[r][c1] = grid[r][c2]
                    return grid
            elif choice == 'col':
                c = random.randint(0, self.grid_size - 1)
                r1, r2 = random.sample(range(self.grid_size), 2)
                if grid[r1][c] != grid[r2][c]:
                    grid[r1][c] = grid[r2][c]
                    return grid
            else: # box
                box_r_start, box_c_start = random.randint(0, 2) * 3, random.randint(0, 2) * 3
                cells_in_box_coords = []
                for r_offset in range(3):
                    for c_offset in range(3):
                        cells_in_box_coords.append((box_r_start + r_offset, box_c_start + c_offset))
                (r1, c1), (r2, c2) = random.sample(cells_in_box_coords, 2)
                if grid[r1][c1] != grid[r2][c2]:
                    grid[r1][c1] = grid[r2][c2]
                    return grid
            attempts += 1
        
        self.logger.warning("Failed to make grid invalid with preferred method, using simple swap.")
        r1,c1 = random.randint(0,8), random.randint(0,8)
        r2,c2 = random.randint(0,8), random.randint(0,8)
        iter_safety = 0
        while ((r1,c1) == (r2,c2) or grid[r1][c1] == grid[r2][c2]) and iter_safety < 100 :
            r1,c1 = random.randint(0,8), random.randint(0,8)
            r2,c2 = random.randint(0,8), random.randint(0,8)
            iter_safety +=1
        if grid[r1][c1] != grid[r2][c2]:
            grid[r1][c1], grid[r2][c2] = grid[r2][c2], grid[r1][c1]
        else:
             grid[0][0] = grid[0][1] if grid[0][1] != 0 else (grid[0][0] % 9) + 1
        return grid

    def _format_grid_to_string(self, grid: List[List[int]]) -> str:
        puzzle_str = "Sudoku Puzzle (0 or . represents an empty cell):\n"
        for r in range(self.grid_size):
            if r > 0 and r % 3 == 0:
                puzzle_str += "------+-------+------\n"
            row_str = []
            for c in range(self.grid_size):
                if c > 0 and c % 3 == 0:
                    row_str.append("|")
                row_str.append(str(grid[r][c]) if grid[r][c] != 0 else ".")
            puzzle_str += " ".join(row_str) + "\n"
        
        return (
            f"{puzzle_str.strip()}\n\n"
            "Is the above Sudoku puzzle configuration valid (i.e., it follows all Sudoku rules)? "
            "Respond with only '<answer>Yes</answer>' or '<answer>No</answer>'."
        )

    def generate(self,
                 count: int = 5,
                 difficulty: Optional[str] = None,
                 prefix: Optional[str] = None,
                 suffix: Optional[str] = None,
                 text_file: Optional[str] = None,
                 template_vars: Optional[Dict[str, str]] = None,
                 **kwargs) -> Iterator[QAItem]:

        if count == 0:
            return

        items_to_generate = []

        valid_grid = self._generate_filled_sudoku_grid()
        if not valid_grid:
            self.logger.error(f"Could not generate a valid Sudoku for {self.name}. Skipping.")
            return

        question_valid_str = self._format_grid_to_string(valid_grid)
        qa_valid = QAItem(
            id=f"{self.name}-valid-{uuid.uuid4().hex[:8]}",
            question=question_valid_str,
            skill_coefficient = 3,
            answer="<answer>Yes</answer>", # Full expected answer string
            modality='text',
            metadata={'sudoku_type': 'valid', 'difficulty_level': difficulty or 'generated'},
            verification_fn=self._verify_sudoku_answer # Assign the static method
        )
        items_to_generate.append(qa_valid)

        invalid_grid = self._make_grid_invalid(valid_grid)
        question_invalid_str = self._format_grid_to_string(invalid_grid)
        qa_invalid = QAItem(
            id=f"{self.name}-invalid-{uuid.uuid4().hex[:8]}",
            question=question_invalid_str,
            answer="<answer>No</answer>", # Full expected answer string
            skill_coefficient = 3,
            modality='text',
            metadata={'sudoku_type': 'invalid', 'difficulty_level': difficulty or 'generated'},
            verification_fn=self._verify_sudoku_answer # Assign the static method
        )
        items_to_generate.append(qa_invalid)
        
        random.shuffle(items_to_generate)

        generated_count = 0
        for item in items_to_generate:
            if generated_count < count:
                yield item
                generated_count += 1
            else:
                break

# Example Usage (assuming base.py is set up and this file can be run)
if __name__ == '__main__':
    from abc import ABC, abstractmethod
    from dataclasses import dataclass, field
    import logging

    @dataclass
    class QAItem:
        id: str
        question: Any
        answer: Any
        modality: str
        metadata: Dict[str, Any] = field(default_factory=dict)
        verification_fn: Optional[Callable[[Any, Any, 'QAItem'], bool]] = field(default=None, compare=False, repr=False)
        
        def verify(self, provided_answer: Any) -> bool:
            if self.verification_fn:
                return self.verification_fn(self.answer, provided_answer, self)
            return self.answer == provided_answer


    class AbstractQATest(ABC):
        def __init__(self, config: Optional[Dict[str, Any]] = None):
            self.config = config or {}
            self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
            logging.basicConfig(level=logging.DEBUG)
            self._stats = {'total_generated': 0, 'errors': 0}
        
        @property
        def name(self) -> str: return self.__class__.__name__
        @property
        def supported_modalities(self) -> List[str]: return ['text']
        @abstractmethod
        def generate(self, count: int = 1, **kwargs) -> Iterator[QAItem]: pass
        def build_question(self, base_question: str, **kwargs) -> str: return base_question

    def register_test(*tags):
        def decorator(cls): return cls
        return decorator

    sudoku_test_gen = SudokuValidationQATest()
    
    print("\n--- Generating 2 Sudoku QA items with verification_fn ---")
    for qa_item in sudoku_test_gen.generate(count=2):
        print(f"ID: {qa_item.id}")
        print(f"Question:\n{qa_item.question}")
        print(f"Expected Answer (full): {qa_item.answer}")
        print(f"Metadata: {qa_item.metadata}")

        # Test verification_fn
        # Case 1: Correct provided answer
        correct_llm_response = f"Some preamble... {qa_item.answer} ...some postamble"
        is_verified_correct = qa_item.verify(correct_llm_response)
        print(f"Verification (correct response '{SudokuValidationQATest._extract_answer_content(correct_llm_response)}'): {is_verified_correct}")
        assert is_verified_correct

        # Case 2: Incorrect provided answer (opposite)
        wrong_core_answer = "No" if "Yes" in qa_item.answer else "Yes"
        incorrect_llm_response = f"<answer>{wrong_core_answer}</answer>"
        is_verified_incorrect = qa_item.verify(incorrect_llm_response)
        print(f"Verification (incorrect response '{SudokuValidationQATest._extract_answer_content(incorrect_llm_response)}'): {not is_verified_incorrect}")
        assert not is_verified_incorrect
        
        # Case 3: Malformed or no answer tag
        malformed_llm_response = "The Sudoku is indeed valid."
        is_verified_malformed = qa_item.verify(malformed_llm_response)
        print(f"Verification (malformed response '{malformed_llm_response}'): {not is_verified_malformed}")
        assert not is_verified_malformed
        print(f"Metadata after malformed: {qa_item.metadata}") # Check for error messages

        print("-" * 20)