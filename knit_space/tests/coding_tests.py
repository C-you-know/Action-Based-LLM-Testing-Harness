# knit_space/tests/coding_tests.py
from .base import AbstractQATest, QAItem, register_test
from .coding_problem_defs import ALL_CODING_PROBLEMS, CodingProblem
from ..utils.code_executor import CodeExecutor, CodeExecutionResult # Adjusted import
import uuid
import re
from typing import Any, Dict, Iterator, List, Optional

def extract_code_from_llm_response(response: str, language: str) -> Optional[str]:
    # Try language-specific backticks first
    pattern_specific = rf"```{language}\s*([\s\S]*?)\s*```"
    match_specific = re.search(pattern_specific, response, re.IGNORECASE)
    if match_specific:
        return match_specific.group(1).strip()

    # Try generic backticks if language-specific not found or language is not typical for backticks
    pattern_generic = r"```(?:\w*\n)?([\s\S]*?)```" # Allow optional language hint
    match_generic = re.search(pattern_generic, response)
    if match_generic:
        return match_generic.group(1).strip()
    
    # Fallback: if no backticks, consider the whole response if it looks like code (very heuristic)
    # This might need to be more language-specific or be removed if too unreliable
    # For Python:
    if language.lower() == "python":
        if "def " in response or "class " in response:
            # Try to remove preceding text if it's not part of the code
            lines = response.splitlines()
            code_lines = []
            in_code_block = False
            for line in lines:
                if line.strip().startswith("def ") or line.strip().startswith("class "):
                    in_code_block = True
                if in_code_block:
                    code_lines.append(line)
            if code_lines:
                return "\n".join(code_lines).strip()

    # If all else fails, and as a last resort, return the whole response
    # but this is likely to fail unless the LLM is very well-behaved.
    # Consider returning None if no clear code block found.
    # For now, let's try returning the whole thing if it's short and contains keywords
    keywords = ["def ", "function ", "class ", "public static", "const ", "var ", "let "]
    if any(kw in response for kw in keywords) and len(response.splitlines()) < 50: # arbitrary line limit
         return response.strip()

    return None # Prefer None if unsure

@register_test('coding', 'problem_solving')
class CodingQATestClass(AbstractQATest):
    @property
    def supported_modalities(self) -> List[str]:
        return ['text']

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config if config else {})
        self.problems = self.config.get("problems", ALL_CODING_PROBLEMS)
        self.code_executor = CodeExecutor(timeout_seconds=self.config.get("timeout", 15))

    def generate(self,
                 count: int = 1,
                 difficulty: Optional[str] = None,
                 prefix: Optional[str] = None,
                 suffix: Optional[str] = None,
                 languages: Optional[List[str]] = None,
                 problem_ids: Optional[List[str]] = None,
                 **kwargs) -> Iterator[QAItem]:

        problems_to_run = self.problems
        if problem_ids:
            problems_to_run = [p for p in self.problems if p.id in problem_ids]

        for _ in range(count):
            for problem in problems_to_run:
                if difficulty and problem.hints.get("difficulty_level") != difficulty:
                    continue

                target_languages = languages if languages else problem.supported_languages
                for lang in target_languages:
                    lang_hints = problem.hints.get(lang, {})
                    func_name_hint = lang_hints.get("function_name", "solve")
                    class_name_hint = lang_hints.get("class_name")

                    prompt_intro = f"Please write a complete and executable code solution in {lang} for the following problem:\n"
                    prompt_problem_desc = problem.description

                    prompt_func_signature = f"The primary entry point should be a function named '{func_name_hint}'."
                    if class_name_hint:
                        prompt_func_signature = (
                            f"The solution should be within a class named '{class_name_hint}', "
                            f"and the primary entry point method within that class should be named '{func_name_hint}'."
                        )
                    
                    prompt_io_spec = (
                        "The function/method should accept inputs and produce outputs strictly according to the problem description. "
                        "Ensure your code defines the function and exports it using 'module.exports = { functionName };' where functionName is the required function. " 
                        "Do not include any example usage or main execution block in your code submission, only the function/class definition."
                    )

                    question_base = f"{prompt_intro}\n{prompt_problem_desc}\n\n{prompt_func_signature}\n{prompt_io_spec}"

                    final_question_str = self.build_question(
                        base_question=question_base,
                        prefix=prefix,
                        suffix=suffix
                    )

                    yield QAItem(
                        id=f"{self.name}-{problem.id}-{lang}-{uuid.uuid4().hex[:8]}",
                        question=final_question_str,
                        answer=True, # Expected verification result is True (code passes tests)
                        skill_coefficient = 2,
                        modality='text',
                        metadata={
                            'problem_id': problem.id,
                            'language': lang,
                            'problem_definition': problem, # Pass the whole problem object
                            'expected_function_name': func_name_hint,
                            'expected_class_name': class_name_hint
                        },
                        verification_fn=self._verify_code_execution
                    )

    @staticmethod
    def _verify_code_execution(expected_answer_bool: bool, # Will be True
                               llm_response: str,
                               qa_item: QAItem) -> bool: # Pass the whole QAItem
        
        metadata = qa_item.metadata
        problem: 'CodingProblem' = metadata['problem_definition'] # Type hint if defined elsewhere
        language: str = metadata['language']
        
        # This instance of CodeExecutor will be used for this verification
        # If CodeExecutor needs config from CodingQATestClass, it could be passed via metadata or re-instantiated
        # For simplicity, let's assume a default executor or that it's managed by the test instance.
        # We'll use a fresh one here to demonstrate, but you might share one from `self` if CodingQATestClass isn't static
        temp_executor_for_static_method = CodeExecutor()


        extracted_code = extract_code_from_llm_response(llm_response, language)
        if not extracted_code:
            print(f"VERIFY: Failed to extract code for {problem.id} in {language} from response.")
            # qa_item.set_verification_details({"passed": False, "reason": "Code extraction failed", "extracted_code": None})
            return False

        problem_lang_hints = problem.hints.get(language, {})

        execution_results: List[CodeExecutionResult] = temp_executor_for_static_method.run_tests_for_problem(
            llm_code=extracted_code,
            language=language,
            test_cases=problem.test_cases,
            problem_hints=problem_lang_hints # Pass language-specific hints from the problem definition
        )

        all_passed = True
        passed_count = 0
        results_summary = []

        # print(f"--- Verifying Code for {problem.id} ({language}) ---")
        # print(f"Extracted Code:\n{extracted_code[:300]}...\n")
        for i, result in enumerate(execution_results):
            tc = problem.test_cases[i]
            summary_item = {
                "test_case_idx": i,
                "input": tc.input,
                "expected_output": tc.expected_output,
                "passed": result.passed,
                "actual_output": result.output,
                "error": result.error,
                "exit_code": result.exit_code,
                "raw_stdout": result.raw_stdout,
                "raw_stderr": result.raw_stderr,
            }
            results_summary.append(summary_item)
            if result.passed:
                passed_count +=1
            else:
                all_passed = False
        
        # Store detailed results in the QAItem if your QAItem supports it
        # qa_item.set_verification_details({
        # "passed_overall": all_passed,
        # "passed_count": passed_count,
        # "total_tests": len(problem.test_cases),
        # "extracted_code": extracted_code,
        # "results_per_case": results_summary
        # })

        print(f"--- Overall Result for {problem.id} ({language}): {'PASSED' if all_passed else 'FAILED'} ({passed_count}/{len(problem.test_cases)}) ---")
        if not all_passed:
            for i, res_sum in enumerate(results_summary):
                if not res_sum["passed"]:
                    print(f"  Failed TC {i+1}: Input: {res_sum['input']}, Expected: {res_sum['expected_output']}, Got: {res_sum['actual_output']}, Error: {res_sum['error']}")
        # print("---------------------------------------\n")
        return all_passed