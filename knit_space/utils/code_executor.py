# knit_space/utils/code_executor.py
import tempfile
import os
import shutil # Not strictly used now, but often useful with tempfile
import json
import subprocess
import pathlib
from typing import List, Dict, Any, NamedTuple, Optional

class TestCase(NamedTuple):
    input: Any
    expected_output: Any
    description: Optional[str] = None

class CodeExecutionResult(NamedTuple):
    passed: bool
    output: Optional[str] = None
    error: Optional[str] = None
    exit_code: Optional[int] = None
    raw_stdout: Optional[str] = None
    raw_stderr: Optional[str] = None

class CodeExecutor:
    def __init__(self, timeout_seconds=15, enable_network=False):
        self.timeout_seconds = timeout_seconds
        self.enable_network_docker_cli_arg = [] if enable_network else ["--network", "none"]

    def _get_docker_image_and_interpreter_cmd(self, language: str, runner_filename_in_container: str):
        if language == "python":
            return "python:3.9-slim", ["python", runner_filename_in_container]
        elif language == "javascript":
            return "node:18-slim", ["node", runner_filename_in_container]
        else:
            raise ValueError(f"Unsupported language for runner execution: {language}")

    def _prepare_runner_script(self, language: str, main_code_filename: str,
                               function_to_call: str,
                               is_class_method: bool = False, class_name: Optional[str] = None) -> str:
        if language == "python":
            runner_code = f"""import sys
import json
import importlib.util
import traceback

solution_module = None
try:
    spec = importlib.util.spec_from_file_location("solution_module", "{main_code_filename}")
    if spec and spec.loader:
        solution_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solution_module)
    else:
        raise ImportError("Could not load solution module spec.")
except Exception as e:
    print(json.dumps({{"error": f"Module import error for '{main_code_filename}': {{str(e)}}", "traceback": traceback.format_exc()}}), file=sys.stderr)
    sys.exit(1)

target_callable = None
try:
    if {is_class_method} and "{class_name}" and {str(bool(class_name)).lower()}:
        SolutionClass = getattr(solution_module, "{class_name}")
        instance = SolutionClass()
        target_callable = getattr(instance, "{function_to_call}")
    else:
        target_callable = getattr(solution_module, "{function_to_call}")
except AttributeError as e:
    err_msg = {{
        "error": f"Attribute error: {{str(e)}} - Could not find function/method.",
        "details": f"Attempted: {class_name}.{function_to_call} (class method: {is_class_method}) or {function_to_call} (function). Check spelling and if defined.",
        "traceback": traceback.format_exc()
    }}
    print(json.dumps(err_msg), file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(json.dumps({{"error": f"Error getting target callable: {{str(e)}}", "traceback": traceback.format_exc()}}), file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    input_data_str = None
    try:
        input_data_str = sys.stdin.read()
        input_args = json.loads(input_data_str)
        if not isinstance(input_args, list):
            input_args = [input_args]
        
        result = target_callable(*input_args)
        print(json.dumps(result), end='')
    except json.JSONDecodeError as e_json:
        err_msg = {{
            "error": f"JSONDecodeError processing stdin: {{str(e_json)}}",
            "stdin_received_prefix": input_data_str[:200] if input_data_str is not None else "Not read or empty",
            "traceback": traceback.format_exc()
        }}
        print(json.dumps(err_msg), file=sys.stderr)
        sys.exit(1)
    except Exception as e_main:
        err_msg = {{
            "error": f"Execution error in runner: {{str(e_main)}}",
            "type": type(e_main).__name__,
            "traceback": traceback.format_exc()
        }}
        print(json.dumps(err_msg), file=sys.stderr)
        sys.exit(1)
"""
            return runner_code.strip()
        elif language == "javascript":
            runner_code = f"""const fs = require('fs');
const path = require('path');
let solutionModule;

try {{
    const solutionPath = path.resolve('/app', '{main_code_filename}');
    solutionModule = require(solutionPath); 
}} catch (e) {{
    process.stderr.write(JSON.stringify({{ 
        error: `Module require error for path '/app/{main_code_filename}': ${{e.message}}`, 
        attemptedPath: path.resolve('/app', '{main_code_filename}'),
        stack: e.stack 
    }}));
    process.exit(1);
}}

let targetCallable;
try {{
    if ({str(is_class_method).lower()} && "{class_name}" && {str(bool(class_name)).lower()}) {{
        const SolutionClass = solutionModule.{class_name};
        if (!SolutionClass) throw new Error(`Class '{class_name}' not found in module.`);
        const instance = new SolutionClass();
        if (typeof instance.{function_to_call} !== 'function') {{
            throw new Error(`Method '{function_to_call}' not found or not a function in class '{class_name}'.`);
        }}
        targetCallable = instance.{function_to_call}.bind(instance);
    }} else {{
        targetCallable = solutionModule.{function_to_call};
    }}
    if (typeof targetCallable !== 'function') {{
        throw new Error(`Function '{function_to_call}' is not a function or not found in module.`);
    }}
}} catch (e) {{
    process.stderr.write(JSON.stringify({{ 
        error: `Error getting target callable: ${{e.message}}`, 
        details: `Attempted: {class_name}.{function_to_call} (class method: {is_class_method}) or {function_to_call} (function).`,
        stack: e.stack 
    }}));
    process.exit(1);
}}

let inputDataStr = "";
try {{
    inputDataStr = fs.readFileSync(0, 'utf-8'); 
    let inputArgs = JSON.parse(inputDataStr);
    if (!Array.isArray(inputArgs)) {{
        inputArgs = [inputArgs];
    }}
    
    const result = targetCallable(...inputArgs);
    process.stdout.write(JSON.stringify(result));
}} catch (e_main) {{
    process.stderr.write(JSON.stringify({{ 
        error: e_main.message, 
        type: e_main.name, 
        stack: e_main.stack,
        stdin_received_prefix: inputDataStr.substring(0,200)
    }}));
    process.exit(1);
}}
"""
            return runner_code.strip()
        raise NotImplementedError(f"Runner script generation not implemented for {language}")

    def run_tests_for_problem(self,
                              llm_code: str,
                              language: str,
                              test_cases: List[TestCase],
                              problem_hints: Dict[str, Any]) -> List[CodeExecutionResult]:
        results = []
        main_code_ext = {"python": "py", "javascript": "js"}.get(language, "txt")
        
        solution_filename_in_container = f"solution.{main_code_ext}" 
        runner_filename_in_container = f"runner.{main_code_ext}"

        function_to_call = problem_hints.get("function_name", "solve")
        is_class_method = "class_name" in problem_hints and bool(problem_hints.get("class_name"))
        class_name = problem_hints.get("class_name")

        with tempfile.TemporaryDirectory() as tmpdir_host_str:
            tmpdir_host_path = pathlib.Path(tmpdir_host_str)

            solution_file_on_host = tmpdir_host_path / solution_filename_in_container
            with open(solution_file_on_host, "w", encoding='utf-8') as f:
                f.write(llm_code)

            runner_script_content = self._prepare_runner_script(
                language, solution_filename_in_container, function_to_call, is_class_method, class_name
            )
            runner_file_on_host = tmpdir_host_path / runner_filename_in_container
            with open(runner_file_on_host, "w", encoding='utf-8') as f:
                f.write(runner_script_content)

            docker_image, interpreter_cmd_parts = self._get_docker_image_and_interpreter_cmd(language, runner_filename_in_container)
            
            for tc_idx, tc in enumerate(test_cases):
                serialized_input = json.dumps(list(tc.input) if isinstance(tc.input, tuple) else tc.input if tc.input is not None else [])
                if isinstance(tc.input, tuple) and len(tc.input) == 1 and tc.input[0] is None:
                    serialized_input = json.dumps([])

                # Minimal debug prefix
                log_prefix = f"CodeExec: [{language} TC {tc_idx+1}]" 

                docker_command = [
                    "docker", "run",
                    "--rm",
                    "-i", 
                    *self.enable_network_docker_cli_arg,
                    "-v", f"{tmpdir_host_path}:/app:ro",
                    "-w", "/app", 
                    docker_image,
                    *interpreter_cmd_parts
                ]
                
                # print(f"{log_prefix} Executing: {' '.join(docker_command)}") # Optional: if you want to see the full docker command
                # print(f"{log_prefix} Input: {serialized_input[:100]}")      # Optional: if you want to see input

                stdout_str = ""
                stderr_str = ""
                exit_code = -1

                try:
                    process = subprocess.run(
                        docker_command,
                        input=serialized_input.encode('utf-8'),
                        capture_output=True,
                        timeout=self.timeout_seconds,
                        check=False
                    )
                    stdout_str = process.stdout.decode('utf-8', errors='replace').strip()
                    stderr_str = process.stderr.decode('utf-8', errors='replace').strip()
                    exit_code = process.returncode
                    
                    # print(f"{log_prefix} Result - ExitCode: {exit_code}, STDOUT: '{stdout_str[:100]}...', STDERR: '{stderr_str[:100]}...'") # Optional

                except subprocess.TimeoutExpired:
                    stderr_str = f"Command timed out after {self.timeout_seconds} seconds."
                    exit_code = -1 
                    # print(f"{log_prefix} TIMEOUT: {stderr_str}") # Optional
                except FileNotFoundError:
                    stderr_str = "Docker command not found. Is Docker installed and in PATH?"
                    exit_code = -1
                    # print(f"{log_prefix} Docker not found: {stderr_str}") # Optional
                except Exception as e:
                    stderr_str = f"Subprocess execution error: {str(e)}"
                    exit_code = -1
                    # print(f"{log_prefix} Subprocess error: {stderr_str}") # Optional
                
                if exit_code == 0 and not stderr_str:
                    try:
                        if not stdout_str:
                            if tc.expected_output is None or tc.expected_output == "":
                                results.append(CodeExecutionResult(passed=True, output="", exit_code=exit_code, raw_stdout="", raw_stderr=""))
                            else:
                                results.append(CodeExecutionResult(passed=False, output="", error="Script produced no stdout, but output was expected.", exit_code=exit_code, raw_stdout="", raw_stderr=""))
                        else:
                            actual_output_val = json.loads(stdout_str)
                            if isinstance(tc.expected_output, bool) and isinstance(actual_output_val, str):
                                if actual_output_val.lower() == 'true': actual_output_val = True
                                elif actual_output_val.lower() == 'false': actual_output_val = False
                            
                            if actual_output_val == tc.expected_output:
                                results.append(CodeExecutionResult(passed=True, output=stdout_str, exit_code=exit_code, raw_stdout=stdout_str, raw_stderr=stderr_str))
                            else:
                                results.append(CodeExecutionResult(passed=False, output=stdout_str, error=f"Output mismatch. Expected: {json.dumps(tc.expected_output)}, Got: {json.dumps(actual_output_val)}", exit_code=exit_code, raw_stdout=stdout_str, raw_stderr=stderr_str))
                    except json.JSONDecodeError:
                        results.append(CodeExecutionResult(passed=False, output=stdout_str, error=f"Output is not valid JSON: '{stdout_str}'", exit_code=exit_code, raw_stdout=stdout_str, raw_stderr=stderr_str))
                else:
                    final_error_msg = stderr_str or f"Non-zero exit code: {exit_code}"
                    if not stderr_str and exit_code != 0 and stdout_str:
                         final_error_msg += f" | Potential stdout: {stdout_str}"
                    results.append(CodeExecutionResult(passed=False, output=stdout_str, error=final_error_msg, exit_code=exit_code, raw_stdout=stdout_str, raw_stderr=stderr_str))
        return results