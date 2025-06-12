# KnitSpace: Automated LLM Testing Harness

KnitSpace is an automated testing harness for evaluating and comparing Large Language Models (LLMs) across diverse tasks. It helps assess LLM performance in areas like problem-solving, knowledge retrieval, coding, and safety.

## Key Features

*   **Multi-LLM Support**: Integrates with providers like OpenAI, Google, Cohere, Mistral, etc.
*   **Diverse Test Suite**: Includes mathematical reasoning, coding challenges, instruction following, knowledge tests (e.g., MMLU), long-context understanding, text obfuscation, and safety evaluations.
*   **Elo Rating System**: Uses a sophisticated Elo system considering task difficulty and response "action cost" for nuanced performance assessment.
*   **Secure Code Execution**: Employs Docker for safe execution of LLM-generated code.
*   **Text Obfuscation**: Tests reasoning with transformed questions (e.g., character mapping).
*   **Interactive Review**: Offers a web interface to inspect test results.
*   **Extensible**: Designed for adding new LLM providers and test types.

## Core Components

KnitSpace is organized into several core components:

*   **`knit_space/models.py`**: Handles all LLM interactions.
    *   Defines an abstract base `Model` class for a consistent LLM interface.
    *   Contains specific provider implementations (e.g., `OpenAIModel`, `GeminiModel`) for API communication.
    *   Includes `ProviderInterface` and `MultiModelInterface` to manage and use models from different providers.

*   **`knit_space/tests/`**: Contains all test definitions and generation logic.
    *   `base.py`: Defines core testing structures:
        *   `QAItem`: A dataclass for a single question-answer item (including question, answer, skill coefficient, modality, verification logic).
        *   `AbstractQATest`: The base class for all test generators. Specific tests inherit from this and implement `generate()` to yield `QAItem`s.
        *   `TestRegistry`: Facilitates discovery and organization of test classes.
    *   Specific test files (e.g., `basic_math_tests.py`, `coding_tests.py`): Implement diverse tests by subclassing `AbstractQATest` and generating `QAItem` instances tailored to different evaluation scenarios.

*   **`knit_space/marker.py`**: Responsible for scoring and evaluating LLM performance.
    *   Collects results (the `QAItem`, LLM's answer, verification outcome).
    *   Calculates statistics (attempted, correct, failed).
    *   Implements an Elo rating system considering `QAItem.skill_coefficient` and an "action trajectory S-value" (a cognitive cost metric derived from a GPT-2 model) for nuanced scoring.
    *   Launches a Flask-based web server (`launch_review_server()`) for interactive review of detailed test results.

*   **`knit_space/utils/code_executor.py`**: Provides secure execution for LLM-generated code.
    *   Uses Docker containers to sandbox Python and JavaScript code.
    *   The `CodeExecutor` class takes LLM code, language, test cases (input/output pairs), and hints (e.g., function names).
    *   Dynamically prepares runner scripts that execute the LLM's code within Docker, capturing results.

*   **`knit_space/obscurers/`**: Contains tools for text transformation to create challenging tests.
    *   `char_obfuscator.py`: Provides `CharObfuscator` to replace characters (e.g., English letters with Greek symbols) based on a map. The map is provided to the LLM to test its ability to de-obfuscate.

*   **`verify-auto.py`** (in the root directory): The main script to orchestrate test runs.
    *   Configures the LLM provider and model.
    *   Selects test classes from `knit_space.tests`.
    *   Runs tests, collects responses, and uses `Marker` for evaluation and Elo scoring.
    *   Launches the web review server.

## Setup

Follow these steps to set up KnitSpace:

### 1. Prerequisites

*   **Python**: Python 3.8+ recommended.
*   **Docker**: Required for coding tests using `CodeExecutor`. Ensure Docker daemon is running.
*   **Git**: For cloning the repository.

### 2. Installation

1.  **Clone**: `git clone <repository_url> && cd knitspace`
2.  **Virtual Environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scriptsctivate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install requests numpy torch transformers Flask Pillow
    ```
    Install Python SDKs for desired LLM providers (e.g., `pip install google-generativeai openai`). *A `requirements.txt` will be added later.*

### 3. Configuration

*   **API Keys**: Set environment variables for LLM providers (e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`).
*   **Cloudflare**: If using Cloudflare, also set `CLOUDFLARE_ACCOUNT_ID`.
*   **Docker**: Ensure Docker is running and accessible.

## Running Tests

The primary script for evaluations is `verify-auto.py`.

### Using `verify-auto.py`

1.  **Ensure Setup is Complete**: Dependencies installed and API keys set.
2.  **Configure `verify-auto.py`**:
    *   **LLM Provider & Model**: Modify `models.get_provider(...)` and the `model_name` in `inference()` calls.
    *   **Test Cases**: Edit the `test_cases` list to include desired test classes from `knit_space.tests`.
3.  **Run**: `python verify-auto.py`
4.  **View Results**:
    *   Console logs progress and final Elo score.
    *   A Flask web server (usually `http://localhost:8000`) launches for detailed review.

*(Note: `time.sleep(10)` in `verify-auto.py` is for API rate limits; adjust as needed.)*

### Inspecting Test Data with `QA-test.py`

To view test questions/answers without running LLMs:
1.  Configure `QA-test.py` by editing its `test` list.
2.  Run: `python QA-test.py`

## Extending the Harness

KnitSpace is designed for extension:

*   **Adding New LLM Providers**:
    1.  Create a new class in `knit_space/models.py` inheriting from `knit_space.models.Model`.
    2.  Implement abstract methods for client initialization and inference.
    3.  Update `PROVIDER_CLASS_MAP` and `KNOWN_MODELS_INFO` in `knit_space/models.py`.

*   **Adding New Tests**:
    1.  Create a new test class in `knit_space/tests/` inheriting from `knit_space.tests.base.AbstractQATest`.
    2.  Implement the `generate()` method to yield `QAItem` instances.
    3.  Optionally use `@register_test()` decorator.
