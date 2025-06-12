
# KnitSpace LLM Ranker: Automated LLM Testing Harness

KnitSpace is an automated testing harness designed to evaluate and compare the capabilities of various Large Language Models (LLMs) across a diverse set of tasks. It provides a comprehensive framework for researchers and developers to assess LLM performance in areas such as problem-solving, knowledge retrieval, coding proficiency, and safety.

## 🔑 Key Features

- **Multi-LLM Support**: Integrates with OpenAI, Google, Cohere, Mistral, and more.
- **Diverse Test Suite**: Includes mathematical reasoning, coding tasks, knowledge tests (MMLU), long-context, instruction-following, and obfuscation-based tests.
- **Elo Rating System**: Scores models using task difficulty and a cognitive cost metric ("S-value") for nuanced benchmarking.
- **Secure Code Execution**: Uses Docker containers to safely execute LLM-generated Python/JS code.
- **Text Obfuscation**: Tests reasoning under character-mapped distortions.
- **Interactive Review**: Launch a web-based viewer for test results.
- **Extensible**: Easily add new LLM providers and new types of tests.

---

## 🧱 Core Components

### 📁 `knit_space/models.py`
- Unified interface for all LLM providers.
- Abstract `Model` class + subclasses like `OpenAIModel`, `GeminiModel`, etc.
- Manages API initialization, inference calls, and model metadata.

### 📁 `knit_space/tests/`
- Contains all test definitions.
- `base.py` defines:
  - `QAItem`: A test prompt, answer, and scoring logic.
  - `AbstractQATest`: Base class for all test sets.
  - `TestRegistry`: Auto-discovers test modules.
- Includes test types: math, coding, chess, long-context, MMLU, etc.

### 📁 `knit_space/marker.py`
- Evaluates model responses.
- Uses `QAItem` scoring logic and tracks correctness.
- Implements Elo scoring using both test difficulty and S-value.
- Launches `Flask` server to review test results interactively.

### 📁 `knit_space/utils/code_executor.py`
- Runs Python and JS code from models inside Docker safely.
- Accepts test cases (input/output pairs) for correctness validation.

### 📁 `knit_space/obscurers/`
- Tools for generating challenging input variants.
- `CharObfuscator`: Replaces characters using a bijective map to test reasoning under noise.

### 🐍 `verify-auto.py`
- Main script to run tests.
- Configures model, loads test classes, and executes tests.
- Starts web server for results review.

---

## ⚙️ Setup

### 1. Prerequisites

- Python 3.8+
- Docker (for coding tasks)
- Git

### 2. Installation

```bash
git clone [<repository_url>](https://github.com/C-you-know/Action-Based-LLM-Testing-Harness)
cd KnitSpace-LLM-Ranker

python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)

pip install -r requirements.txt  # Or manually install dependencies
````

### 3. API Key Setup

Set the following environment variables based on the providers you wish to use:

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
export MISTRAL_API_KEY="..."
export COHERE_API_KEY="..."
# Cloudflare-specific
export CLOUDFLARE_API_KEY="..."
export CLOUDFLARE_ACCOUNT_ID="..."
```

---

## 🚀 Running Tests

### Run via `verify-auto.py`

1. Configure:

   * Choose model/provider in `verify-auto.py`
   * Select tests in `test_cases` list
2. Run:

   ```bash
   python verify-auto.py
   ```
3. View:

   * Console logs test stats
   * Web UI opens at `http://localhost:8000`

### Debug Test Inputs (optional)

Use `QA-test.py` to inspect generated test data without invoking an LLM:

```bash
python QA-test.py
```
## 🔌 Extending the Harness

### ➕ Adding New LLM Providers

1. Subclass `Model` in `knit_space/models.py`
2. Implement:

   * `_initialize_client()`
   * `inference(...)`
3. Update:

   * `PROVIDER_CLASS_MAP`
   * `_get_api_key_for_provider()` and optionally `_list_api_models()`

### 🧪 Adding New Test Types

1. Create a new file in `knit_space/tests/`
2. Subclass `AbstractQATest`
3. Implement `generate()` to yield `QAItem`s
4. Optionally register using `@register_test()`

---

## 📦 Install as a Package

You can also install this project as a pip package (once published):

```bash
pip install ks-llm-ranker
```
