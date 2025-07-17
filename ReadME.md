[![PyPI version](https://badge.fury.io/py/ks-llm-ranker.svg)](https://pypi.org/project/ks-llm-ranker/)
[![Python Versions](https://img.shields.io/pypi/pyversions/ks-llm-ranker.svg)](https://pypi.org/project/ks-llm-ranker/)

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
