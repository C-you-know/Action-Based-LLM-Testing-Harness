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
## ⚙️ Total Action-Based Evaluation

We model text generation as a physical system traversing high-dimensional semantic space. Each step contributes kinetic and potential energy, and the **Total Action** quantifies overall generation effort:

![LLM Physics Equations](https://latex.codecogs.com/svg.image?\dpi{150}&space;\begin{align*}\vec{v}_t&=H_{t+1}-H_t\\m_t&=\text{base\_FLOPs}+t\cdot\text{growth}\\L_t&=\frac{1}{2}m_t\|\vec{v}_t\|^2+\alpha(-\log%20P_{t+1})\\S&=\sum_{t=1}^{T-1}L_t\\\text{ELO}&=\frac{1}{\log_{10}(S)}\end{align*})

📄 [Read the full design doc here (PDF)](./Chinmay_LLMHarnes.pdf)

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
