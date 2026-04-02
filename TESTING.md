# Testing Guide for airada_poc

## Overview
This project includes comprehensive unit and functional tests for the core modules in `src/`. The test suite covers:

- **Data Pipeline** (`src/data/`): Download, preprocessing, and ingestion modules
- **LLM Factory** (`src/providers/llm_factory.py`): OpenAI client and embedding function creation
- **Prompts** (`src/prompts/prompts.py`): System instruction management

## Test Statistics
- **Total Tests**: 44
- **Test Coverage**: Core pipeline and provider modules
- **Framework**: pytest with mocking support

## Running Tests

### Prerequisites
Install test dependencies:
```bash
pip install -e ".[test]"
```

### Execute All Tests
```bash
pytest tests/
```

### Run with Verbose Output
```bash
pytest tests/ -v
```

### Run Specific Test Module
```bash
# Data pipeline tests
pytest tests/test_data_01_download.py -v
pytest tests/test_data_02_preprocess.py -v
pytest tests/test_data_03_ingest_data.py -v

# LLM Factory tests
pytest tests/test_llm_factory.py -v

# Prompts tests
pytest tests/test_prompts.py -v
```

### Run Specific Test Class or Function
```bash
pytest tests/test_data_01_download.py::TestDownloadData::test_download_data_uses_stream -v
```

### Generate Coverage Report
```bash
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
```

## Test Modules

### `test_data_01_download.py` (7 tests)
Tests for data download module:
- URL validation
- HTTP request handling with mocking
- Chunked response processing
- Timeout configuration
- Stream flag usage
- Error handling

### `test_data_02_preprocess.py` (8 tests)
Tests for data preprocessing:
- Regex pattern matching for keywords (agents, LLMs, RAG)
- Case-insensitive matching
- Keyword filtering logic
- Null value handling
- Process function existence

### `test_data_03_ingest_data.py` (14 tests)
Tests for ChromaDB ingestion:
- Document building from title + abstract
- Metadata extraction with required fields
- CSV loading validation
- Required columns checking
- Null row removal
- Batch processing for large datasets

### `test_llm_factory.py` (6 tests)
Tests for LLM factory:
- OpenAI client creation
- API key handling
- Embedding function initialization
- Model constants validation

### `test_prompts.py` (9 tests)
Tests for system prompts:
- Instruction string generation
- Content validation (personality, scope, guardrails)
- Formatting and structure
- Deterministic output

## Fixtures
Common test fixtures in `conftest.py`:
- `temp_dir`: Temporary directory for test files
- `sample_papers_csv`: Pre-populated CSV with test data
- `sample_papers_dataframe`: Pandas DataFrame for testing
- `mock_openai_client`: Mocked OpenAI client
- `mock_embedding_function`: Mocked embedding function
- `mock_chroma_collection`: Mocked ChromaDB collection

## Next Steps

### Add Integration Tests
```bash
pytest tests/ -m integration
```

### Add Performance Benchmarks
```bash
pytest tests/ --benchmark-only
```

### Continuous Integration
Configure in `.github/workflows/ci.yml` to run tests on push/PR.

## Notes
- Tests use mocking extensively to avoid external API calls
- Some tests for `02_preprocess.py` focus on regex logic (core testable part)
- Full end-to-end tests (download → preprocess → ingest) should be run separately
- Numeric-prefixed module names (`01_download_data.py`) are loaded via `importlib.import_module()`
