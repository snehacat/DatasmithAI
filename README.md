# DataSmith AI - Intelligent Dataset Generation Platform

**Status**: Phase 0 Complete ✅ - Foundations Ready

---

## Overview

DataSmith AI is a systematic dataset generation platform built with strong foundations:
- **Data Contract**: Universal JSON schema for the entire pipeline
- **LLM Wrapper**: Reusable Ollama interface with proper error handling
- **Orchestrator**: Pipeline runner with save/resume capability

Phase 0 provides the infrastructure that all future agents will use.

---

## Project Structure

```
DataSmith_AI/
├── core/                    # Phase 0 foundations
│   ├── __init__.py
│   ├── dataset_spec.py      # Data contract
│   ├── llm_wrapper.py       # LLM wrapper
│   └── orchestrator.py      # Pipeline orchestrator
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_dataset_spec.py
│   ├── test_llm_wrapper.py
│   └── test_orchestrator.py
├── venv/                    # Virtual environment
├── .gitignore
├── requirements.txt
├── run.bat
└── README.md                # This file
```

---

## Installation

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate
.\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Ollama and pull model
# Download from: https://ollama.com/download
ollama pull qwen2.5:3b
```

---

## Running Tests

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_dataset_spec.py -v
```

**Expected**: 19/19 tests passing (4 skipped without Ollama)

---

## Phase 0 Components

### 1. Data Contract (`core/dataset_spec.py`)
- Universal JSON schema for entire pipeline
- Provenance tracking (source_url on every record)
- Confidence scores throughout
- Execution history for save/resume
- **Tests**: 8/8 passing

### 2. LLM Wrapper (`core/llm_wrapper.py`)
- Reusable Ollama interface
- JSON-only output with validation
- Total timeout limit (60s default)
- Maximum 1 retry with error feedback
- Connection error detection
- **Tests**: 5/5 passing (4 skipped)

### 3. Orchestrator (`core/orchestrator.py`)
- Runs agents in sequence
- Saves spec after each agent
- Resume from any step
- Skip conditions and error handling
- **Tests**: 6/6 passing

---

## Next Steps: Phase 1

Phase 1 will build agents using these foundations:

1. **Requirement Understanding Agent**
   - Uses LLMWrapper
   - Writes to spec.requirement
   - Handles user input

2. **Domain Detection Agent**
   - Reads spec.requirement
   - Adds domain classification
   - Uses shared data contract

All future agents will follow the same pattern.

---

## Key Principles

1. ✅ **No Code Duplication** - All agents use LLMWrapper
2. ✅ **Unified Data Contract** - One spec flows through pipeline
3. ✅ **Provenance** - source_url on all data
4. ✅ **Testable** - Comprehensive test coverage
5. ✅ **Resumable** - Save/resume at any step
6. ✅ **Generic** - Works for any domain

---

## Dependencies

- Python 3.8+
- Ollama (local LLM)
- qwen2.5:3b model
- pydantic (data validation)
- pytest (testing)

---

## Documentation

- `core/dataset_spec.py` - Data contract with inline docs
- `core/llm_wrapper.py` - LLM wrapper with inline docs
- `core/orchestrator.py` - Orchestrator with inline docs
- `LLM_WRAPPER_IMPROVEMENTS.md` - Recent LLM wrapper improvements

---

**Last Updated**: 2026-10-03  
**Phase**: 0 Complete, 1 Pending
