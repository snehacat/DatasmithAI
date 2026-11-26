# DataSmith AI - Intelligent Dataset Generation Platform

**Status**: Phase 2 Complete ✅ - Requirement Analysis and Planning

---

## Overview

DataSmith AI is a systematic dataset generation platform built with strong foundations:
- **Data Contract**: Universal JSON schema for the entire pipeline
- **LLM Wrapper**: Reusable Ollama interface with proper error handling
- **Orchestrator**: Pipeline runner with save/resume capability

Phase 0 provides the infrastructure, and Phase 1 adds the Requirement Analyzer.

---

## Project Structure

```
DatasmithAI/
├── agents/
│   ├── requirement_analyzer.py
│   ├── planning_agent.py   # Source and search planning
│   └── source_catalog.py   # Editable provider hints
├── core/                    # Phase 0 foundations
│   ├── __init__.py
│   ├── dataset_spec.py      # Data contract
│   ├── llm_wrapper.py       # LLM wrapper
│   ├── orchestrator.py      # Pipeline orchestrator
│   └── run_manager.py       # Run persistence and pruning
├── main.py                  # CLI entry point
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_dataset_spec.py
│   ├── test_llm_wrapper.py
│   ├── test_orchestrator.py
│   └── test_planning_agent.py
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

**Real suite count**: 446 tests passing, 13 skipped without Ollama

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

## Phase 1

### 1. Requirement Analyzer — DONE
- Uses LLMWrapper with rule-based fallback
- Extracts geography, time range, expected size, output format, and constraints
- Performs scope rejection and typo suggestions
- Writes clarification questions and provenance to the dataset spec

Future agents will follow the same pattern.

## Phase 2

### Planning Agent — DONE
- Selects source types and search queries from the editable `agents/source_catalog.py` hints
- Applies geography, time, feature, and access constraints to the candidate sources
- Produces a plan only; it does not invent source URLs or retrieve data
- Uses the LLM only for optional search-query and reasoning refinement

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

- Python 3.9+
- Uses tuple[...] type hints and PEP 585 generic syntax
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
