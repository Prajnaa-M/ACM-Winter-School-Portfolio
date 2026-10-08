\# AI-Infused Software Quality Profiler



An interactive static code analysis application that evaluates Python source code using deterministic software quality metrics and heuristic architectural analysis.



\## Overview



The Software Quality Profiler analyzes submitted Python code and generates a structured assessment of its complexity, maintainability, documentation, control-flow structure, and potential code-quality risks.



The application combines deterministic static analysis with an interactive Streamlit interface to provide developers with actionable feedback for improving code quality.



\## Features



\### 📊 Code Metrics



The profiler analyzes:



\- Total lines of code

\- Executable, blank, and documentation lines

\- Cyclomatic complexity estimate

\- Comment and documentation density

\- Maximum control-flow nesting depth

\- Decision-keyword frequency



\### 🔬 Function-Level Analysis



Individual functions are profiled based on:



\- Function length

\- Complexity

\- Nesting depth

\- Documentation presence

\- Risk classification



\### 🛡️ Structural Risk Detection



The application identifies patterns including:



\- Deeply nested control blocks

\- Unguarded I/O operations

\- Manually managed file resources

\- Bare `except:` clauses

\- Nested or inefficient comprehensions

\- Repetitive `.append()` patterns

\- Repeated dictionary lookups

\- Missing `\_\_main\_\_` guards



\### 📈 Maintainability Index



A maintainability score from 0–100 is calculated using penalties based on:



\- Complexity

\- Documentation density

\- Nesting depth

\- Error-handling and resource-management risks



The result is classified into maintainability bands ranging from highly maintainable to critical maintenance risk.



\### 🧠 Architectural Review



The application generates a heuristic structural review containing:



\- Critical structural vulnerabilities

\- Refactoring and maintainability recommendations

\- Maintainability scoring breakdown



The architectural review is explicitly a \*\*heuristic simulation\*\* based on the static-analysis results rather than a call to an external language model.



\## Methodology



The profiler uses lexical and indentation-based analysis of Python source code.



The cyclomatic complexity estimate is calculated as:



`CC = 1 + count(if, for, while, elif, def, except)`



The maintainability score is derived from complexity, documentation, nesting, and resilience penalties and is constrained to a range of 0–100.



\## Technology Stack



\- Python

\- Pandas

\- Streamlit

\- Regular Expressions

\- Static Code Analysis

\- Heuristic Software Quality Metrics



\## Run Locally



Install the dependencies:



```bash

pip install -r requirements.txt

