# Job Advertisement Discrimination Checker

This project reads a `.xlsx` file of job advertisements, uses a local Ollama model (`qwen3:4b`) to detect discrimination or unfairness, converts the model output into the teacher's fixed label system, and writes the final Excel result to `analysis_result.xlsx`.

## Project Structure

- `app.py`: main batch-processing script
- `requirements.txt`: minimal Python dependencies
- `README.md`: local setup and usage instructions

## Features

- Reads an Excel file with:
  - one row per job post
  - one ground-truth label column such as `Discrimination`
- Automatically detects:
  - the ground-truth column
  - the job post text column
- Calls local Ollama through `http://localhost:11434/api/chat`
- Uses model `qwen3:4b` by default
- Requires strict JSON from the model and includes a repair pass if parsing fails
- Normalizes both prediction and ground truth into sets before comparison
- Writes a new Excel file with these added columns:
  - `llm_status`
  - `predicted_categories`
  - `analysis`
  - `llm_raw_output`
  - `complete_match`
- Prints the required summary in the terminal

## Allowed Categories

Only these labels are used, with exact casing:

- `Age`
- `Gender`
- `Origin`
- `Location`
- `Mobility`
- `PPE`
- `Standing`
- `Salary`
- `Role clarity`

## Important Label Rules Implemented

- `Origin`: nationality, ethnic origin, native speaker requirements, country/region-origin preference
- `Standing`: appearance or image requirements such as `well-presented` or `good appearance`
- `Role clarity`: vague duties such as `various tasks`, `as needed`, `whatever is needed`
- `Salary`: salary is mentioned but vague, such as `to be defined`, `to be agreed`, `depends on candidate`
- Missing salary can optionally be treated as `Salary`, but this is disabled by default

## Prerequisites

### 1. Python

Use Python 3.10+ on macOS.

### 2. Ollama

Install and run Ollama locally. 



If Ollama is not already installed, see the official setup instructions:

```bash
https://ollama.com
```

### 3. Download the model

Pull the required model locally:

```bash
ollama pull qwen3:4b
```

You can verify it is available:

```bash
ollama list
```

## Installation

Create a virtual environment if you want, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Input Excel Requirements

Your input `.xlsx` should contain:

- one column with job advertisement text
- one ground-truth label column

The script does **not** assume fixed column names.

### Ground-truth column auto-detection

Priority order:

- `Discrimination`
- similar names such as:
  - `expected_categories`
  - `labels`
  - `ground_truth`
  - `target`

### Job text column auto-detection

The script looks for names like:

- `job post`
- `job_post`
- `description`
- `text`
- `content`
- `advertisement`

## Ground-Truth Parsing Rules

The ground-truth label column can contain:

- empty values
- `none`
- `no`
- `n/a`
- `-`

These are treated as **no discrimination**.

It also supports multiple labels separated by:

- comma
- semicolon
- newline
- pipe
- slash, for example `Origin / Location`

All labels are normalized into a set before comparison.

## Run the Batch Checker

Example:

```bash
python app.py --excel "/path/to/input.xlsx"
```

This will:

- create `analysis_result.xlsx` in the current directory
- print the summary in the terminal

You can also choose a custom output path:

```bash
python app.py --excel "/path/to/input.xlsx" --output "/path/to/analysis_result.xlsx"
```



### `OLLAMA_MODEL`

Override the default model:

```bash
export OLLAMA_MODEL="qwen3:4b"
```

### `FLAG_MISSING_SALARY`

Controls whether a job ad with **no salary mentioned at all** should be treated as `Salary`.

Default is off:

```bash
export FLAG_MISSING_SALARY=0
```

Enable it if your teacher wants missing salary to count:

```bash
export FLAG_MISSING_SALARY=1
```

## Output Excel Columns

The script writes the original data plus these new columns:

- `llm_status`: `success` or `failed`
- `predicted_categories`: normalized category list used for comparison
- `analysis`: short explanation or failure reason
- `llm_raw_output`: raw or repaired LLM output for debugging
- `complete_match`: `True` only if predicted category set exactly equals the ground-truth set

## Required Summary Output

The terminal summary includes:

- `Ground truth column used: ...`
- `Complete match percentage: XX.XX%`
- `LLM successful rows: a/b`
- `First failure: ...`
- `Model: ...`

## JSON and Repair Strategy

The script follows this sequence for each row:

1. Call Ollama `/api/chat` with `stream=false`
2. Try to parse the returned content directly as JSON
3. If parsing fails, run one repair call to convert the raw output into valid JSON
4. If that still fails, mark the row as:
   - `llm_status=failed`
   - `analysis=<failure reason>`

## Notes

- The script compares **sets** of labels, not raw strings.
- `complete_match=True` only when the normalized prediction set is exactly the same as the normalized ground-truth set.
- Even when a row fails, the raw output and failure reason are preserved for debugging.
