# RAG LLM Data Preparation Tool

A Python utility for processing and structuring raw text files to prepare them for Retrieval-Augmented Generation (RAG) systems using OpenAI's models.

## Overview

This tool takes scraped or raw text content, processes it through a Large Language Model (LLM), and outputs clean, structured text files optimized for RAG applications. It handles multiple files concurrently and includes error handling and retry mechanisms.

## Features

- **Batch Processing**: Process multiple text files in parallel
- **LLM-Powered Structuring**: Uses OpenAI models to intelligently structure content
- **Error Handling**: Built-in retry mechanism for API failures
- **Configurable**: Command-line options for customizing behavior
- **Detailed Logging**: Comprehensive logging of processing progress

## Requirements

- Python 3.6+
- OpenAI Python SDK
- OpenAI API key

## Installation

1. Clone this repository or download the script
2. Install the required dependencies:

```bash
pip install openai
```

3. Set up your OpenAI API key as an environment variable:

```bash
# On Windows
set OPENAI_API_KEY=your_api_key_here

# On macOS/Linux
export OPENAI_API_KEY=your_api_key_here
```

## Usage

Basic usage with default settings:

```bash
python ragllm_data_prep.py
```

Custom configuration:

```bash
python ragllm_data_prep.py --input_folder "path/to/input" --output_folder "path/to/output" --max_workers 8
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input_folder` | Path to folder containing raw text files | `C:\Users\hp\Documents\GenAI\RAGLLMDataPrep\input` |
| `--output_folder` | Path to output folder for processed files | `C:\Users\hp\Documents\GenAI\RAGLLMDataPrep\output` |
| `--max_workers` | Maximum number of concurrent processing threads | `4` |

## How It Works

1. The script scans the input folder for `.txt` files
2. Each file is sent to OpenAI's API with a specialized system prompt
3. The LLM cleans and structures the content following detailed guidelines:
   - Removes noise and metadata
   - Preserves valuable content
   - Adds appropriate section headings
   - Maintains source metadata
   - Formats text for readability
4. Processed files are saved to the output folder with the same base filename

## System Prompt

The system prompt instructs the LLM to:

- Clean up noise (metadata, empty sections, duplicates)
- Preserve valuable content (information, quotes, speaker attributions)
- Add dynamic section headings based on content themes
- Include source metadata at the top
- Format text for readability
- Optionally suggest chunk splits for lengthy content

## Error Handling

The script includes:

- Multiple retry attempts for API failures
- Detailed error logging
- Summary of successful and failed processing attempts

## Example Output

```
file name: example-document.txt

## Introduction

This is the cleaned and structured content with proper paragraph breaks and formatting.

## Key Findings

The important information is preserved while noise has been removed.

## Conclusion

The original insights are maintained in a format optimized for RAG systems.
```

## License

[Your license information here]

## Contributing

[Your contribution guidelines here]
