# llm-speed-benchmark

`llm-speed-benchmark` is a simple tool designed to help you evaluate and compare the performance of different large language models (LLMs). Whether you are testing models from OpenAI, Mistral, Groq, or any other provider with an OpenAI-compatible API, `llm-speed-benchmark` is available to make your benchmarking process more efficient and insightful.

## Current Results

View current results in the [results directory](results/README.md).

## Key Features

- **Multi-Provider Compatibility**: Works with any LLM provider offering an OpenAI-compatible API.
- **Comprehensive Metrics**: Measures and reports on multiple performance metrics.
- **Easy to Use**: Simple command-line interface for initiating benchmarks and generating reports.
- **Detailed Visualization**: Generates box and whisker plots for easy comparison of model performance.
- **Markdown Summaries**: Outputs performance summaries in markdown format for easy sharing and documentation.

## Known Issues

- **Tokenization for non-OpenAI models**: Right now, any model that tiktoken is unaware of just defaults to the GPT-4 tokenizer. This makes the results for non-OpenAI models approximate, and not fully accurate.
- **Prompt Tok/s**: The OpenAI API does not provide any way to directly measure how long prompt processing took, so the time is inferred from the time taken to reach the first token of response. The larger the input prompt, the more accurate this number should be.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/coder543/llm-speed-benchmark.git
    cd llm-speed-benchmark
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Set the `OPENAI_API_KEY` environment variable to your API key. If you're using a different provider, set the `OPENAI_BASE_URL` environment variable to point to the provider's API endpoint.
    ```sh
    export OPENAI_API_KEY="your_openai_api_key"
    export OPENAI_BASE_URL="https://your-provider-api-url"
    ```

## Usage

You can run the benchmarking tool with the following command:

```sh
python benchmark.py --models "gpt-4o,gpt-4o-mini"
```

NOTE: This tool will always append to output.csv if it already exists, so you can run the tool against multiple APIs and collect the results into one set of outputs, with the plots and markdown being regenerated at the end of each time the tool is run.

## Command-Line Options

- `--models` (`-m`): Comma-separated list of model names to benchmark.
- `--prompt`: The prompt to send to the models. (Defaults to a predefined prompt)
- `--number` (`-n`): Number of times to run the benchmark for each model (default 10).
- `--plot`: Generate plots. Options: `yes` (default), `no`, `only` (re-generate the plots and markdown from the existing output.csv file).

## Example Usage

### Benchmarking and Generating Plots with 20 samples per model

```sh
python benchmark.py --models "gpt-4o,gpt-4o-mini" -n 20
```

### Regenerating Plots Only:

```sh
python benchmark.py --plot only
```

### Benchmarking Without Generating Plots:

```sh
python benchmark.py --models "gpt-4o,gpt-4o-mini" --plot no
```

## Understanding the Outputs

### CSV File

The results of the benchmarks are saved in output.csv, which contains detailed information about each run, including:

- Model Name
- Time to First Token
- Prompt Tokens per Second
- Response Tokens per Second
- Number of Response Tokens
- Average Tokens per Response Chunk
- Average Time Between Response Chunks

### Plots

llm-speed-benchmark generates several box and whisker plots to visualize the performance metrics of the models:

- `time_to_first_token_boxplot.png`
- `prompt_tokens_per_second_boxplot.png`
- `response_tokens_per_second_boxplot.png`
- `num_response_tokens_boxplot.png`
- `avg_tokens_per_chunk_boxplot.png`
- `avg_time_between_chunks_boxplot.png`

### Markdown Summary

A detailed markdown summary of the performance metrics is saved in `output.md`. This file includes a table with median values and interquartile ranges (IQR) for each metric, making it easy to compare the models' performance.

*Values are presented as median +/- IQR (Interquartile Range). Tokenization of non-OpenAI models is approximate.*