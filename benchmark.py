import os
import time
import argparse
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import tiktoken
from openai import OpenAI

# Default prompt
DEFAULT_PROMPT = ("Imagine you are planning a week-long vacation to a place you've never visited before. "
                  "Describe the destination, including its main attractions and cultural highlights. "
                  "What activities would you prioritize during your visit? Additionally, explain how you would prepare for the trip, "
                  "including any specific items you would pack and any research you would conduct beforehand. "
                  "Finally, discuss how you would balance relaxation and adventure during your vacation.")

# Function to measure time to first token and response time
def benchmark_model(client, model_name, prompt):
    start_time = time.time()
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    # Initialize variables to measure time and gather response
    time_to_first_token = None
    response_time_start = None
    response_time_end = None
    full_response = ""
    num_chunks = 0
    chunk_times = []
    chunk_tokens = []

    # Stream the response
    for chunk in response:
        if time_to_first_token is None:
            time_to_first_token = time.time() - start_time
            response_time_start = time.time()
            previous_chunk_time = response_time_start
            # Skip the first chunk's content
            continue
        current_chunk_time = time.time()
        full_response += chunk.choices[0].delta.content or ""
        chunk_duration = current_chunk_time - previous_chunk_time
        chunk_times.append(chunk_duration)
        chunk_tokens.append(len(chunk.choices[0].delta.content or ""))
        response_time_end = current_chunk_time
        previous_chunk_time = current_chunk_time
        num_chunks += 1

    # Calculate response time
    response_time = response_time_end - response_time_start

    # Tokenize the full response using tiktoken
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except:
        encoding = tiktoken.encoding_for_model('gpt-4')
    
    num_tokens = len(encoding.encode(full_response))
    prompt_tokens = len(encoding.encode(prompt))

    # Calculate tokens per second
    tokens_per_second = num_tokens / response_time if response_time > 0 else float('inf')
    prompt_tokens_per_second = prompt_tokens / time_to_first_token if time_to_first_token > 0 else float('inf')
    avg_tokens_per_chunk = sum(chunk_tokens) / num_chunks if num_chunks > 0 else float('inf')
    avg_time_between_chunks = sum(chunk_times) / len(chunk_times) if len(chunk_times) > 0 else float('inf')

    # Return the benchmark results
    return {
        "time_to_first_token": time_to_first_token,
        "prompt_tokens_per_second": prompt_tokens_per_second,
        "tokens_per_second": tokens_per_second,
        "num_response_tokens": num_tokens,
        "avg_tokens_per_chunk": avg_tokens_per_chunk,
        "avg_time_between_chunks": avg_time_between_chunks
    }

def write_results(model_name, results):
    file_exists = os.path.isfile("output.csv")
    with open("output.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "Model Name", "Time To First Token", "Prompt Tok/s", "Response Tok/s",
                "Num Response Tokens", "Avg Tokens per Chunk", "Avg Time Between Chunks"
            ])
        for result in results:
            writer.writerow([
                model_name,
                f"{result['time_to_first_token']:.2f}",
                f"{result['prompt_tokens_per_second']:.2f}",
                f"{result['tokens_per_second']:.2f}",
                result['num_response_tokens'],
                f"{result['avg_tokens_per_chunk']:.2f}",
                f"{result['avg_time_between_chunks']:.2f}"
            ])

def calculate_model_ranks(df):
    medians = df.groupby('Model Name').median().reset_index()
    sorted_medians = medians.sort_values(by='Response Tok/s', ascending=True)
    return sorted_medians['Model Name'].tolist()

def generate_plots(csv_file):
    df = pd.read_csv(csv_file)
    
    # Calculate the ranks and order the models
    model_order = calculate_model_ranks(df)
    
    # Boxplot for Time To First Token
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Model Name", y="Time To First Token", data=df, order=model_order)
    plt.title("Time To First Token by Model")
    plt.xticks(rotation=15)
    plt.savefig("time_to_first_token_boxplot.png")
    plt.close()

    # Boxplot for Prompt Tok/s
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Model Name", y="Prompt Tok/s", data=df, order=model_order)
    plt.title("Prompt Tokens per Second by Model")
    plt.xticks(rotation=15)
    plt.savefig("prompt_tokens_per_second_boxplot.png")
    plt.close()

    # Boxplot for Response Tok/s
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Model Name", y="Response Tok/s", data=df, order=model_order)
    plt.title("Response Tokens per Second by Model")
    plt.xticks(rotation=15)
    plt.savefig("response_tokens_per_second_boxplot.png")
    plt.close()

    # Boxplot for Number of Response Tokens
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Model Name", y="Num Response Tokens", data=df, order=model_order)
    plt.title("Number of Response Tokens by Model")
    plt.xticks(rotation=15)
    plt.savefig("num_response_tokens_boxplot.png")
    plt.close()

    # Boxplot for Average Tokens per Chunk
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Model Name", y="Avg Tokens per Chunk", data=df, order=model_order)
    plt.title("Average Tokens per Chunk by Model")
    plt.xticks(rotation=15)
    plt.savefig("avg_tokens_per_chunk_boxplot.png")
    plt.close()

    # Boxplot for Average Time Between Chunks
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Model Name", y="Avg Time Between Chunks", data=df, order=model_order)
    plt.title("Average Time Between Chunks by Model")
    plt.xticks(rotation=15)
    plt.savefig("avg_time_between_chunks_boxplot.png")
    plt.close()

def generate_markdown_summary(csv_file):
    df = pd.read_csv(csv_file)
    
    # Calculate medians and IQR for each model and variable
    summary = df.groupby('Model Name').agg(
        time_to_first_token_median=pd.NamedAgg(column='Time To First Token', aggfunc='median'),
        time_to_first_token_iqr=pd.NamedAgg(column='Time To First Token', aggfunc=lambda x: x.quantile(0.75) - x.quantile(0.25)),
        prompt_tok_s_median=pd.NamedAgg(column='Prompt Tok/s', aggfunc='median'),
        prompt_tok_s_iqr=pd.NamedAgg(column='Prompt Tok/s', aggfunc=lambda x: x.quantile(0.75) - x.quantile(0.25)),
        response_tok_s_median=pd.NamedAgg(column='Response Tok/s', aggfunc='median'),
        response_tok_s_iqr=pd.NamedAgg(column='Response Tok/s', aggfunc=lambda x: x.quantile(0.75) - x.quantile(0.25)),
        num_response_tokens_median=pd.NamedAgg(column='Num Response Tokens', aggfunc='median'),
        num_response_tokens_iqr=pd.NamedAgg(column='Num Response Tokens', aggfunc=lambda x: x.quantile(0.75) - x.quantile(0.25)),
        avg_tokens_per_chunk_median=pd.NamedAgg(column='Avg Tokens per Chunk', aggfunc='median'),
        avg_tokens_per_chunk_iqr=pd.NamedAgg(column='Avg Tokens per Chunk', aggfunc=lambda x: x.quantile(0.75) - x.quantile(0.25)),
        avg_time_between_chunks_median=pd.NamedAgg(column='Avg Time Between Chunks', aggfunc='median'),
        avg_time_between_chunks_iqr=pd.NamedAgg(column='Avg Time Between Chunks', aggfunc=lambda x: x.quantile(0.75) - x.quantile(0.25))
    ).reset_index()
    
    # Sort the summary by the same order as the plots
    model_order = calculate_model_ranks(df)
    summary['Model Name'] = pd.Categorical(summary['Model Name'], categories=model_order, ordered=True)
    summary = summary.sort_values('Model Name')
    
    with open("output.md", "w") as md_file:
        md_file.write("# Model Performance Summary\n\n")
        md_file.write("| Model | Time To First Token | Prompt Tok/s | Response Tok/s | Num Response Tokens | Avg Tokens per Chunk | Avg Time Between Chunks |\n")
        md_file.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        
        for _, row in summary.iterrows():
            md_file.write(f"| {row['Model Name']} | {row['time_to_first_token_median']:.2f} +/- {row['time_to_first_token_iqr']:.2f} | "
                          f"{row['prompt_tok_s_median']:.2f} +/- {row['prompt_tok_s_iqr']:.2f} | "
                          f"{row['response_tok_s_median']:.2f} +/- {row['response_tok_s_iqr']:.2f} | "
                          f"{row['num_response_tokens_median']:.2f} +/- {row['num_response_tokens_iqr']:.2f} | "
                          f"{row['avg_tokens_per_chunk_median']:.2f} +/- {row['avg_tokens_per_chunk_iqr']:.2f} | "
                          f"{row['avg_time_between_chunks_median']:.2f} +/- {row['avg_time_between_chunks_iqr']:.2f} |\n")
        
        md_file.write("\n*Values are presented as median +/- IQR (Interquartile Range). Tokenization of non-OpenAI models is approximate.*\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark OpenAI GPT models.")
    parser.add_argument("--models", "-m", type=str, help="Comma-separated list of model names to benchmark.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="The prompt to send to the models.")
    parser.add_argument("--number", "-n", type=int, default=10, help="Number of times to run the benchmark (default 10).")
    parser.add_argument("--plot", type=str, choices=["yes", "no", "only"], default="yes", help="Generate plots: 'yes' (default), 'no', or 'only'.")
    args = parser.parse_args()

    # Initialize the OpenAI client
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY")
    )

    if args.plot != "only":
        # Calculate total number of steps for the overall progress bar
        models = args.models.split(",")
        total_steps = len(models) * args.number

        # Run the benchmark for each model
        with tqdm(total=total_steps, desc="Overall Progress") as overall_progress:
            for model_name in models:
                results = []
                for _ in tqdm(range(args.number), desc=f"Benchmarking {model_name.strip()}", leave=False):
                    result = benchmark_model(client, model_name.strip(), args.prompt)
                    results.append(result)
                    overall_progress.update(1)

                # Write the results to the CSV file
                write_results(model_name.strip(), results)

        print(f"Results have been written to output.csv")

    if args.plot != "no":
        # Generate the plots
        generate_plots("output.csv")
        print("Plots have been saved as time_to_first_token_boxplot.png, prompt_tokens_per_second_boxplot.png, response_tokens_per_second_boxplot.png, num_response_tokens_boxplot.png, avg_tokens_per_chunk_boxplot.png, and avg_time_between_chunks_boxplot.png.")
        # Generate the markdown summary
        generate_markdown_summary("output.csv")
        print("Markdown summary has been saved as output.md")
