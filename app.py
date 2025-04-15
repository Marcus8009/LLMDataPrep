
#!/usr/bin/env python3
import os
import sys
import time
import argparse
import logging
import concurrent.futures
from pathlib import Path
from openai import OpenAI

# =======================
# Configuration Settings
# =======================
DEFAULT_INPUT_FOLDER = r"C:\Users\hp\Documents\GenAI\RAGLLMDataPrep\input"
DEFAULT_OUTPUT_FOLDER = r"C:\Users\hp\Documents\GenAI\RAGLLMDataPrep\output"
DEFAULT_MODEL = "gpt-4o-mini"

def process_file(file_path, output_folder, system_prompt, client, model=DEFAULT_MODEL, max_retries=3, retry_delay=5):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            logging.warning(f"File {file_path} is empty. Skipping.")
            return (file_path, False, "Empty file")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        retries = 0
        response_text = ""
        while retries < max_retries:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    timeout=60
                )
                response_text = response.choices[0].message.content
                break
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}. Retrying in {retry_delay} seconds...")
                retries += 1
                time.sleep(retry_delay)
        if not response_text:
            logging.error(f"Failed to process file after {max_retries} retries: {file_path}")
            return (file_path, False, "Failed after retries")

        original_name = os.path.basename(file_path)
        file_stem, _ = os.path.splitext(original_name)
        output_file_name = f"{file_stem}.txt"
        output_file_path = os.path.join(output_folder, output_file_name)
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(response_text)
        logging.info(f"Processed file {file_path} successfully.")
        return (file_path, True, None)
    except Exception as e:
        logging.error(f"Unexpected error processing file {file_path}: {e}")
        return (file_path, False, str(e))

def main():
    parser = argparse.ArgumentParser(description="Process scraped text files using GPT-4o and output cleaned files for a RAG pipeline.")
    parser.add_argument("--input_folder", default=DEFAULT_INPUT_FOLDER, help="Path to the input folder containing raw .txt files")
    parser.add_argument("--output_folder", default=DEFAULT_OUTPUT_FOLDER, help="Path to the output folder for cleaned .txt files")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of worker threads for processing files")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    max_workers = args.max_workers

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY environment variable not set. Exiting.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    os.makedirs(output_folder, exist_ok=True)

    system_prompt = """
You are a content structuring expert for Retrieval-Augmented Generation (RAG) systems.

Your task is to clean, structure, and prepare scraped web content for ingestion into a RAG pipeline. The raw content may include metadata, inconsistent formatting, missing headings, or noisy data. Your job is to produce clear, structured plain text that matches the natural flow of the original content, while improving readability and organization.

### Guidelines:

1. **Clean Up Noise**
   - Remove non-essential metadata (e.g., word counts, "===" separators, scrapey timestamps).
   - Eliminate empty sections (e.g., "No transcript found").
   - Remove duplicated or repeated lines.

2. **Preserve Valuable Content**
   - Retain all meaningful information, quotes, and insights.
   - Preserve useful data about individuals, companies, tools, or results.
   - If speaker names are present, maintain them clearly (e.g., "Mr. Yamamoto said: ...").

3. **Dynamic Section Headings**
   - Analyze the content to generate appropriate section headings based on the themes present.
   - Use clear Markdown-style headings (e.g., ## Introduction, ## Customer Challenges, ## Solution, ## Results).
   - Adapt headings to match the content—do not enforce a fixed structure.
   - If no natural section divisions exist, maintain a clean and readable flow without forced headings.

4. **Source Metadata**
   - At the very top of the output, always include:
     - File name (without path)
     - Optional: Title or date if available from the scraped content

5. **Readable and Clean Output**
   - Ensure paragraphs are well-formed with clear line breaks.
   - Avoid adding unnecessary symbols or formatting.
   - **Do not fabricate, infer, or add any information not present in the original content.** If certain details are missing or unclear, leave them as is without attempting to fill in the gaps.

6. **Optional: Chunk Awareness**
   - For lengthy content (e.g., over ~1,000 words), consider logical points to split into chunks.
   - Use clear dividers like "--- CHUNK SPLIT ---" if necessary for easier RAG ingestion.

### Output Format Example:

file name: www.microsoft.com#en-us#worklab#enlist-ai-in-your-fight-against-email-overload.txt

## [Dynamic Section Name Based on Content]

[Cleaned and structured body text here...]

## [Next Natural Section Name]

[Body text continues...]

## Conclusion

[Closing summary or remarks from the source]

---

### Notes:
- The goal is to make the content easy to retrieve, reference, and understand in RAG applications.
- Preserve speaker attributions and quotes in a cleaned and readable manner.
- If the content is unstructured or too short for headings, simply clean the text without adding headings.
- **If you encounter information that is unclear or missing, do not attempt to infer or create content.** Only include information that is explicitly present in the original content.
"""

    txt_files = [str(p) for p in Path(input_folder).rglob("*.txt")]
    if not txt_files:
        logging.error(f"No .txt files found in input folder: {input_folder}")
        sys.exit(1)

    start_time = time.time()
    total_files = len(txt_files)
    success_count = 0
    error_list = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_file, file_path, output_folder, system_prompt, client): file_path for file_path in txt_files}
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                file_result, succeeded, error_msg = future.result()
                if succeeded:
                    success_count += 1
                else:
                    error_list.append((file_path, error_msg))
            except Exception as exc:
                logging.error(f"Exception generated for file {file_path}: {exc}")
                error_list.append((file_path, str(exc)))

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Processing complete.")
    logging.info(f"Total files processed: {total_files}")
    logging.info(f"Successfully processed: {success_count}")
    if error_list:
        logging.info(f"Files with errors: {len(error_list)}")
        for fp, err in error_list:
            logging.error(f"File: {fp}, Error: {err}")
    logging.info(f"Total time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
