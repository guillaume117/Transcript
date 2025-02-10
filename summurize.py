from transformers import pipeline

def load_text_from_file(filename: str) -> str:
    """
    Loads text from a file.
    
    :param filename: Path to the input text file.
    :return: The content of the file as a string.
    """
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
    return text

def split_text(text: str, max_chars: int = 2000) -> list:
    """
    Splits text into chunks of maximum `max_chars` length to avoid token limit issues.
    
    :param text: The input text to be split.
    :param max_chars: Maximum number of characters per chunk.
    :return: A list of text chunks.
    """
    chunks = []
    while len(text) > max_chars:
        split_idx = text[:max_chars].rfind(". ") + 1  # Split at the last period to avoid cutting sentences.
        if split_idx == 0:  # No period found within limit.
            split_idx = max_chars
        chunks.append(text[:split_idx].strip())
        text = text[split_idx:]
    chunks.append(text.strip())  # Add the remaining text.
    return chunks

def generate_summary_from_file(input_file: str, output_file: str = "summary.txt") -> None:
    """
    Generates a summary from a text file and saves the result.
    
    :param input_file: Path to the input text file.
    :param output_file: Path to save the generated summary.
    """
    # Load text from file
    text = load_text_from_file(input_file)
    
    # Split text into manageable chunks
    text_chunks = split_text(text)
    
    # Initialize the summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)  # Uses GPU if available
    
    # Generate summary for each chunk
    summaries = []
    for i, chunk in enumerate(text_chunks):
        try:
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            print(f"Error summarizing segment {i + 1}: {e}")
    
    # Combine all summaries
    final_summary = " ".join(summaries)
    
    # Save the summary to a file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(final_summary)
    
    print(f"Summary saved to {output_file}")

# Input file (existing transcription)
input_file = "transcription.txt"

# Generate the summary
generate_summary_from_file(input_file)
