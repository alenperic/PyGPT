import tensorflow as tf
from transformers import GPT2Tokenizer

def load_data(file_path):
    """
    Load chat data from a file.
    Each line in the file should be a message.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def clean_data(lines):
    """
    Basic cleaning of the data.
    This function can be expanded based on data cleaning requirements.
    """
    cleaned_lines = []
    for line in lines:
        # Example: simple cleaning step, more can be added
        cleaned_line = line.replace('\n', ' ').strip()
        cleaned_lines.append(cleaned_line)
    return cleaned_lines

def tokenize_and_encode(lines):
    """
    Tokenize and encode the lines using a pre-trained GPT-2 tokenizer.
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    encoded_lines = [tokenizer.encode(line, add_special_tokens=True) for line in lines]
    return encoded_lines

def preprocess_data(file_path):
    """
    Full preprocessing pipeline.
    """
    # Load data
    lines = load_data(file_path)

    # Clean data
    cleaned_lines = clean_data(lines)

    # Tokenize and encode data
    encoded_lines = tokenize_and_encode(cleaned_lines)

    return encoded_lines

if __name__ == "__main__":
    # Example usage
    file_path = "data/raw_data/conversation_data.txt"
    preprocessed_data = preprocess_data(file_path)
    # You might want to save the preprocessed data to a file or proceed with further processing
