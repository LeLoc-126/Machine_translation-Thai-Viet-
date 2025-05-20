import sentencepiece as spm
from tqdm import tqdm
import pandas as pd
import os

# Preprocess CSV to text
def preprocess_csv_to_text(csv_file, text_file, thai_column="Thai", vietnamese_column="Viet"):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    if not os.path.exists(text_file):
        print(f"Converting {csv_file} to {text_file}...")
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")
        
        # Check if columns exist
        columns = df.columns
        with open(text_file, 'w', encoding='utf-8') as f:
            if thai_column in columns and vietnamese_column in columns:
                # Write both Thai and Vietnamese sentences
                for _, row in tqdm(df.iterrows(), total=len(df), desc="Writing sentences"):
                    if isinstance(row[thai_column], str):
                        f.write(row[thai_column].strip() + '\n')
                    if isinstance(row[vietnamese_column], str):
                        f.write(row[vietnamese_column].strip() + '\n')
            elif 'text' in columns:
                # Write single text column
                for text in tqdm(df['text'], desc="Writing sentences"):
                    if isinstance(text, str):
                        f.write(text.strip() + '\n')
            else:
                raise ValueError(f"CSV must have '{thai_column}' and '{vietnamese_column}' columns or a 'text' column")
        print(f"Created {text_file}")
    else:
        print(f"{text_file} already exists, skipping conversion.")

# Train SentencePiece
def train_sentencepiece(input_file):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    spm.SentencePieceTrainer.train(
        input=input_file,                       # Path to preprocessed text file
        model_prefix='th_vi_spm',               # Output: th_vi_spm.model, th_vi_spm.vocab
        vocab_size=100000,                      # Vocabulary size (adjust to 256000 for NLLB-200)
        model_type='bpe',                       # BPE for NLLB compatibility
        max_sentence_length=10000,              # Max sentence length
        character_coverage=0.9995,              # High coverage for Thai and Vietnamese
        normalization_rule_name='nfkc',         # Use nfkc to avoid nfc error
        split_by_whitespace=True,               # Split by whitespace
        input_sentence_size=1000000,            # Sample 1M lines to manage memory
        shuffle_input_sentence=True,            # Shuffle sentences
        pad_id=0,                               # Padding token ID
        unk_id=1,                               # Unknown token ID
        bos_id=2,                               # Beginning of sentence token ID
        eos_id=3                                # End of sentence token ID
    )

def main():
    # Input configuration
    csv_file = '/home/leloc/Document/USTH/Thesis/Data/preprocess.csv'
    text_file = '/home/leloc/Document/USTH/Thesis/Data/preprocess.txt'

    # Convert CSV to text
    try:
        preprocess_csv_to_text(csv_file, text_file, thai_column="Thai", vietnamese_column="Viet")
    except Exception as e:
        print(f"Error during CSV preprocessing: {e}")
        return

    # Train with progress bar
    print("Starting SentencePiece training...")
    with tqdm(total=100, desc="Training SentencePiece") as pbar:
        try:
            train_sentencepiece(text_file)
            pbar.update(100)  # Basic progress indicator
        except Exception as e:
            print(f"Error during SentencePiece training: {e}")
            return
    print("Training completed. Model saved as th_vi_spm.model in /home/leloc/Document/USTH/Thesis/Data/")

    # Test the model
    try:
        sp = spm.SentencePieceProcessor()
        sp.load('/home/leloc/Document/USTH/Thesis/Data/th_vi_spm.model')
        thai_sentence = "สวัสดีครับ ผมชื่อนายก"
        vietnamese_sentence = "Xin chào, tôi tên là Anh G"
        thai_tokens = sp.encode_as_pieces(thai_sentence)
        vietnamese_tokens = sp.encode_as_pieces(vietnamese_sentence)
        print("Thai Tokens:", thai_tokens)
        print("Vietnamese Tokens:", vietnamese_tokens)
        print("Decoded Thai:", sp.decode_pieces(thai_tokens))
        print("Decoded Vietnamese:", sp.decode_pieces(vietnamese_tokens))
    except Exception as e:
        print(f"Error testing model: {e}")

if __name__ == "__main__":
    main()