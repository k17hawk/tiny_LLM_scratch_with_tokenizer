"""
Train BPE Tokenizer on Generated Nepali Dataset
Run: pip install tokenizers
Then: python train_tokenizer.py
"""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Initialize BPE tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Configure trainer
trainer = trainers.BpeTrainer(
    vocab_size=10000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    min_frequency=2,
)

# Train
print("Training tokenizer...")
tokenizer.train(["nepali_tokenizer_data.txt"], trainer)

# Save
tokenizer.save("nepali_tokenizer.json")
print("✓ Tokenizer saved to nepali_tokenizer.json")

# Test
test_samples = [
    "नमस्ते, तपाईंलाई कस्तो छ?",
    "namaste, tapaaiilaaii kasto chh?",
    "hey k cha khabar",
    "Hi bro, के खबर?",
]

print("\nTesting tokenizer:")
for sample in test_samples:
    encoded = tokenizer.encode(sample)
    print(f"\n'{sample}'")
    print(f"Tokens: {encoded.tokens}")
    print(f"IDs: {encoded.ids}")
