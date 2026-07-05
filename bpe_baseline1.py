from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

trainer = trainers.BpeTrainer(
    vocab_size=16000,
    min_frequency=2,
    special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
)


with open("gold_segmentation.txt", "r", encoding="utf-8") as f:
    tokenizer.train_from_iterator(f, trainer=trainer)

tokenizer.save("baseline_byte_bpe.json")