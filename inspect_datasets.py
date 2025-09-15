from datasets import load_dataset

# Load datasets
de_ds = load_dataset("wmt16", "de-en")
ur_ds = load_dataset("opus100", "en-ur")

print("=== German Dataset (WMT16: de-en) ===")
print(de_ds)  # overview
print("\nColumn names:", de_ds["train"].column_names)
print("\nFirst example (train):", de_ds["train"][0])

print("\n=== Urdu Dataset (Opus100: en-ur) ===")
print(ur_ds)  # overview
print("\nColumn names:", ur_ds["train"].column_names)
print("\nFirst example (train):", ur_ds["train"][0])

# Explore sizes
print("\nGerman train size:", len(de_ds["train"]))
print("German validation size:", len(de_ds["validation"]))
print("Urdu train size:", len(ur_ds["train"]))
print("Urdu validation size:", len(ur_ds["validation"]))

# Look at random samples
print("\nRandom German sample:", de_ds["train"].shuffle(seed=42)[0])
print("Random Urdu sample:", ur_ds["train"].shuffle(seed=42)[0])

# Show first 5 examples
print("\nFirst 5 German examples:")
print(de_ds["train"][:5])

print("\nFirst 5 Urdu examples:")
print(ur_ds["train"][:5])
