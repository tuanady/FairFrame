import pandas as pd


def filter_crows_pairs():
    # 1. Load the original dataset
    input_file = "crows_pairs_anonymized.csv"
    output_file = "cpDataset.csv"

    print(f"📖 Reading {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{input_file}'. Make sure it's in the same folder.")
        return

    # 2. Define the bias types we want to KEEP
    # We strictly select the 3 categories relevant to your new scope
    TARGET_BIAS_TYPES = [
        "gender",
        "race-color",
        "disability"
    ]

    # 3. Apply the Filter
    # We use .isin() to select rows where 'bias_type' matches our list
    filtered_df = df[df['bias_type'].isin(TARGET_BIAS_TYPES)].copy()

    # 4. Print Statistics
    original_count = len(df)
    new_count = len(filtered_df)
    print(f"✂️  Filtering complete!")
    print(f"   - Original rows: {original_count}")
    print(f"   - Filtered rows: {new_count}")
    print(f"   - Removed: {original_count - new_count} irrelevant examples")

    # 5. Save to new CSV
    filtered_df.to_csv(output_file, index=False)
    print(f"✅ Saved clean dataset to: {output_file}")


if __name__ == "__main__":
    filter_crows_pairs()