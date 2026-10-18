# MLOps Lab: LLM Data Pipeline (Lab 1) with Improvements

This repo contains my completed version of **Lab 1 from the LLM Data Pipeline lab** in the course repository ([raminmohammadi/MLOps → Labs/Data_Labs/LLM_Data_Pipeline](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Data_Labs/LLM_Data_Pipeline)), plus three additions of my own, each measured with before/after results.

## What the original lab does

The lab prepares raw text so it can be used to train a GPT-2 style language model:

1. Loads the WikiText-2 dataset (36,718 lines of Wikipedia text)
2. Loads GPT-2's tokenizer
3. Converts all text into token IDs
4. Joins all tokens and cuts them into fixed-length blocks of 128 tokens
5. Wraps the blocks in a PyTorch DataLoader (batches of 8)
6. Verifies one batch has the expected shape `[8, 128]`

## Fix needed to run the lab

- **Dataset ID updated:** `load_dataset("wikitext", ...)` now fails with newer Hugging Face library versions, which require the `owner/name` format. Changed to `"Salesforce/wikitext"`, which is the same dataset.

## My additions

### 1. Faster concatenation with `itertools.chain`
The original `group_texts` joins token lists using `sum(list_of_lists, [])`, which copies the growing list on every step. I replaced it with `itertools.chain.from_iterable` and timed both versions.

| Version | Time |
|---|---|
| Original (`sum`) | 11.68 s |
| Mine (`itertools.chain`) | 3.22 s |
| **Speedup** | **3.6x** |

Both versions produce **identical output** (verified sequence by sequence).

### 2. Removing blank lines before tokenizing
WikiText-2 contains many empty lines, which waste processing time and produce empty token lists.

| Metric | Original | Cleaned |
|---|---|---|
| Lines | 36,718 | 23,767 (12,951 blank lines removed, 35.3%) |
| Tokenizing time | 13.22 s | 9.45 s (~29% faster) |
| Training sequences | 18,667 | 18,676 |

Cleaning also produced **9 extra training sequences**. Grouping is done in batches of 1,000 lines, and each batch discards its leftover tokens. With fewer lines there are fewer batches, so fewer tokens are thrown away.

### 3. Comparing block sizes
Tested how block size affects the number of training sequences and how many tokens are discarded.

| Block size | Sequences | Tokens used | Tokens dropped | % dropped |
|---|---|---|---|---|
| 128 | 18,676 | 2,390,528 | 1,356 | 0.06% |
| 256 | 9,331 | 2,388,736 | 3,148 | 0.13% |
| 512 | 4,660 | 2,385,920 | 5,964 | 0.25% |
| 1024 | 2,325 | 2,380,800 | 11,084 | 0.46% |

Doubling the block size halves the number of sequences. Dropped tokens rise with block size but stay under 0.5%, so larger blocks (longer context for the model) can be used without meaningful data loss.

## How to run

1. Open `LLM_Data_Pipeline_Lab1.ipynb` in [Google Colab](https://colab.research.google.com/) (File → Upload notebook).
2. Run all cells from top to bottom (skip the first commented-out `pip` cell).

Everything is free: no GPU, API key, or paid service is needed. Dependencies (`datasets`, `transformers`, `torch`) come preinstalled in Colab.

## Files

- `LLM_Data_Pipeline_Lab1.ipynb`: the original lab plus my three additions, with all outputs
- `README.md`: this file.
