# 🧠 Step 2 — Input Embeddings (BERT)

After tokenization BERT maps token IDs to vectors. Example token IDs:

```
[101, 19081, 2024, 6429, 1012, 102]
```

Each ID becomes a 768-dimensional vector (for BERT‑base). The final input vector for a token is the element-wise sum of three learned embeddings:

InputEmbedding = TokenEmbedding + SegmentEmbedding + PositionEmbedding

All three vectors have size 768.

## 1) Token embeddings
These encode the word/token meaning. Each token ID has a learned 768-d vector updated during pretraining.

Example:
- `transformers` → token embedding for "transformers"
- `are` → token embedding for "are"
- `amazing` → token embedding for "amazing"

## 2) Segment (sentence) embeddings
BERT can take sentence pairs. A segment embedding indicates which sentence a token belongs to:
- Sentence A → segment ID 0 (embedding for 0)
- Sentence B → segment ID 1 (embedding for 1)

Example (tokens with segment IDs):
| Token | Segment ID |
|---:|:---:|
| [CLS] | 0 |
| what | 0 |
| is | 0 |
| ai | 0 |
| ? | 0 |
| [SEP] | 0 |
| ai | 1 |
| means | 1 |
| artificial | 1 |
| intelligence | 1 |
| . | 1 |
| [SEP] | 1 |

For a single sentence, all segment IDs are 0.

## 3) Position embeddings
Transformers are order‑agnostic by default, so BERT adds a learned position embedding for each token position (0…511). Example positions:
- [CLS] → pos 0
- `transformers` → pos 1
- `are` → pos 2
- `amazing` → pos 3
- `.` → pos 4
- [SEP] → pos 5

## Final input vector
For each token:
```
InputEmbedding = TokenEmbedding + SegmentEmbedding + PositionEmbedding
```
Each resulting 768-d vector is fed into the Transformer encoder layers.

## Quick visual summary
| Token | Token Embedding | Segment Embedding | Position Embedding | Final (768-d) |
|---|---|---:|---:|---|
| [CLS] | word meaning | sentence A (0) | pos 0 | sum |
| transformers | word meaning | sentence A (0) | pos 1 | sum |
| are | word meaning | sentence A (0) | pos 2 | sum |
| amazing | word meaning | sentence A (0) | pos 3 | sum |
| . | word meaning | sentence A (0) | pos 4 | sum |
| [SEP] | word meaning | sentence A (0) | pos 5 | sum |

These 768-dimensional input vectors are then processed by the Transformer encoder.
