# ClauseGuard

ClauseGuard analyzes contract clauses using segmentation, retrieval, judgment, aggregation, verification, and PostgreSQL persistence.

## Evaluation

The evaluation dataset contains 28 labeled cases:

- 22 training cases
- 6 held-out evaluation cases
- Fine-tuned model: `Qwen/Qwen2.5-1.5B-Instruct`
- Fine-tuning method: LoRA
- Training epochs: 3

### Held-out model comparison

| Model | Retrieval | Accuracy |
|---|---:|---:|
| Frontier | Enabled | 0.6667 |
| Frontier | Disabled | 0.0000 |
| Fine-tuned Qwen | Enabled | 0.1667 |
| Fine-tuned Qwen | Disabled | 0.0000 |

The frontier model performed better than the fine-tuned model on the six held-out cases. Retrieval was important for both models, and both models scored zero without retrieved references.

These results are exploratory because the fine-tuning dataset contains only 22 training examples. The fine-tuned adapter is not currently a replacement for the frontier model.

## Running the API

```powershell
uv run uvicorn main:app --reload
```

Available endpoints:

- `GET /health`
- `POST /api/analyze`
- `GET /api/runs`
- `GET /api/runs/{run_id}`
- `GET /api/eval/latest`

## Running tests

```powershell
uv run ruff check .
uv run pytest
```

## Fine-tuning

Generate the dataset:

```powershell
uv run python -m fine_tuning.build_dataset
```

Train the LoRA adapter:

```powershell
uv run python -m fine_tuning.train
```

Run the model comparison:

```powershell
uv run python -m eval.model_comparison
```

The fine-tuned model weights are local artifacts and should not be committed to Git.
