# Project Notes

## Running

```bash
source venv/bin/activate
python backend.py  # serves on http://localhost:8765
```

## Testing

Backend must be running before tests:
```bash
python backend.py &
pytest tests/test_e2e.py -v
```

## Architecture

- `backend.py` - FastAPI + WebSocket, model loading (Qwen3-0.6B on MPS), activation hooks, directional ablation
- `static/index.html` - Single-file frontend (D3.js, vanilla JS, no build step)
- `tests/test_e2e.py` - 24 Playwright E2E tests

## Key details

- Model: Qwen/Qwen3-0.6B (28 layers, hidden_size=1024, intermediate_size=3072)
- Device: MPS (Apple Metal) preferred, CPU fallback
- Must load to CPU first then `.to('mps')` to avoid segfault
- Uses `local_files_only=True` + `dtype=torch.float32`
