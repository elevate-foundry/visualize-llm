# LLM Neuron Firing Visualizer

Interactive visualization of neuron activations and directional ablation for small language models. Watch neurons fire in real-time, then surgically erase concepts from the model using Heretic-style directional ablation.

![Ablation Screenshot](https://github.com/user-attachments/assets/placeholder)

## What it does

- **Neuron firing visualization**: See which neurons fire across all 28 layers of Qwen3-0.6B as it generates text, rendered as interactive heatmaps
- **Firing rate analysis**: Bar charts showing what fraction of neurons activate per layer
- **Network view**: Visual representation of the model architecture with firing intensity
- **Directional ablation**: Erase arbitrary concepts (dog, math, love, etc.) from the model by projecting out their direction in residual space, then compare normal vs ablated generation side-by-side

## How directional ablation works

Based on [Arditi et al. 2024](https://arxiv.org/abs/2406.11717) ("Refusal in Language Models Is Mediated by a Single Direction") and [Heretic](https://github.com/p-e-w/heretic):

1. Run concept-related prompts ("The dog is", "A dog can", ...) and baseline prompts through the model
2. Collect residual stream activations at each layer
3. Compute the **concept direction** = normalized(mean_concept_residuals - mean_baseline_residuals) per layer
4. During generation, **project out** that direction: `h' = h - alpha * (h . d) * d`

This removes the concept from the model's representations while preserving everything orthogonal to it.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download the model (first run only)
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')"
```

## Usage

```bash
python backend.py
# Open http://localhost:8765
```

- Enter a prompt and click **Generate & Visualize** to see neuron activations
- Enter a concept to erase and click **Ablate & Compare** for side-by-side comparison
- Adjust **alpha** (0-2) to control ablation strength

## Running tests

```bash
# Start the server first
python backend.py &

# Run E2E tests
pip install pytest-playwright playwright
playwright install chromium
pytest tests/test_e2e.py -v
```

## Architecture

- **Backend** (`backend.py`): FastAPI + WebSocket server, loads Qwen3-0.6B via HuggingFace Transformers on MPS/CPU, registers forward hooks to capture activations, implements directional ablation
- **Frontend** (`static/index.html`): Single-page app with D3.js for visualization, WebSocket client for real-time streaming
- **Tests** (`tests/test_e2e.py`): 24 Playwright E2E tests covering page load, generation, all visualization tabs, and directional ablation

## Example results

**Ablating "dog" (alpha=1.0)**:
- Normal: "A dog is a wonderful pet because it is so much more than just a dog. It is a companion, a friend, and a"
- Ablated: "A dog is a wonderful pet because it is made of a number of pieces. A piece is made of a number of pieces."

**Ablating "math" (alpha=0.5)**:
- Normal: "The square root of 144 is 12. So, the square root of 144 is 12."
- Ablated: "The square root of 144 is a number that...? The square root of 144 is a number that...?"

## References

- Arditi et al., "Refusal in Language Models Is Mediated by a Single Direction", NeurIPS 2024
- [Heretic](https://github.com/p-e-w/heretic) - Automatic directional ablation for censorship removal
- [LEACE](https://arxiv.org/abs/2306.03819) - Closed-form concept erasure
