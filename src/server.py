"""FastAPI server — WebSocket handler, REST API, and application startup.

This is the thin orchestrator that connects the model backend, ablation engine,
experiment store, and frontend via WebSocket and REST endpoints.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.models.loader import LocalModelBackend, ModelBackend
from src.ablation.prompts import make_concept_prompts, make_baseline_prompts
from src.experiments.store import ExperimentStore
from src.experiments.schemas import Experiment, ExperimentResult, ConceptDirection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────────

backend: ModelBackend | None = None
store: ExperimentStore | None = None

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")
DEVICE = os.environ.get("DEVICE", "auto")
DB_PATH = os.environ.get("DB_PATH", "data/experiments.db")
LOCAL_FILES_ONLY = os.environ.get("LOCAL_FILES_ONLY", "true").lower() == "true"


# ── Lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global backend, store
    store = ExperimentStore(DB_PATH)
    backend = LocalModelBackend(
        model_name=MODEL_NAME,
        device=DEVICE,
        local_files_only=LOCAL_FILES_ONLY,
    )
    yield
    store.close()


app = FastAPI(title="LLM Neuron Visualizer", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static files ──────────────────────────────────────────────────────────

@app.get("/")
async def root():
    # Serve React build if it exists, otherwise fall back to legacy static/index.html
    react_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist", "index.html")
    legacy_path = os.path.join(os.path.dirname(__file__), "..", "static", "index.html")
    if os.path.exists(react_path):
        return FileResponse(react_path)
    return FileResponse(legacy_path)

# Mount static dirs — React build takes priority if it exists
_frontend_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
_legacy_static = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(_frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(_frontend_dist, "assets")), name="assets")
if os.path.isdir(_legacy_static):
    app.mount("/static", StaticFiles(directory=_legacy_static), name="static")


# ── REST API: Experiments ─────────────────────────────────────────────────

@app.get("/api/experiments")
async def list_experiments(type: str | None = None, concept: str | None = None, limit: int = 50, offset: int = 0):
    experiments = store.list_experiments(type_filter=type, concept_filter=concept, limit=limit, offset=offset)
    return [e.model_dump() for e in experiments]


@app.get("/api/experiments/{exp_id}")
async def get_experiment(exp_id: int):
    exp = store.get_experiment(exp_id)
    if exp is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    results = store.get_results(exp_id)
    directions = store.get_concept_directions(exp_id)
    return {
        "experiment": exp.model_dump(),
        "results": [r.model_dump() for r in results],
        "concept_directions": [d.model_dump() for d in directions],
    }


@app.get("/api/experiments/{exp_id}/export")
async def export_experiment(exp_id: int):
    data = store.export_experiment(exp_id)
    if not data:
        return JSONResponse({"error": "not found"}, status_code=404)
    return data


@app.delete("/api/experiments/{exp_id}")
async def delete_experiment(exp_id: int):
    deleted = store.delete_experiment(exp_id)
    if not deleted:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"deleted": True}


@app.get("/api/models")
async def list_models():
    """Return info about the currently loaded model and available profiles."""
    info = backend.get_model_info() if backend else {}
    return {"current": info}


# ── WebSocket handler ─────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket connected")
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg["type"] == "generate":
                await _handle_generate(ws, msg)

            elif msg["type"] == "ablate":
                await _handle_ablate(ws, msg)

            elif msg["type"] == "model_info":
                info = backend.get_model_info()
                await ws.send_text(json.dumps({"type": "model_info", "data": info}))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        backend.remove_ablation()
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass


async def _handle_generate(ws: WebSocket, msg: dict) -> None:
    """Handle a 'generate' message — run inference and stream results."""
    prompt = msg["prompt"]
    max_tokens = msg.get("max_tokens", 32)
    logger.info(f"Generating: {prompt!r} (max_tokens={max_tokens})")

    # Create experiment record
    exp = Experiment(
        model_name=backend.get_model_info()["name"],
        model_backend=f"local_{backend.get_model_info()['device']}",
        type="generation",
        prompt=prompt,
        max_tokens=max_tokens,
    )
    exp_id = store.create_experiment(exp)

    all_tokens = []
    for event, data_dict, tokens in backend.generate(prompt, max_tokens):
        all_tokens = tokens
        if event == "prefill":
            await ws.send_text(json.dumps({"type": "prefill", "data": data_dict}))
        elif event == "token":
            await ws.send_text(json.dumps({"type": "token", "data": data_dict}))
            await asyncio.sleep(0.05)
        elif event == "eos":
            await ws.send_text(json.dumps({"type": "done", "all_tokens": tokens}))
        elif event == "complete":
            await ws.send_text(json.dumps({
                "type": "complete",
                "all_tokens": tokens,
                "text": "".join(tokens),
            }))

    # Save result
    store.add_result(ExperimentResult(
        experiment_id=exp_id,
        variant="normal",
        output_text="".join(all_tokens),
        tokens=all_tokens,
    ))


async def _handle_ablate(ws: WebSocket, msg: dict) -> None:
    """Handle an 'ablate' message — compute directions, generate normal + ablated."""
    concept = msg["concept"]
    prompt = msg["prompt"]
    max_tokens = msg.get("max_tokens", 32)
    alpha = msg.get("alpha", 1.0)
    logger.info(f"Directional ablation: concept={concept!r}, prompt={prompt!r}, alpha={alpha}")

    # Create experiment record
    model_info = backend.get_model_info()
    exp = Experiment(
        model_name=model_info["name"],
        model_backend=f"local_{model_info['device']}",
        type="ablation",
        prompt=prompt,
        concept=concept,
        alpha=alpha,
        max_tokens=max_tokens,
    )
    exp_id = store.create_experiment(exp)

    concept_prompts = make_concept_prompts(concept)
    baseline_prompts = make_baseline_prompts()

    # Step 1: Compute concept directions
    await ws.send_text(json.dumps({
        "type": "ablation_status",
        "status": "probing",
        "message": f"Computing '{concept}' direction in residual space...",
    }))

    result = backend.compute_concept_directions(concept_prompts, baseline_prompts)
    directions = result["directions"]

    # Build per-layer stats
    num_layers = model_info["num_layers"]
    layer_stats = {}
    layer_stats_list = []
    for layer_idx in range(num_layers):
        stat = {
            "layer_idx": layer_idx,
            "has_direction": layer_idx in directions,
            "cosine_sim": round(result["cosine_sims"][layer_idx], 4),
            "diff_norm": round(result["norms"][layer_idx], 4),
            "concept_norm": round(result["concept_mean_norms"][layer_idx], 4),
            "baseline_norm": round(result["baseline_mean_norms"][layer_idx], 4),
        }
        layer_stats[str(layer_idx)] = stat
        layer_stats_list.append(stat)

    await ws.send_text(json.dumps({
        "type": "ablation_status",
        "status": "found",
        "message": f"Found '{concept}' direction across {len(directions)} layers "
                   f"(mean cos_sim={np.mean(result['cosine_sims']):.3f})",
        "layer_stats": layer_stats,
        "method": "directional",
        "total_layers": len(directions),
    }))

    # Save concept directions
    store.add_concept_direction(ConceptDirection(
        experiment_id=exp_id,
        model_name=model_info["name"],
        concept=concept,
        layer_stats=layer_stats_list,
    ))

    # Step 2: Normal generation
    await ws.send_text(json.dumps({
        "type": "ablation_status",
        "status": "generating_normal",
        "message": "Running normal generation...",
    }))

    normal_tokens = []
    normal_prefill = None
    for event, data_dict, all_tokens in backend.generate(prompt, max_tokens):
        if event == "prefill":
            normal_prefill = data_dict
        normal_tokens = all_tokens

    await ws.send_text(json.dumps({
        "type": "ablation_normal",
        "text": "".join(normal_tokens),
        "tokens": normal_tokens,
        "prefill": normal_prefill,
    }))

    store.add_result(ExperimentResult(
        experiment_id=exp_id,
        variant="normal",
        output_text="".join(normal_tokens),
        tokens=normal_tokens,
    ))

    # Step 3: Ablated generation
    await ws.send_text(json.dumps({
        "type": "ablation_status",
        "status": "generating_ablated",
        "message": f"Running directional ablation (alpha={alpha}, {len(directions)} layers)...",
    }))

    backend.install_ablation(directions, alpha=alpha)

    ablated_tokens = []
    ablated_prefill = None
    for event, data_dict, all_tokens in backend.generate(prompt, max_tokens):
        if event == "prefill":
            ablated_prefill = data_dict
        ablated_tokens = all_tokens

    backend.remove_ablation()

    await ws.send_text(json.dumps({
        "type": "ablation_ablated",
        "text": "".join(ablated_tokens),
        "tokens": ablated_tokens,
        "prefill": ablated_prefill,
    }))

    store.add_result(ExperimentResult(
        experiment_id=exp_id,
        variant="ablated",
        output_text="".join(ablated_tokens),
        tokens=ablated_tokens,
    ))

    await ws.send_text(json.dumps({
        "type": "ablation_complete",
        "concept": concept,
        "experiment_id": exp_id,
        "normal_text": "".join(normal_tokens),
        "ablated_text": "".join(ablated_tokens),
        "method": "directional",
        "alpha": alpha,
        "layers_affected": len(directions),
        "mean_cosine_sim": round(float(np.mean(result["cosine_sims"])), 4),
        "mean_diff_norm": round(float(np.mean(result["norms"])), 4),
    }))


# ── Entrypoint ────────────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)


if __name__ == "__main__":
    main()
