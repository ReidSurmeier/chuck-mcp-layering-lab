#!/usr/bin/env python3
"""
Auto-optimize via OpenClaw → Opus subagent.

When triggered, saves the image + params to a temp dir and calls
`openclaw system event` to notify the main session. The main agent
then spawns an Opus subagent to iteratively evaluate and tune parameters.
"""
import hashlib
import io
import json
import os
import subprocess
import tempfile

from PIL import Image
import numpy as np

OPTIMIZE_DIR = "/tmp/woodblock-optimize"


def trigger_optimization(image_bytes: bytes, initial_plates: int = 8) -> dict:
    """
    Save image and trigger OpenClaw to start the optimization loop.
    Returns a job ID the frontend can poll for results.
    """
    os.makedirs(OPTIMIZE_DIR, exist_ok=True)
    
    # Generate job ID from image hash
    job_id = hashlib.md5(image_bytes[:1024]).hexdigest()[:12]
    job_dir = os.path.join(OPTIMIZE_DIR, job_id)
    import re
    if not re.fullmatch(r"[a-f0-9]{12}", job_id):
        raise ValueError(f"Invalid job_id: {job_id}")
    os.makedirs(job_dir, exist_ok=True)
    
    # Save input image
    img_path = os.path.join(job_dir, "input.png")
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.save(img_path)
    
    # Save initial params
    params = {
        "plates": initial_plates,
        "dust": 30,
        "median_size": 3,
        "chroma_boost": 1.3,
        "shadow_threshold": 8,
        "highlight_threshold": 95,
        "use_edges": True,
        "edge_sigma": 1.5,
    }
    params_path = os.path.join(job_dir, "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f)
    
    # Write status
    status = {"job_id": job_id, "status": "queued", "iteration": 0, "best_score": 0}
    with open(os.path.join(job_dir, "status.json"), "w") as f:
        json.dump(status, f)
    
    # Trigger OpenClaw system event
    event_text = (
        f"Auto-optimize job {job_id} queued. "
        f"Image saved at {img_path}, params at {params_path}. "
        f"Spawn an Opus vision subagent to run the optimization loop: "
        f"cd /home/reidsurmeier/sites/woodblock-tools/backend && "
        f"python3 -c \"from auto_optimize import run_iteration; run_iteration('{job_id}')\" "
        f"Evaluate the composite vs original, adjust params, iterate up to 5 times. "
        f"Save best result. Update status.json after each iteration."
    )
    
    try:
        subprocess.Popen(
            ["openclaw", "system", "event", "--text", event_text, "--mode", "now"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"Failed to trigger OpenClaw event: {e}")
    
    return status


def run_iteration(job_id: str, params_override: dict = None) -> dict:
    """
    Run one iteration of the separation with current/given params.
    Called by the Opus subagent.
    Returns the result manifest + saves composite.
    """
    import separate_v2 as v2
    
    job_dir = os.path.join(OPTIMIZE_DIR, job_id)
    img_path = os.path.join(job_dir, "input.png")
    
    # Load params
    params_path = os.path.join(job_dir, "params.json")
    with open(params_path) as f:
        params = json.load(f)
    
    if params_override:
        params.update(params_override)
        with open(params_path, "w") as f:
            json.dump(params, f)
    
    # Load status
    status_path = os.path.join(job_dir, "status.json")
    with open(status_path) as f:
        status = json.load(f)
    
    iteration = status.get("iteration", 0) + 1
    
    # Run separation
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    
    result = v2.separate(
        arr,
        n_plates=params["plates"],
        dust_threshold=params["dust"],
        use_edges=params.get("use_edges", True),
        edge_sigma=params.get("edge_sigma", 1.5),
        median_size=params.get("median_size", 3),
        chroma_boost=params.get("chroma_boost", 1.3),
        shadow_threshold=params.get("shadow_threshold", 8),
        highlight_threshold=params.get("highlight_threshold", 95),
        return_data=True,
    )
    
    # Save composite for this iteration
    comp_path = os.path.join(job_dir, f"composite_iter{iteration}.png")
    result["composite"].save(comp_path)
    
    # Save plate images
    for plate_info in result["manifest"]["plates"]:
        name = plate_info["name"]
        plate_path = os.path.join(job_dir, f"{name}_iter{iteration}.png")
        result["plates"][name]["image"].save(plate_path)
    
    # Update status
    status["iteration"] = iteration
    status["status"] = "running"
    status["last_params"] = params
    status["last_composite"] = comp_path
    with open(status_path, "w") as f:
        json.dump(status, f)
    
    return {
        "iteration": iteration,
        "params": params,
        "composite_path": comp_path,
        "original_path": img_path,
        "manifest": result["manifest"],
    }


def finalize(job_id: str, best_score: int, best_iteration: int):
    """Mark optimization as complete."""
    job_dir = os.path.join(OPTIMIZE_DIR, job_id)
    status_path = os.path.join(job_dir, "status.json")
    
    with open(status_path) as f:
        status = json.load(f)
    
    status["status"] = "complete"
    status["best_score"] = best_score
    status["best_iteration"] = best_iteration
    status["best_composite"] = os.path.join(job_dir, f"composite_iter{best_iteration}.png")
    
    with open(status_path, "w") as f:
        json.dump(status, f)


def get_status(job_id: str) -> dict:
    """Get current optimization status."""
    job_dir = os.path.join(OPTIMIZE_DIR, job_id)
    status_path = os.path.join(job_dir, "status.json")
    
    if not os.path.exists(status_path):
        return {"job_id": job_id, "status": "not_found"}
    
    with open(status_path) as f:
        return json.load(f)
