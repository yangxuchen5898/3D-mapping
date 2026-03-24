from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="rishitdagli/nerf-gs-datasets",
    repo_type="dataset",
    local_dir="data/nerf_synthetic",
    allow_patterns=[
        "lego/*",
    ],
)