from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="IPEC-COMMUNITY/bridge_orig_lerobot",
    repo_type="dataset",
    local_dir="examples/SimplerEnv/bridge_orig_lerobot/",
    max_workers=4,
    resume_download=True
)