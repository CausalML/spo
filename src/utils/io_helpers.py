import os, json, pandas as pd, torch

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_metrics(path, metrics_dict):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)

def append_csv(path, row_dict):
    ensure_dir(os.path.dirname(path))
    df = pd.DataFrame([row_dict])
    if not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode='a', header=False, index=False)

def save_model(path, model, accelerator=None):
    ensure_dir(os.path.dirname(path))
    if accelerator is None or accelerator.is_local_main_process:
        torch.save(model.state_dict(), path)

