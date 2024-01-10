from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 50,
        "lr": 10**-4,
        "seq_len": 400,
        "d_model": 512,
        "question": "question",
        "answer": "answer",
        "model_folder": r'C:\Users\OS\Desktop\AiBag\listen_speak\weights',
        "model_basename": "tmodel_",
        "preload": '05',
        "tokenizer_file": r"C:\Users\OS\Desktop\AiBag\listen_speak\tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


