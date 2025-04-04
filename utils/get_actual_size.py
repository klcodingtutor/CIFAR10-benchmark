import os
import tempfile
import torch

def get_actual_size(model):
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "saving_file.pth")
        torch.save(model.state_dict(), model_path)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Model size: {size_mb:.2f} MB")
        return size_mb
