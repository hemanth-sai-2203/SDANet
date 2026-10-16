
from pathlib import Path
import torch
import yaml

from basicsr.data import build_dataset
from basicsr.models import build_model

CONFIG = Path("options/train/HSI/SDAnetChux4.yml")

with CONFIG.open("r") as f:
    opt = yaml.load(f, Loader=yaml.FullLoader)

opt["is_train"] = True
opt["dist"] = False
opt["rank"] = 0
opt["world_size"] = 1
opt["num_gpu"] = 1

# Test actual training dataset loader
print("Building training dataset...")
train_opt = dict(opt["datasets"]["train"])
train_opt["phase"] = "train"
train_opt["scale"] = opt["scale"]
dataset = build_dataset(train_opt)

print("Dataset samples:", len(dataset))
sample = dataset[0]

for key in ("gt", "lq", "lms"):
    print(key, tuple(sample[key].shape), sample[key].dtype)
    assert torch.isfinite(sample[key]).all(), f"Non-finite values in {key}"

# Add batch dimension and test CUDA forward pass
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda", "CUDA is not available"

print("Building SDANet...")
model = build_model(opt)
net = model.net_g
net.eval()

lq = sample["lq"].unsqueeze(0).to(device)
lms = sample["lms"].unsqueeze(0).to(device)
gt = sample["gt"].unsqueeze(0).to(device)

print("Running forward pass...")
with torch.no_grad():
    output = net(lq, lms)

print("Output shape:", tuple(output.shape))
print("GT shape:    ", tuple(gt.shape))

assert output.shape == gt.shape, "Output shape does not match GT"
assert torch.isfinite(output).all(), "Model output contains NaN/Inf"

print("Checking L1 loss...")
loss = torch.nn.functional.l1_loss(output, gt)
print("L1 loss:", loss.item())

print("\nSMOKE TEST PASSED")
