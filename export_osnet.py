import os
import sys
import torch
from huggingface_hub import hf_hub_download

# Download OSNet model
print("Downloading OSNet model from Hugging Face...")
model_path = hf_hub_download(repo_id="kaiyangzhou/osnet", filename="osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth")

print(f"Model downloaded to {model_path}")

sys.path.append(os.path.abspath('deep-person-reid'))
from torchreid.models import build_model

print("Building OSNet_x1_0...")
osnet = build_model(name='osnet_x1_0', num_classes=1041, pretrained=False) # Number of classes in MSMT17 or we can just load the backbone

print("Loading state dict...")
state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
if 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']

# Remove 'module.' prefix if it exists (from DataParallel)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "") if k.startswith("module.") else k
    # We don't need the classifier head because we just want features
    if not name.startswith('classifier.'):
        new_state_dict[name] = v

osnet.load_state_dict(new_state_dict, strict=False)
osnet.eval()

print("Exporting OSNet to ONNX...")
dummy_input = torch.randn(1, 3, 256, 128)

if not os.path.exists('models'):
    os.makedirs('models')

torch.onnx.export(osnet, dummy_input, "models/osnet_x1_0.onnx", 
                  opset_version=11, 
                  input_names=['input'], 
                  output_names=['output'])

print("Successfully exported OSNet to models/osnet_x1_0.onnx")
