import dnnlib
import pickle
import torch
import numpy as np
from training.networks import MappingNetwork, SynthesisNetwork, SynthesisBlock, Generator
import matplotlib.pyplot as plt
import numpy as np
from tvm.contrib import emcc

import tvm
from tvm import relay

def imshow(img):
  npimg = img.detach().numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.savefig('img.png')

def disable_not_implemeted_property(mod):
  if hasattr(mod, 'use_fp16'):
    mod.use_fp16 = False
  if hasattr(mod, 'use_noise'):
    mod.use_noise = False
  children = list(mod.children())
  for child in children:
    disable_not_implemeted_property(child)

# Loat pre-trained model
print('Begin load pre-trained model')
url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl' # cat
with dnnlib.util.open_url(url) as f:
  G = pickle.load(f)['G_ema']
# Handle some modules with CPU&fp16
disable_not_implemeted_property(G.synthesis)
print('End load pre-trained model')

# Load state-dict
g = Generator(G.z_dim, G.c_dim, G.w_dim, G.img_resolution, G.img_channels)
g.load_state_dict(G.state_dict())
G = g.eval()

# Set random array to input_data 
rnd = np.random.RandomState(32)
input_data = torch.tensor(rnd.randn(1, G.z_dim), dtype=torch.float32) # latent codes
label_data = torch.tensor([0], dtype=torch.float32)
print("input_data.dtype:", input_data.dtype)

# Run for test
img = g(input_data, label_data)
print("output : ", img.shape)
print(img.dtype)
imshow(img.view(img.shape[1:]))

# Compile to TorchScripted model
print('Begin torch.jit.trace')
with torch.no_grad():
  # Run for test
  print(' Begin test inference')
  img = G(input_data, label_data)
  print(' End test inference')
  print(' Begin do torch.jit.trace')
  # Compile to TorchScripted model
  model_traced = torch.jit.trace(G, (input_data, label_data))
  model_traced = model_traced.eval()
  print(' End do torch.jit.trace')
print('End torch.jit.trace')

# Relay Build
print('Begin relay.build')
# Set target
target = "llvm -mtriple=wasm32-unknown-unknown-wasm -system-lib"
if not tvm.runtime.enabled(target):
    raise RuntimeError("Target %s is not enbaled" % target)

dev = tvm.cpu()
print("input : ", input_data.shape, "/ label : ", label_data.shape)
shape_list = [("input", input_data.shape), ("label", label_data.shape)]
mod, params = relay.frontend.from_pytorch(model_traced, shape_list)
with tvm.transform.PassContext(opt_level=3):
  graph,lib,params = relay.build(mod, target=target, params=params)
print('End relay.build')

# Save the library at local temporary directory.
net = 'stylegan'
lib.export_library(f"{net}.wasm", emcc.create_tvmjs_wasm)
with open(f"{net}.json", "w") as f_graph_json:
    f_graph_json.write(graph)
with open(f"{net}.params", "wb") as f_params:
    f_params.write(relay.save_param_dict(params))
