# Run StyleGAN2-ada on Web with TVM
## Setup (TVM)
- TVMのリポジトリをclone
  - 0b2f30aef2 のcommitでテストしました（@TODO submodule）
```
git clone --recursive https://github.com/apache/tvm.git
```
- 用意されているスクリプトでDocker imageをビルドし，コンテナを起動
  - 今後の手順は基本全てコンテナ内で行う
```
./tvm/docker/build.sh ci_wasm
./tvm/docker/bash.sh tvm.ci_wasm
```

## Setup (Stylegan2-ada-pytorch)
- stylegan2-ada-pytorchのリポジトリをclone
  - d4b2afe9c2 のcommitでテストしました（@TODO submodule）
```
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
```
- `requirements` にあるライブラリはインストールしておく

## Clone this repository
- このリポジトリをclone
```
git clone https://github.com/eTakazawa/stylegan-wasm-with-tvm.git
```

## Build TVM
- https://tvm.apache.org/docs/install/from_source.html の手順と同じ
- `build`ディレクトリを作成し，`cmake/config.make`をコピー
```
cd /workspace/tvm
mkdir build
cp cmake/config.cmake build
```
- `build/config.cmake`を編集して，`set(USE_LLVM /usr/bin/llvm-config-11)`を追加
- TVMと関連ライブラリをビルド
```
cd build
cmake ..
make -j4
```

### Setup for Web
- `web`ディレクトリに移動し，TVM WebAssembly Runtimeをビルド
```
cd /workspace/tvm/web
make
```
- TVM Wasm JS Frontendをビルド
```
npm install
npm run bundle
```
- 後で使うパッケージをインストール
```
sudo npm install -g browserify
npm install brfs
```

## Install some packages
- 使うパッケージ群を一通りインストール，また，tvmのパスを通しておく
```
sudo pip3 install torch==1.7.0
sudo pip3 install torchvision==0.8.1
sudo pip3 install matplotlib==3.3.4
sudo pip3 install mxnet==1.8.0.post0
export PYTHONPATH=/workspace/tvm/python:${PYTHONPATH}
```

## Fix for compiling the model
- エラー対策用に変更
```
cd /workspace/stylegan2-ada-pytorch
patch -p1 < ../diff_stylegan2.patch
cd /workspace/tvm
patch -p1 < ../diff_tvm.patch
```

## Deploy the model
- StyleGANをWasmにコンパイルする
  - 以下のファイルが出力されていれば成功
    - CPUで実行して生成した画像： `img.png`
    - Wasm：`stylegan.wasm`
    - パラメータ情報：`stylegan.params`
    - モデル情報：`stylegan.json`
```
cd /workspace/stylegan-wasm-with-tvm
cp deploy_model.py /workspace/stylegan2-ada-pytorch
cd /workspace/stylegan2-ada-pytorch
chown user:group /workspace/.cache
python3 deploy_model.py
```

## Web
- 先に生成したファイルやWasmを読み込む用のjsファイル等を`tvm/web`に移動
```
cp stylegan.wasm stylegan.params stylegan.json /workspace/tvm/web
cd /workspace/stylegan-wasm-with-tvm
cp load_stylegan.js index.html /workspace/tvm/web
cd /workspace/tvm/web
```
- ブラウザで動く形に変換
  - TVM公式サンプルがNode.js処理系で動く様に出来ていたので，それを踏襲していた
  - ブラウザで動く様に変換
```
browserify -t brfs load_stylegan.js -s bundle > bundle.js
```
- `tvm/web/index.html` を開いて数分待つ
  - デベロッパーツールからコンソールを確認すると，処理時間等が出力されている

---

## Details of "Fix for compiling the model" step
- emccでのコンパイル時にエラーが出るので，TOTAL_MEMORY を設定
  - `ALLOW_MEMORY_GROWTH` が設定されているので，エラーが出るのがおかしい気もする...
```diff
diff --git a/python/tvm/contrib/emcc.py b/python/tvm/contrib/emcc.py
index 89431dc..2702860 100644
--- a/python/tvm/contrib/emcc.py
+++ b/python/tvm/contrib/emcc.py
@@ -46,6 +46,7 @@ def create_tvmjs_wasm(output, objects, options=None, cc="emcc"):
     cmd += ["-s", "ERROR_ON_UNDEFINED_SYMBOLS=0"]
     cmd += ["-s", "STANDALONE_WASM=1"]
     cmd += ["-s", "ALLOW_MEMORY_GROWTH=1"]
+    cmd += ["-s", "TOTAL_MEMORY=33554432"]
```

### Modify TVM
- `tvm/python/tvm/relay/frontend/pytorch.py` を書き換え
  - `aten::square`,`profiler::_record_function_enter`,`profiler::_record_function_exit` の対応
```diff
diff --git a/python/tvm/relay/frontend/pytorch.py b/python/tvm/relay/frontend/pytorch.py
index b5cfcf5e3..a3dad3f94 100644
--- a/python/tvm/relay/frontend/pytorch.py
+++ b/python/tvm/relay/frontend/pytorch.py
@@ -2305,9 +2305,18 @@ class PyTorchOpConverter:
             unique_sliced = _op.strided_slice(unique, begin=[0], end=num_uniq, slice_mode="size")
             return (unique_sliced, indices)


+   def square(self, inputs, input_types):
+       data = inputs[0]
+       dtype = input_types[0]
+       exponent = _expr.const(2.0, dtype=dtype)
+       return _op.power(data, exponent)

    # Operator mappings
    def create_convert_map(self):
        self.convert_map = {
+           "aten::square": self.square,
+           "profiler::_record_function_enter": self.none,
+           "profiler::_record_function_exit": self.none,
            "aten::is_floating_point": self.is_floating_point,

```

### Modify stylegan2-ada-pytorch
- `SynthesisLayer`の`noise_mode`を`const`に設定
```diff
diff --git a/training/networks.py b/training/networks.py
index b046eba..5bbc150 100755
--- a/training/networks.py
+++ b/training/networks.py
@@ -283,7 +283,7 @@ class SynthesisLayer(torch.nn.Module):
             self.noise_strength = torch.nn.Parameter(torch.zeros([]))
         self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

-    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
+    def forward(self, x, w, noise_mode='const', fused_modconv=True, gain=1):
         assert noise_mode in ['random', 'const', 'none']
         in_resolution = self.resolution // self.up
         misc.assert_shape(x, [None, self.weight.shape[1], in_resolution, in_resolution
])
```
- flipの変換が未対応なので，flipと等価な処理に書き換え
```diff
diff --git a/torch_utils/ops/upfirdn2d.py b/torch_utils/ops/upfirdn2d.py
index ceeac2b..c6f5a53 100755
--- a/torch_utils/ops/upfirdn2d.py
+++ b/torch_utils/ops/upfirdn2d.py
@@ -165,6 +165,21 @@ def upfirdn2d(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1, impl='cu

+def flip_by_slice_0_1(x):
+    x = torch.stack((x[3,:], x[2,:], x[1,:], x[0,:]))
+    x = torch.stack((x[3,:], x[2,:], x[1,:], x[0,:]), dim=1)
+    return x
+    # aten::copy_ is not implmented
+    # y = torch.zeros(x.shape)
+    # n0 = x.shape[0]
+    # n1 = x.shape[1]
+    # for i in range(n0):
+    #     for j in range(n1):
+    #         y[i,j] = x[n0-i-1,j]
+    # for i in range(n1):
+    #     for j in range(n0):
+    #         y[j,i] = x[j,n1-i-1]
+
 @misc.profiled_function
 def _upfirdn2d_ref(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
     """Slow reference implementation of `upfirdn2d()` using standard PyTorch ops.
@@ -193,7 +208,8 @@ def _upfirdn2d_ref(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
     f = f * (gain ** (f.ndim / 2))
     f = f.to(x.dtype)
     if not flip_filter:
-        f = f.flip(list(range(f.ndim)))
+        # f = f.flip(list(range(f.ndim)))
+        f = flip_by_slice_0_1(f)
```
