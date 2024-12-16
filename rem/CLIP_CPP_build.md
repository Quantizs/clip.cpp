
# CLIP Documentation

## 1. CLIP Download and Subrepo (ggml) Update

### Clone and Initialize Submodules
```bash
git clone https://github.com/monatis/clip.cpp
cd clip.cpp
git submodule update --init --recursive
mkdir build && cd build
```

### Shared Library Creation
- DLL exports are required on Windows to generate `.lib` files.
- **TODO**: Bind the DLL export defines in the `.h` file to the `BUILD_SHARED_LIBS` option.

Commands:
```bash
cmake -DBUILD_SHARED_LIBS=ON ..
cmake --build . --config Release -j 8
cmake --build . --config Debug -j 8
```

### Building Examples
- Examples do not build if the `DBUILD_SHARED_LIBS` option is enabled.

Build command:
```bash
cmake -DCLIP_BUILD_EXAMPLES=ON ..
```

### Enabling CUDA Support
```bash
which nvcc
cmake .. -DCLIP_BUILD_EXAMPLES=ON -DGGML_CUBLAS=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build . --config Release -j 8
cmake --build . --config Debug -j 8
```

---

## 2. Running Examples

### Running Samples
```bash
cd E:\clip\clip.cppuildin\Release
```

### Example Execution
```bash
./quantize.exe E:\clip\CLIP-ViT-B-32-laion2B-s34B-b79K\CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-f32.gguf E:\clip\CLIP-ViT-B-32-laion2B-s34B-b79K\CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-int4_1.gguf 3
```

---

## 3. Managing Models

### Download Model
```bash
git clone https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K
```

### Convert Model
```bash
cd E:\clip\clip.cpp\models
python convert_hf_to_gguf.py -m ../../CLIP-ViT-B-32-laion2B-s34B-b79K
```

### Quantize Model
```bash
cd E:\clip\clip.cppuildin\Releasequantize.exe E:\clip\CLIP-ViT-B-32-laion2B-s34B-b79K\CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-f32.gguf E:\clip\CLIP-ViT-B-32-laion2B-s34B-b79K\CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-int4_1.gguf 3
```

---

## 4. Updating CMake on Jetson

### Remove Old Version and Install New One
```bash
sudo apt-get update
sudo apt-get install -y libssl-dev
wget https://github.com/Kitware/CMake/releases/download/v3.25.0/cmake-3.25.0.tar.gz
tar xzf cmake-3.25.0.tar.gz
cd cmake-3.25.0
./bootstrap && make -j$(nproc) && sudo make install
```

### Verification
```bash
which cmake
cmake --version
```

---

## 5. Zero-Shot Demo on Jetson

### Execution Example
```bash
candidadmin@jetson-bp-2:~/clip/clip.cpp/build/bin$ ./zsl --model "../../../clip-vit-base-patch32_ggml-model-f32.gguf" --image "../../../burglar.jpg" --text "an image of a person with visible face" --text "an image of a person with covered face" --text "an image of a male person" --text "an image of a female"
```
