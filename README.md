# Wan-MultiGPU

<p align="center">
    <img src="assets/logo.png" width="400"/>
<p>
    
-----

## Quickstart

#### Installation

Install Pytorch and Triton (Nightly version preferred for faster speed and fp16 accumulation support
```
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu126
pip install triton
```

Install SageAttention
```
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention 
python setup.py install  # or pip install -e .
```

Clone the repo:
```
git clone https://github.com/intervitens/Wan-MultiGPU
cd Wan-MultiGPU
```

Install dependencies:
```
pip install -r requirements.txt
```


#### Model Download (FP16)

| Models        |                       Download Link                                           |    Notes                      |
| --------------|-------------------------------------------------------------------------------|-------------------------------|
| T2V-14B       |      🤗 [Huggingface](https://huggingface.co/IntervitensInc/Wan2.1-T2V-14B-FP16)         | Supports both 480P and 720P
| I2V-14B-720P  |      🤗 [Huggingface](https://huggingface.co/IntervitensInc/Wan2.1-I2V-14B-720P-FP16)     | Supports 720P
| I2V-14B-480P  |      🤗 [Huggingface](https://huggingface.co/IntervitensInc/Wan2.1-I2V-14B-480P-FP16)      | Supports 480P
| T2V-1.3B      |      🤗 [Huggingface](https://huggingface.co/IntervitensInc/Wan2.1-T2V-1.3B-FP16)         | Supports 480P

> Note: The 1.3B model is capable of generating videos at 720P resolution. However, due to limited training at this resolution, the results are generally less stable compared to 480P. For optimal performance, we recommend using 480P resolution.

> Note: Also supports using [original fp32 weights](https://huggingface.co/Wan-AI), strongly recommended when running in bf16 or fp32 precision


Download models using huggingface-cli:
```
pip install "huggingface_hub[cli]"
huggingface-cli download IntervitensInc/Wan2.1-T2V-14B-FP16 --local-dir ./Wan2.1-T2V-14B-FP16
```

#### Run Text-to-Video Generation

- Single-GPU inference

```
python generate.py  --task t2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-T2V-14B --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

If you encounter OOM (Out-of-Memory) issues, you can use the `--offload_model True` and `--t5_cpu` options to reduce GPU memory usage. For example, on an RTX 4090 GPU:

```
python generate.py  --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --offload_model True --t5_cpu --sample_shift 8 --sample_guide_scale 6 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

> 💡Note: If you are using the `T2V-1.3B` model, we recommend setting the parameter `--sample_guide_scale 6`. The `--sample_shift parameter` can be adjusted within the range of 8 to 12 based on the performance.

- Multi-GPU inference using FSDP + xDiT USP. Supports running the 14B model at 1280x720x81f on 4x24GB consumer GPUs, like RTX 3090 and RTX 4090.

```
torchrun --nproc_per_node=4 generate.py --task t2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-T2V-14B-FP16 --fp16_acc --dit_fsdp --dit_fsdp_offload --t5_cpu --ring_size 4 --offload_model True --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

#### Run Image-to-Video Generation

- Single-GPU inference
```
python generate.py --task i2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-I2V-14B-720P --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

> 💡For the Image-to-Video task, the `size` parameter represents the area of the generated video, with the aspect ratio following that of the original input image.

- Multi-GPU inference using FSDP + xDiT USP. Supports running the 14B model at 1280x720x81f on 4x24GB consumer GPUs, like RTX 3090 and RTX 4090.

```
torchrun --nproc_per_node=4 generate.py --task i2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-I2V-14B-720P-FP16 --image examples/i2v_input.JPG --fp16_acc --dit_fsdp --dit_fsdp_offload --t5_cpu --ring_size 4 --offload_model True --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

#### Run Text-to-Image Generation

Wan2.1 is a unified model for both image and video generation. Since it was trained on both types of data, it can also generate images. The command for generating images is similar to video generation, as follows:

##### (1) Without Prompt Extension

- Single-GPU inference
```
python generate.py --task t2i-14B --size 1024*1024 --ckpt_dir ./Wan2.1-T2V-14B  --prompt '一个朴素端庄的美人'
```

- Multi-GPU inference using FSDP + xDiT USP

```
torchrun --nproc_per_node=4 generate.py --fp16_acc --dit_fsdp --dit_fsdp_offload --t5_fsdp --ring_size 4 --offload_model True --frame_num 1 --task t2i-14B  --size 1024*1024 --prompt '一个朴素端庄的美人' --ckpt_dir ./Wan2.1-T2V-14B-FP16
```
