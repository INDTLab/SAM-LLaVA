# SAM-LLaVA: A Segmentation-Aware Vision-Language Framework for Industrial Defect Diagnosis

Shengwang An, Chengjia Wang, Xinghui Dong


### Requirements

The project depends on a very small set of packages which are listed in `requirements.txt`.  To install the requirements run:

```bash
pip install -r requirements.txt
```

### Dataset preparation

Training SAM‑LLaVA requires paired image–mask–text data.  We uses industrial defect datasets such as MVTec‑AD and VisA, and also a few‑shot support set to calibrate the CLIP–SAM cascade.  The data structure is as follows: 

```
data/
├── images/
│   ├── img_0001.png
│   ├── img_0002.png
│   └── ...
├── masks/
│   ├── img_0001_mask.png
│   ├── img_0002_mask.png
│   └── ...
└── descriptions.json
```



### Training
Two training scripts are provided:


* **training (`scripts/train_full.py`)** – Uses the official CLIP, SAM and Vicuna models with LoRA.  An example configuration file is provided in `code/config/train_full.yaml`.  You must download the SAM ViT‑H checkpoint manually and specify its path in the config.  During training, the CLIP, SAM and Vicuna weights are frozen while the LoRA adapters and a context projection layer are optimised.  To train the full model run:

  ```bash
  cd sam_llava_project/code
  python scripts/train_full.py --config ../code/config/train_full.yaml
  ```



### Evaluation


Once training is complete you can evaluate the model on a validation or test split using one of the following commands:



* **evaluation:**

  ```bash
  python scripts/test_full.py --config ../code/config/train_full.yaml --checkpoint path/to/full_checkpoint.pth
  ```


### Downloading pretrained weights

To run the full SAM‑LLaVA model you must download several pretrained checkpoints yourself:

1. **CLIP ViT‑B/16** – The image and text encoders are provided by the HuggingFace model `openai/clip-vit-base-patch16`.  The code will automatically download the weights via the HuggingFace hub the first time it is run, or you can manually download them with `huggingface-cli` and place them in your cache directory.

2. **SAM ViT‑H** – Download `sam_vit_h_4b8939.pth` from the official Segment‑Anything Model repository: <https://github.com/facebookresearch/segment-anything>.  Place this file in a directory of your choosing and set the `sam_checkpoint` field in `train_full.yaml` to the appropriate path.

3. **Vicuna‑7B‑v1.5** – The base language model is available from HuggingFace under the model id `lmsys/vicuna-7b-v1.5`.  You must agree to the model licence on HuggingFace to download it.  




## Citation

If you find our work useful in your research, please consider citing:
```
```