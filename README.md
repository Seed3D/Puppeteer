<div align="center">

<h1>Puppeteer: Rig and Animate Your 3D Models</h1>

<p>
  <a href="https://chaoyuesong.github.io"><strong>Chaoyue Song</strong></a><sup>1,2</sup>,
  <a href="https://lixiulive.com/"><strong>Xiu Li</strong></a><sup>2</sup>,
  <a href="https://scholar.google.com/citations?user=afDvaa8AAAAJ&hl"><strong>Fan Yang</strong></a><sup>1</sup>,
  <a href="https://zcxu-eric.github.io/"><strong>Zhongcong Xu</strong></a><sup>2</sup>,
  <a href="https://plusmultiply.github.io/"><strong>Jiacheng Wei</strong></a><sup>1</sup>,
 <br>
  <a href="https://sites.google.com/site/fayaoliu"><strong>Fayao Liu</strong></a><sup>3</sup>,
  <a href="https://scholar.google.com.sg/citations?user=Q8iay0gAAAAJ"><strong>Jiashi Feng</strong></a><sup>2</sup>,
  <a href="https://guosheng.github.io/"><strong>Guosheng Lin</strong></a><sup>1*</sup>,
  <a href="https://jfzhang95.github.io/"><strong>Jianfeng Zhang</strong></a><sup>2*</sup>
  <br>
  *Corresponding authors
  <br>
    <sup>1 </sup>Nanyang Technological University
  <sup>2 </sup>Bytedance Seed
  <sup>3 </sup>A*STAR
</p>

<h3>arXiv 2025</h3>

<div align="center">
  <img width="80%" src="assets/puppeteer_teaser.gif">
</div>

<p>
  <a href="https://chaoyuesong.github.io/Puppeteer/"><strong>Project</strong></a> |
  <a href="https://arxiv.org/abs/2508.10898"><strong>Paper</strong></a> | 
  <a href="https://www.youtube.com/watch?v=eJP_VR4cVnk"><strong>Video</strong></a> |
  <a href="https://huggingface.co/datasets/chaoyue7/Articulation-XL2.0"><strong>Data: Articulation-XL2.0</strong></a>
</p>


</div>

<br/>

Puppeteer is proposed for **automatic rigging and animation of 3D objects**. Given a 3D object, Puppeteer first automatically generates skeletal structures and skinning weights, then animates the rigged model with video guidance through a differentiable optimization pipeline. This comprehensive approach aims to enable fully automated transformation of static 3D models into dynamically animated assets, eliminating the need for manual rigging expertise and significantly streamlining 3D content creation workflows.

<br/>

## 🔥 News
- Sep 04, 2025: We have released the diverse pose subset of Articulation-XL2.0, and now the [complete Articulation-XL2.0 dataset](https://huggingface.co/datasets/Seed3D/Articulation-XL2.0) is available, featuring 59.4K high-quality rigged data samples. We also release the inference codes and [model checkpoints](https://huggingface.co/Seed3D/Puppeteer).
- Aug 15, 2025: Release [paper](https://arxiv.org/abs/2508.10898) of Puppeteer!


## 🔧 Installtation
We use Python 3.10 with PyTorch 2.1.1 and CUDA 11.8. The environment and required packages can be installed as follows:

```
git clone https://github.com/Seed3D/Puppeteer.git --recursive && cd Puppeteer
conda create -n puppeteer python==3.10.13 -y
conda activate puppeteer
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install flash-attn==2.6.3 --no-build-isolation
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.1+cu118.html
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt211/download.html
```

## 🚀 Demo

We provide a complete pipeline for rigging and animating 3D models. Before running the pipeline, visit each folder (skeleton, skinning, animation) to download the necessary model checkpoints.

### Rigging

Given 3D meshes, we first predict the skeleton and skinning weights:

```
bash demo_rigging.sh
```

The final rig files will be saved in `results/final_rigging`. To evaluate the [skeleton](https://github.com/Seed3D/Puppeteer/tree/main/skeleton) and [skinning](https://github.com/Seed3D/Puppeteer/tree/main/skinning) components separately, refer to their respective folders. 

### Video-guided 3D animation

To animate the rigged model using video guidance, run:

```
bash demo_animation.sh
```

The rendered 3D animation sequence from different views will be saved in `results/animation`. Refer to the [animation folder](https://github.com/Seed3D/Puppeteer/tree/main/animation) for comprehensive details on data processing and structure.


## 😊 Acknowledgment

The code builds upon [MagicArticulate](https://github.com/Seed3D/MagicArticulate), [MeshAnything](https://github.com/buaacyw/MeshAnything), [Functional Diffusion](https://1zb.github.io/functional-diffusion/), [RigNet](https://github.com/zhan-xu/RigNet), [Michelangelo](https://github.com/NeuralCarver/Michelangelo/), [PartField](https://github.com/nv-tlabs/PartField), [AnyMole](https://github.com/kwanyun/AnyMoLe) and [Lab4D](https://github.com/lab4d-org/lab4d). We gratefully acknowledge the authors for making their work publicly available.


## 📚 Citation

```
@article{song2025puppeteer,
  title={Puppeteer: Rig and Animate Your 3D Models},
  author={Chaoyue Song and Xiu Li and Fan Yang and Zhongcong Xu and Jiacheng Wei and Fayao Liu and Jiashi Feng and Guosheng Lin and Jianfeng Zhang},
  journal={arXiv preprint arXiv:2508.10898},
  year={2025}
}
```