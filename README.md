# stable-diffusion-upscaler-demo
Stable Diffusion Upscaler Demo

Base notebook: https://colab.research.google.com/drive/1o1qYJcFeywzCIdkfKJy7cTpgZTCM2EI4#scrollTo=D8TJlpNohGiI

```bash
$ conda create --name sd01 python=3.10.12
$ conda activate sd01
$ pip install -r requirements.txt
$ cd ./src
$ git clone https://github.com/CompVis/stable-diffusion
$ git clone https://github.com/CompVis/taming-transformers
$ git clone https://github.com/CompVis/latent-diffusion
$ python run.py --prompt <your prompt>
```
Your generated image will be saved in "./src/outputs".
