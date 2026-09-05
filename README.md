# LLM pretraining

An educational project for implementing and experimenting with GPT-style
language models from scratch.

## Features

The project currently implements:

- GPT-2 style Transformer architecture
- Training and validation pipeline
- Text generation with top-k sampling and temperature scaling
- Loading pretrained GPT-2 weights
- Preprocessing of research paper files for further pretraining
- First experiment: Continued pretraining on 10 papers

## Current experiment

I'm currently investigating how well a GPT-2 model can be adapted to the
language of research papers through continued pretraining.

The current experiment uses a small sample of research papers and compares text 
generation before and after continued pretraining.

## Future work

- Expand corpus of training data
- Evaluate model performance quantatively
- Explore instruction fine-tuning

## Credits

The GPT-style language model implemented in this project is based on the 
architecture and implementation presented in the excellent book *Build a 
Large Language Model from Scratch* by Sebastian Raschka.