## CosyVoice2-GRPO

[Demos](https://ryuclc.github.io/LLM-TTS-GRPO/); [Paper](https://arxiv.org/abs/2509.18798)

A simple implementation for improving CosyVoice2 by GRPO method.

This is the code of paper “Group Relative Policy Optimization for Text-to-Speech with Large Language Models”

We modify official CosyVoice2 code with trl to achieve GRPO fine-tune. Only need the CosyVoice2 environment, without need to install any other modules or frameworks.

Most codes are borrowed from official [CosyVoice2](https://github.com/FunAudioLLM/CosyVoice), https://github.com/channel-io/ch-tts-llasa-rl-grpo, https://github.com/SebastianBodza/blog_projects/tree/main/00_GRPO_LLaSa, https://github.com/huggingface/trl.


### Note

Now the implementation is rough, plan to refine code or migrate to other framework in the future.


### To-Do

- [x] paper
- [x] code
- [ ] inference with ckpt
- [ ] training script
