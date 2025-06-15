# IFEval-Audio

## Overview
IFEval-Audio is a dataset to evaluate instruction-following in audio-based LLMs, with 280 audio-instruction-answer triples across six dimensions: Content, Capitalization, Symbol, List Structure, Length, and Format.

## Dataset Structure
- **Audio Input**: From Spoken SQUAD, TED-LIUM 3, Muchomusic, etc.
- **Text Instruction**: Specifies one dimension (e.g., "Use JSON format").
- **Expected Answer**: Reference output.
- **Dimensions**: Content, Capitalization, Symbol, List Structure, Length, Format.
- **Distribution**: 240 speech triples (40/dimension), 40 music/environmental triples.

## Evaluation Metrics
IFR: Format adherence score (0/1).
SCR: Semantic correctness score (0/1).
OSR: Triples with IFR=1 and SCR=1.

## Citation
```bibtex
@article{gao2025ifevalaudio,
  title={IFEval-Audio: Benchmarking Instruction-Following Capability in Audio-based Large Language Models},
  author={Gao, Yiming and Wang, Bin and Wei, Chengwei and Sun, Shuo and Aw, AiTi},
  journal={arXiv preprint arXiv:2505.16774},
  year={2025}
}
```
