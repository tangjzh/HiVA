# HiVA: Self-organized Hierarchical Variable Agent via Goal-driven Semantic-Topological Evolution

This repository contains the official implementation of the paper "HiVA: Self-organized Hierarchical Variable Agent via Goal-driven Semantic-Topological Evolution".

## Abstract

Autonomous agents play a crucial role in advancing Artificial General Intelligence, enabling problem decomposition and tool orchestration through Large Language Models (LLMs). However, existing paradigms face a critical trade-off. On one hand, reusable fixed workflows require manual reconfiguration upon environmental changes; on the other hand, flexible reactive loops fail to distill reasoning progress into transferable structures. We introduce Hierarchical Variable Agent (HiVA), a novel framework modeling agentic workflows as self-organized graphs with the Semantic-Topological Evolution (STEV) algorithm, which optimizes hybrid semantic-topological spaces using textual gradients as discrete-domain surrogates for backpropagation. The iterative process comprises Multi-Armed Bandit-infused forward routing, diagnostic gradient generation from environmental feedback, and coordinated updates that co-evolve individual semantics and topology for collective optimization in unknown environments. Experiments on dialogue, coding, Long-context Q&A, mathematical, and agentic benchmarks demonstrate improvements of 5-10% in task accuracy and enhanced resource efficiency over existing baselines, establishing HiVA's effectiveness in autonomous task execution.

## Environment Setup

```bash
# Clone the repository
git clone https://github.com/tangjzh/HiVA.git
cd HiVA

# Create and activate a Conda environment (Python 3.10)
conda create -n hiva python=3.10 -y
conda activate hiva

# Install dependencies
pip install -U pip
pip install -r requirements.txt

```

## Repository Structure

```
hiva/
├── core/         # Core implementation of HiVA
├── engines/      # Processing and evolution engines
├── env/          # Environments
└── utils.py      # Utility functions
```

## Running Experiments

To reproduce our main results:

```bash
python scripts/test_math.py
```

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{tang2025hiva,
  title={HiVA: Self-organized Hierarchical Variable Agent via Goal-driven Semantic-Topological Evolution},
  author={Tang, Jinzhou and Zhang, Jusheng and Lv, Qinhan and Liu, Sidi and Yang, Jing and Tang, Chengpei and Wang, Keze},
  journal={arXiv preprint arXiv:2509.00189},
  year={2025}
}
```

## License

MIT
