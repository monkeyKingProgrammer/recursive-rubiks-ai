---
title: 'Stabilizing Deep Recursive Transformers for Combinatorial Reasoning: A Study on the 2x2 Rubiks Cube'
tags:
  - Python
  - PyTorch
  - Machine Learning
  - Artificial Intelligence
  - Combinatorial Optimization
authors:
  - name: Chan Seng Tham
    affiliation: 1
affiliations:
 - name: Independent Researcher
   index: 1
date: 28 November 2025
bibliography: paper.bib
---

# Summary

Recent developments in Large Language Models (LLMs) suggest that "Test-Time Compute"—allowing a model to process information iteratively before outputting a response—appears increasingly important for complex reasoning. However, training models to utilize deep recurrent loops often leads to optimization instability and the vanishing gradient problem. In this work, we introduce the **Tiny Recursive Model (TRM)**, a parameter-efficient Transformer (approx. 800k parameters) designed to solve the 2x2 Rubik’s Cube by refining its internal state over 20 discrete time steps. We demonstrate that under standard optimization settings (Adam, lr=3e-4), initializing the model with a static recursion depth of 20 results in catastrophic model collapse (**< 9% accuracy**, equivalent to random guessing). To overcome this, we propose a **Curriculum of Thought** training strategy, progressively increasing recursion depth from 4 to 20. This method stabilizes training, achieving **approx. 65% next-move accuracy**. Empirical tests on 100 random scrambles per depth reveal a distinct phase transition: the model achieves a 100% solve rate on Depth 10 scrambles, often finding non-trivial shortcuts, but struggles with "Greedy Loops" on Depth 20 scrambles. This suggests a critical boundary beyond which recursive supervised learning likely requires augmentation by look-ahead search or reinforcement learning.

# Statement of Need

The scaling laws of deep learning have traditionally focused on increasing parameter counts and dataset sizes [@kaplan2020scaling]. However, a new paradigm is emerging: The **Test-Time Compute** hypothesis [@graves2016adaptive; @wei2022chain] posits that for specific classes of problems—such as arithmetic, logic puzzles, and combinatorial planning—performance scales not just with model size but also with the duration of inference.

Implementing this capability requires an architecture that can apply a transformation function *f* repeatedly to its own output: *h(t+1) = f(h(t))*. While Recurrent Neural Networks (RNNs) and Universal Transformers [@dehghani2019universal] offer this capability, training them over long horizons is notoriously difficult. As the recursion depth *N* increases, backpropagating gradients through time (BPTT) becomes unstable [@pascanu2013difficulty], leading to vanishing or exploding gradients that prevent the model from learning the relationship between the input state and the desired output.

In this paper, we use the 2x2 Rubik's Cube as a proxy for complex non-linear reasoning. We explore:

1.  **The Limits of Recursion:** We empirically show that a Transformer fails to learn when initialized at a high recursion depth (*N=20*) using standard optimization.
2.  **Curriculum Stabilization:** We demonstrate that progressively increasing *N* during training allows the model to learn stable, identity-preserving transformations [@bengio2009curriculum].
3.  **The Greedy Trap:** We analyze the failure mode of the model at high depths, showing that even a well-trained recursive network falls into repetitive loops without a value function, highlighting the necessity of beam search or tree search for long-horizon planning [@agostinelli2019solving].

# Methodology

## The Physics Environment
We utilize a simulated 2x2 "Pocket Cube." The state space is represented by a flattened vector mapping the 24 stickers to 6 discrete colors.

### Action Notation
The action space consists of 12 atomic moves based on standard Singmaster notation. We do not include double turns (e.g., R2) in the action set:
* **Faces:** **U** (Up), **D** (Down), **L** (Left), **R** (Right), **F** (Front), **B** (Back).
* **Direction:** A standalone letter (e.g., **R**) denotes a 90-degree Clockwise turn. A letter followed by an apostrophe (e.g., **R'**) denotes a 90-degree Counter-Clockwise turn.

## The Tiny Recursive Model (TRM)
The TRM consists of a **single** Transformer block [@vaswani2017attention] with weights shared across time steps. This separates parameter count from computational depth.

The forward pass is defined as:
1.  *h(0)* = Embedding(Input) + PositionalEmbedding
2.  *h(t)* = TransformerBlock(*h(t-1)*) ... repeated for *N* steps
3.  Output = Softmax(Head(*h(N)*))

**Model Specifications:**
* **Embedding Dimension:** 256
* **Attention Heads:** 8
* **Feed-Forward Dimension:** 1024
* **Total Parameters:** Approx. 800,000

## Training Regimes
We compared two training regimes to validate the curriculum hypothesis:
1.  **Baseline (Static Deep Training):** The model is initialized and trained immediately with a fixed recursion depth of *N=20* for 5,000 steps.
2.  **Ours (Curriculum of Thought):** The model begins training at *N=4*. After 2,000 steps per level, the depth is increased to *N = 5, 6, 8, 12, 16, 20*.

## Hardware and Software Setup
All experiments were conducted on a desktop workstation configured for deep learning tasks (Intel i5-14400F, 32GB RAM, NVIDIA GeForce RTX 4060 Ti 16GB). The software environment utilizes PyTorch 2.0 (CUDA 11.8).

## Training Details
To ensure reproducibility, we specify the exact hyperparameters used:
* **Optimizer:** AdamW [@kingma2014adam] with learning rate 3e-4.
* **Loss Function:** Cross-Entropy Loss on the next-move prediction.
* **Batch Size:** 512.
* **Total Training Steps:** Baseline (5,000 steps at fixed N=20); Curriculum (12,000 steps total).
* **Note on Compute:** The curriculum regimen utilizes more total parameter updates (12k vs 5k). Our goal is not strict compute parity, but to study whether a depth curriculum can prevent collapse at high recursion depths.

# Experiments and Results

## Experiment 1: The Stability Gap
We trained both the Baseline and the Curriculum model. We then evaluated both models on their ability to solve scrambles up to a 20-step horizon.

**Table 1: Comparison of model performance when targeting Depth 20.**

| Target Inference Depth | Training Strategy | Final Accuracy | Status |
| :--- | :--- | :--- | :--- |
| **20** | **Static Initialization** (Start at 20) | 8.05% | **Collapsed** |
| **20** | **Curriculum** (Grow 4 -> 20) | **64.84%** | **Converged** |

*Note: 8.33% represents random guessing (1 out of 12).*

## Experiment 2: Inference Capabilities
We subjected the trained Curriculum model to "Live Solving" tests on unseen scrambles. For each target depth *d* in {5, 10, 20}, we generated 100 test scrambles. We decoded the solutions using **Beam Search (width 8)**.

**Result A: Short Horizon (Depth 5)**
* **Solve Rate:** 100%
* **Efficiency:** The model consistently solved the cube in 5 moves or fewer.

**Result B: Medium Horizon (Depth 10)**
* **Solve Rate:** 100%
* **Efficiency:** The model frequently found shortcuts.

**Result C: Long Horizon (Depth 20)**
* **Solve Rate:** ~10% (Greedy/Beam Search)
* **Failure Mode:** "Greedy Loops."
* *Observation:* At high depths, the model enters repetitive cycles (e.g., R -> L -> R -> L). Despite high confidence in individual moves (> 40%), the lack of a global value function prevents the model from escaping local minima in the state space.

# Discussion

Our results highlight a crucial dichotomy in Neural Reasoning.

**1. Recursion Stabilizes Representation:** By forcing the model to iterate on its own output, we successfully trained a "System 2" simulator that refines its state. The curriculum approach is essential for this; without it, the bridge between input and output is too long for the gradient to traverse during initialization.

**2. The Limits of Supervised Greedy Search:** The failure at Depth 20 indicates that "Next-Token Prediction" (or Next-Move Prediction) is insufficient on its own for long-horizon planning in this environment without search or a learned value function. While the model knows *valid* moves, it lacks a mechanism to judge *progress* toward the solved state.

**Conclusion:** We successfully stabilized the training of deep recursive networks using Curriculum Learning. While the TRM excels at short-to-medium term reasoning (Depth 10), solving deep combinatorial problems (Depth 20+) likely requires augmenting the recursive "intuition" with explicit search mechanisms (MCTS) or Reinforcement Learning value estimators.

# References