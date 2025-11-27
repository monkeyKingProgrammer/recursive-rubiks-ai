# Stabilizing Deep Recursive Transformers for Combinatorial Reasoning: A Study on the 2x2 Rubik's Cube

This repository contains the official code, trained weights (`rubik_master.pth`), and experimental logs accompanying the paper titled "Stabilizing Deep Recursive Transformers for Combinatorial Reasoning: A Study on the 2x2 Rubik's Cube."

## 🔬 Core Contribution
We investigate the limits of applying a single, recurrent Transformer block (the Tiny Recursive Model, TRM) to solve combinatorial search problems.

**Key Finding:**
* Training deep recursive networks (Depth 20) directly leads to **model collapse (8.05% accuracy)** due to vanishing gradients.
* Using a **Curriculum of Thought** (gradually increasing recursion depth from 4 to 20) stabilizes training, achieving **~65% next-move accuracy**.
* The model successfully solves all scrambles up to **Depth 10**, demonstrating efficient geometric reasoning.

## 💻 Setup and Dependencies

This project requires Python and PyTorch with CUDA support.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/monkeyKingProgrammer/recursive-rubiks-ai.git
    cd recursive-rubiks-ai
    ```

2.  **Install Requirements:**
    ```bash
    # Ensure you have a working environment with Python 3.x
    pip install torch numpy
    ```

## ▶️ Usage and Reproduction

The main script, `rubik_solver.py`, performs three sequential tests (Depth 5, Depth 10, Depth 20) using the trained weights (`rubik_master.pth`).

1.  **Run the Live Solver Demo:**
    ```bash
    python rubik_solver.py
    ```
    *(Note: The script will load the saved model and immediately begin the live solving tests using Beam Search.)*

2.  **Reproduction of Collapse (Optional):**
    If you wish to reproduce the **8.05% collapse** shown in the paper (Table 1), delete the `rubik_master.pth` file and modify the `CURRICULUM_SCHEDULE` in `rubik_solver.py` to `[20]`.

## 📎 Citation

If you use this code or the findings from this paper in your research, please cite our work:

```bibtex
@article{ChanSeng_TRM_2025,
  author = {Chan Seng Tham},
  title = {Stabilizing Deep Recursive Transformers for Combinatorial Reasoning: A Study on the 2x2 Rubik's Cube},
  journal = {Preprint submitted for review},
  year = {2025},
  url = {https://github.com/monkeyKingProgrammer/recursive-rubiks-ai.git}
}