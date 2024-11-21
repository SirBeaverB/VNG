# **Variable Node Graph (VNG)**

*Welcome to VNG!*

This is the repository for VNG, a Markov-chain-based dyanamic graph convolution network.

---

## **Introduction**
VNG is a novel technique that reformulates the iterative computation of node representations into a Markov chain framework, enabling efficient updates and precise management of evolving nodes and edges. This Markov-chain-based approach efficiently handles evolving problems in both fixed-node and variable-node scenarios and incrementally updates node representations based on prior results, avoiding needless recomputation while preserving accuracy.

---

## **Datasets**
We follow the setup of APPNP and conduct experiments on four benchmark node-classification datasets: Cora_ML, Citeseer, PubMed, and Microsoft Academic (MS_Academic).

---

## **How to run**

To run the code, simply run [main.py](blob/main/main.py). The code automatically masks the dataset, runs APPNP/PPNP on the updating graph first, and then runs VNG.

---

## **Reference**
The code refers to [PPNP](https://github.com/gasteigerjo/ppnp), [SDG](https://github.com/DongqiFu/SDG), and [PT_propagation_then_training](https://github.com/DongHande/PT_propagation_then_training).
