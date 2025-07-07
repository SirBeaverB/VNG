# **Variable Node Graph (VNG)**

*Welcome to VNG!*

This is the repository for VNG, a Markov-chain-based dyanamic graph convolution network. 

[VNG: A Markov-Chain-Based Dynamic Graph Convolution Network](https://github.com/SirBeaverB/VNG/blob/main/VNG_arxiv.pdf)

Jialing Bi, Yanping Zheng, Zhifei Li, Zhewei Wei*

Renmin University of China, Beijing, China, 100872

{bijialing22, zhengyanping, ZhifeiLi, zhewei}@ruc.edu.cn



## **Abstract**
_Decoupled Graph Convolutional Networks (GCNs), which decouples the feature transformation and information propagation stages for node representation learning, have achieved great success and become the latest paradigm of GCNs. However, a significant limitation of current decoupled GCNs is their focus on fixed and static graphs. While some recent approaches have attempted to address evolving graphs, they mainly focus on edge-level dynamics and assume static node numbers and features, limiting their applicability to real-world data. In this paper, we introduce Variable Node GCN (VNG), a novel framework capable of addressing both edge-level and node-level dynamics. Our core insight is that the iterative computation of information propagation in decoupled GCNs can be reformulated as a homogeneous first-order Markov chain through specific mathematical transformations. Building on this, we employ iterative aggregation techniques for decoupling Markov chains to effectively update node representations in evolving graph structures. This approach provides a flexible solution for both fixed-node and variable-node evolving problems. We present formal proofs and experimental results demonstrating that VNG is both efficient and effective, outperforming state-of-the-art baselines across various datasets._



## **Datasets**
We follow the setup of APPNP and conduct experiments on four benchmark node-classification datasets: Cora_ML, Citeseer, PubMed, Microsoft Academic (MS_Academic), and Movielens.



## **How to run**

To run the code, simply run [main.py](main.py). The code automatically masks the dataset, runs APPNP/PPNP on the updating graph first, and then runs VNG.


## **Reference**
The code refers to [PPNP](https://github.com/gasteigerjo/ppnp), [SDG](https://github.com/DongqiFu/SDG), and [PT_propagation_then_training](https://github.com/DongHande/PT_propagation_then_training).
