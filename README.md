Welcome to VNG!

VNG基础代码已完成。运行main.py即可。
目前有4个可以直接运行的数据集，都是结点分类任务，正在进行实验。这部分实验结果将作为VNG的基础结果，也就是说，无论如何我们已经有东西可以往论文里写了！
各位可以帮忙检查一下代码，确定算法实现中没有出错。

TODO:
1. 目前的baseline只有APPNP一个。想增加一些新的baseline。可以参考Instant GCN的baseline和图表。https://arxiv.org/pdf/2206.01379
2. （额外）尝试用VNG解决边预测问题，参考yanping老师的这篇。(https://arxiv.org/pdf/2305.08273)（把stationary distribution随时间变化的过程输入到时序模型（如RNN、LSTM）里……？）
3. （额外的额外）增加recommender system的case study。找一些推荐系统背景的数据集（Mind, MovieLen这种）。
