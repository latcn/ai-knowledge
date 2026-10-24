## 目录

1. Attention 与 FFN 的位置关系：常见误解与校正
2. FFN 的具体结构：它才是“全连接 + 激活”的地方
3. 全连接层参数计数：每个节点有多少权重和偏置
4. FFN 第一层的逐节点计算与矩阵写法
5. 权重矩阵 $$W_{j,k}$$ 的行列约定：一行还是一个列对应一个节点？
6. 数值例子：完整走一遍 FFN 第一层
7. 序列情形：$$T$$ 个 token 时参数量不变
8. 现代 LLM 的 FFN 变体：SwiGLU 与无 bias
9. Pre-LN 与 Post-LN：现代 LLM 为何以 Pre-LN 为主
10. 与第 9–10 天反向传播文档的衔接
11. 全部问答速查表
12. 最终总结

---

# 1. Attention 与 FFN 的位置关系：常见误解与校正

## 1.1 原始描述

> 输入向量经过自注意层 QKV 计算，获取上下文自注意向量 $$Z$$。$$Z$$ 通过激活函数计算获取 $$a$$，$$a$$ 转到下一层，FFN 在哪？$$a$$ 传到下一层，影响下一层每个节点，这时一个节点（假设上层 $$L$$ 个节点，那一个节点就有 $$n$$ 个权重系数，一个 $$b$$ 偏置？）

## 1.2 校正：$$Z$$ 不是直接过逐元素激活进入“下一层全连接”

实际 Transformer block（**本文统一采用 Pre-LN**）是：

$$
X' = X + \mathrm{MultiHeadAttention}(\mathrm{LayerNorm}(X))
$$
$$
X_{\text{out}} = X' + \mathrm{FFN}(\mathrm{LayerNorm}(X'))
$$

> **修订说明（Pre-LN/Post-LN 统一）**：原稿开头声明“以 Pre-LN 为例”，但“更细拆开”部分误写成 $$X'=\mathrm{LayerNorm}(X+Z_{\text{out}})$$（这是 Post-LN 的 Attention 子层写法），随后又用 Pre-LN 的 FFN 写法，形成混合体。本修订版统一采用 **Pre-LN**：Attention 子层和 FFN 子层都先对输入做 LayerNorm，再进入子层，最后加残差。Post-LN 的写法在 9.3 节单独说明。

更细地拆开：

1. **QKV 与自注意力**  
   输入 $$X\in\mathbb{R}^{T\times d_{\text{model}}}$$（$$T$$ 是序列长度，$$d_{\text{model}}$$ 是每个 token 的向量维度）：
   $$
   Q=XW^Q,\quad K=XW^K,\quad V=XW^V
   $$
   $$
   Z=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
   $$
   这里 $$Z$$ 是**上下文自注意力向量**，形状仍是 $$T\times d_{\text{model}}$$（多头时每个头维度更小，拼起来再投影）。

2. **输出投影与残差 + LayerNorm（Pre-LN）**  
   $$Z$$ 先经过输出投影：
   $$
   Z_{\text{out}}=ZW^O
   $$
   然后加残差：
   $$
   X'=X+Z_{\text{out}}
   $$
   **注意（Pre-LN）**：LayerNorm 是在进入 Attention 之前做的，即 $$\mathrm{MHA}(\mathrm{LN}(X))$$ 中的 $$\mathrm{LN}(X)$$；残差加回的是原始 $$X$$，不是归一化后的值。这里没有“逐元素激活函数”作用于 $$Z$$。注意力打分与加权机制内部的主要非线性是 softmax（用于注意力权重归一化），不是 ReLU/GELU 那种逐元素激活。

   > **修订说明（“唯一非线性是 softmax”）**：原稿称“自注意力内部唯一的非线性是 softmax”，不够严谨。精确说法是：**在注意力打分与加权机制内部，主要的非线性是 softmax**；整个 Attention 子层还包含 LayerNorm，LayerNorm 的归一化（除法、平方、开方）也引入非线性，但它不是 ReLU/GELU 式逐元素激活。不要把“没有逐元素激活”说成“没有非线性”。

3. **FFN 在哪里？**  
   FFN 紧接着 Attention 之后，是**同一个 Transformer block 的后半部分**（Pre-LN）：
   $$
   X_{\text{out}}=X'+\mathrm{FFN}(\mathrm{LayerNorm}(X'))
   $$
   所以 FFN 不是“Attention 的下一层网络”，而是 Attention 后面的**前馈子层**。一个 Transformer block = Attention 子层 + FFN 子层，两者各带残差；Pre-LN 中 LayerNorm 位于每个子层的输入端。

4. **“$$a$$ 传到下一层”应如何理解？**  
   在 Pre-LN 中，如果把 $$X'=X+Z_{\text{out}}$$ 称为 $$a$$，那么 $$a$$ 进入的是 **FFN**（更准确地说，$$a$$ 先经过 $$\mathrm{LayerNorm}$$ 再进入 FFN），不是“下一层 Transformer block”。FFN 内部才包含“全连接 + 激活 + 全连接”。

## 1.3 一句话结论

> **Attention 负责混合不同位置的信息，FFN 负责对每个位置做非线性特征变换；FFN 不是 Attention 的“下一层”，而是同一 block 的后半部分。现代 LLM 默认采用 Pre-LN。**

---

# 2. FFN 的具体结构：它才是“全连接 + 激活”的地方

## 2.1 原始 Transformer 的 FFN

对序列中**每个位置独立**（position-wise）作用：

$$
\mathrm{FFN}(x)=\sigma(xW_1+b_1)W_2+b_2
$$

其中：

- $$x\in\mathbb{R}^{d_{\text{model}}}$$：一个 token 的向量；
- $$W_1\in\mathbb{R}^{d_{\text{model}}\times d_{\text{ff}}}$$，$$b_1\in\mathbb{R}^{d_{\text{ff}}}$$（**行向量约定**）；
- $$\sigma$$：逐元素激活（ReLU 或 GELU）；
- $$W_2\in\mathbb{R}^{d_{\text{ff}}\times d_{\text{model}}}$$，$$b_2\in\mathbb{R}^{d_{\text{model}}}$$；
- 通常 $$d_{\text{ff}}=4d_{\text{model}}$$。

所以 FFN 是**两层全连接 + 中间一个逐元素激活**。用户描述的“上层 $$L$$ 个节点 → 下一层每个节点有若干权重和 1 个偏置”，对应的正是 FFN 的第一层线性变换 $$xW_1+b_1$$。

> **修订说明（$$W_1$$ 形状约定切换）**：本节采用**行向量约定**，$$x$$ 是行向量，$$W_1\in\mathbb{R}^{d_{\text{model}}\times d_{\text{ff}}}$$。第 3.4 节将采用**列向量约定**，$$W_1\in\mathbb{R}^{d_{\text{ff}}\times d_{\text{model}}}$$。两者是转置关系，参数量相同。切换处会显式标注，避免快速阅读时误以为前后矛盾。

## 2.2 现代 LLM 的 FFN：SwiGLU

很多现代 LLM（LLaMA、PaLM 等）用：

$$
\mathrm{FFN}(x)=(\mathrm{Swish}(xW_{\text{gate}})\odot xW_{\text{up}})W_{\text{down}}
$$

其中 $$W_{\text{gate}},W_{\text{up}}\in\mathbb{R}^{d_{\text{model}}\times d_{\text{ff}}}$$，$$W_{\text{down}}\in\mathbb{R}^{d_{\text{ff}}\times d_{\text{model}}}$$，通常 $$d_{\text{ff}}=\frac{8}{3}d_{\text{model}}$$ 以保持参数量与原始 FFN 相当。

**SwiGLU 常不带偏置**；LLaMA 的 attention 投影和 FFN 线性层也都不带 bias。

> **修订说明（RMSNorm 与 bias）**：原稿写“偏置功能由 RMSNorm 的可学习缩放替代”，这不准确。RMSNorm 只有逐维可学习缩放 $$\gamma$$，**没有加性平移项 $$\beta$$**；乘性缩放不能直接替代加性偏置。LLaMA 等去掉线性层 bias 的主要依据是**工程实践/经验简化**：去 bias 对性能影响很小，且有利于训练稳定性和效率。正确表述应为：“LLaMA 等模型的线性层通常省略 bias，这是实践中的简化，不是由 RMSNorm 替代偏置。RMSNorm 只提供逐维缩放，没有加性平移项。”另外，LLaMA 用的是 RMSNorm，不是 LayerNorm，不应混写为“RMSNorm/LayerNorm”。

---

# 3. 全连接层参数计数：每个节点有多少权重和偏置

## 3.1 符号约定

设：

- **输入维度** $$L$$：上一层（或当前层输入向量）的节点数；
- **输出维度** $$n$$：下一层的节点数。

则全连接层的权重矩阵：

$$
W\in\mathbb{R}^{n\times L}
$$

偏置：

$$
b\in\mathbb{R}^{n}
$$

## 3.2 每个输出节点的参数

**每个输出节点** $$j$$（$$j=1,\dots,n$$）：

- 连接到输入的所有 $$L$$ 个节点，所以有 **$$L$$ 个权重** $$W_{j,1},\dots,W_{j,L}$$；
- 有 **1 个偏置** $$b_j$$。

正确说法是：

> 上层 $$L$$ 个节点，下一层每个节点有 **$$L$$ 个权重 + 1 个偏置**；整个层有 $$n\times L$$ 个权重 + $$n$$ 个偏置。

## 3.3 常见符号混淆的纠正

用户原话：

> 假设上层 $$L$$ 个节点，那一个节点就有 $$n$$ 个权重系数，一个 $$b$$ 偏置？

这里把输入维度和输出维度混了：如果“上层 $$L$$ 个节点”，每个下一层节点应有 $$L$$ 个权重，而不是 $$n$$ 个。若把 $$n$$ 定义为输入维度，则“$$n$$ 个权重”是对的；但用户又写“上层 $$L$$ 个节点”，符号不一致。

**正确表述**：

> 若上层有 $$L$$ 个节点、下层有 $$n$$ 个节点，则**每个下层节点有 $$L$$ 个权重 + 1 个偏置**；整个层有 $$nL$$ 个权重 + $$n$$ 个偏置。

> **修订说明（符号 $$L$$ 重载）**：本节用 $$L$$ 表示“输入维度”，第 4 节用 $$L_1$$ 表示“FFN 第一层节点数/输出维度”。同一个字母族承担不同角色，容易混淆。阅读时请注意：第 3 节的 $$L$$ 是输入维度，第 4 节的 $$L_1$$ 是 FFN 隐藏层节点数；二者不是同一个量。若需统一，可用 $$d_{\text{in}}$$ 和 $$d_{\text{out}}$$ 替代。

## 3.4 对应到 FFN 第一层（列向量约定）

> **约定切换提示**：以下采用**列向量约定**，$$x\in\mathbb{R}^{d_{\text{model}}}$$ 是列向量，$$W_1\in\mathbb{R}^{d_{\text{ff}}\times d_{\text{model}}}$$。这与 2.1 节的行向量约定互为转置，参数量相同。

- 输入维度 $$L=d_{\text{model}}$$；
- 输出维度 $$n=d_{\text{ff}}$$；
- 权重 $$W_1\in\mathbb{R}^{d_{\text{ff}}\times d_{\text{model}}}$$，共 $$d_{\text{ff}}\times d_{\text{model}}$$ 个；
- 偏置 $$b_1\in\mathbb{R}^{d_{\text{ff}}}$$，共 $$d_{\text{ff}}$$ 个；
- 每个隐藏节点有 $$d_{\text{model}}$$ 个权重 + 1 个偏置。

## 3.5 对应到 FFN 第二层

- 输入维度 $$L=d_{\text{ff}}$$；
- 输出维度 $$n=d_{\text{model}}$$；
- 权重 $$W_2\in\mathbb{R}^{d_{\text{model}}\times d_{\text{ff}}}$$；
- 偏置 $$b_2\in\mathbb{R}^{d_{\text{model}}}$$。

---

# 4. FFN 第一层的逐节点计算与矩阵写法

## 4.1 问题

> 假设 FFN 第一层有 $$L_1$$ 个节点，输入向量是 $$X$$，那么这个是怎么计算的？第一层每个节点都和输入向量 $$X$$ 计算 $$(xW_1+b_1)$$ 和激活值？哪些输入向量的维度 $$d_x$$，第一层则有 $$d_x\times L_1$$ 加上 $$L_1$$ 个偏置个权重系数？

## 4.2 结论先行

设输入向量 $$x\in\mathbb{R}^{d_x}$$，FFN 第一层有 $$L_1$$ 个节点。则：

$$
z=W_1x+b_1,\qquad a=\sigma(z)
$$

其中：

- $$W_1\in\mathbb{R}^{L_1\times d_x}$$：权重矩阵（**列向量约定**）；
- $$b_1\in\mathbb{R}^{L_1}$$：偏置向量；
- $$z\in\mathbb{R}^{L_1}$$：第一层的 pre-activation；
- $$a\in\mathbb{R}^{L_1}$$：第一层激活后的输出；
- $$\sigma$$：逐元素激活（ReLU/GELU/SwiGLU 中的 Swish 等）。

**参数量**：

- 权重系数：$$L_1\times d_x$$ 个；
- 偏置系数：$$L_1$$ 个；
- 总可学习参数：$$L_1d_x+L_1=L_1(d_x+1)$$。

用户说法“第一层则有 $$d_x\times L_1$$ 加上 $$L_1$$ 个偏置”**基本正确**，只需把“权重系数”和“偏置系数”分开说清楚。

## 4.3 逐节点看：每个节点到底在算什么

第一层第 $$j$$ 个节点（$$j=1,\dots,L_1$$）的计算是：

$$
z_j=\sum_{k=1}^{d_x}W_{1,j,k}\,x_k+b_{1,j}
$$
$$
a_j=\sigma(z_j)
$$

含义：

1. **每个节点都和整个输入向量 $$x$$ 计算**：不是只连一个输入分量，而是对 $$x$$ 的所有 $$d_x$$ 个分量做加权求和；
2. 第 $$j$$ 个节点有 **$$d_x$$ 个权重** $$W_{1,j,1},\dots,W_{1,j,d_x}$$，对应输入 $$x$$ 的 $$d_x$$ 个分量；
3. 再加 **1 个偏置** $$b_{1,j}$$；
4. 最后过激活函数 $$\sigma$$ 得到 $$a_j$$。

所以“每个节点有 $$d_x$$ 个权重 + 1 个偏置”是对的；整个第一层有 $$L_1$$ 个节点，因此：

$$
\underbrace{L_1\times d_x}_{\text{权重}}+\underbrace{L_1}_{\text{偏置}}
$$

## 4.4 矩阵写法与维度检查

列向量约定：

$$
x\in\mathbb{R}^{d_x},\quad
W_1\in\mathbb{R}^{L_1\times d_x},\quad
b_1\in\mathbb{R}^{L_1}
$$
$$
z=W_1x+b_1\in\mathbb{R}^{L_1}
$$

维度检查：

$$
(L_1\times d_x)(d_x\times1)=L_1\times1
$$

再加 $$b_1\in\mathbb{R}^{L_1}$$，结果仍是 $$\mathbb{R}^{L_1}$$，正确。

行向量约定（很多框架内部用）：

$$
x^\top\in\mathbb{R}^{1\times d_x},\quad
W_1\in\mathbb{R}^{d_x\times L_1}
$$
$$
z^\top=x^\top W_1+b_1^\top\in\mathbb{R}^{1\times L_1}
$$

两种约定只是转置关系，参数量都是 $$d_xL_1+L_1$$。

---

# 5. 权重矩阵 $$W_{j,k}$$ 的行列约定：一行还是一个列对应一个节点？

## 5.1 问题

> $$W_{j,k}$$ 中一行代表一个节点的 $$d_x$$ 个权重系数？

## 5.2 列向量约定下：一行对应一个节点

在列向量约定下，$$W_1\in\mathbb{R}^{L_1\times d_x}$$，第 $$j$$ 行就是第 $$j$$ 个节点的 $$d_x$$ 个权重系数。

具体写开：

$$
W_1=
\begin{bmatrix}
W_{1,1} & W_{1,2} & \cdots & W_{1,d_x}\\
W_{2,1} & W_{2,2} & \cdots & W_{2,d_x}\\
\vdots & \vdots & \ddots & \vdots\\
W_{L_1,1} & W_{L_1,2} & \cdots & W_{L_1,d_x}
\end{bmatrix}
$$

- **第 $$j$$ 行**：$$(W_{j,1},W_{j,2},\dots,W_{j,d_x})$$，就是第 $$j$$ 个节点连接输入 $$x$$ 的 $$d_x$$ 个分量的权重；
- **第 $$k$$ 列**：$$(W_{1,k},W_{2,k},\dots,W_{L_1,k})^\top$$，是所有 $$L_1$$ 个节点对输入第 $$k$$ 个分量的权重。

所以：

$$
z_j=\sum_{k=1}^{d_x}W_{j,k}x_k+b_j
$$

第 $$j$$ 个节点用第 $$j$$ 行与 $$x$$ 做内积，再加偏置 $$b_j$$，过激活得 $$a_j$$。

## 5.3 行向量约定下：一列对应一个节点

很多框架/教材把输入写成行向量：

$$
x^\top\in\mathbb{R}^{1\times d_x},\qquad
W_1\in\mathbb{R}^{d_x\times L_1}
$$
$$
z^\top=x^\top W_1+b_1^\top
$$

这时：

- **第 $$j$$ 列** 才是第 $$j$$ 个节点的 $$d_x$$ 个权重；
- 一行对应的是“输入某个分量对所有节点的权重”。

## 5.4 两种约定的对照表

两种写法只是转置关系，参数量都是 $$L_1d_x+L_1$$。**关键是看 $$W$$ 的形状和乘法顺序：**

| 约定 | $$W$$ 形状 | 乘法 | 一个节点的权重在哪 |
|---|---|---|---|
| 列向量 | $$L_1\times d_x$$ | $$z=Wx+b$$ | **一行** |
| 行向量 | $$d_x\times L_1$$ | $$z^\top=x^\top W+b^\top$$ | **一列** |

用户前面写的是 $$z=W_1x+b_1$$、$$W_1\in\mathbb{R}^{L_1\times d_x}$$，所以**一行代表一个节点的 $$d_x$$ 个权重**，用户的理解正确。

---

# 6. 数值例子：完整走一遍 FFN 第一层

## 6.1 设定

设 $$d_x=3$$，$$L_1=2$$，输入：

$$
x=\begin{bmatrix}1\\2\\-1\end{bmatrix}
$$

权重和偏置：

$$
W_1=\begin{bmatrix}
1&0&-1\\
2&1&0
\end{bmatrix},\qquad
b_1=\begin{bmatrix}0.5\\-1\end{bmatrix}
$$

## 6.2 第 1 个节点

$$
z_1=1\cdot1+0\cdot2+(-1)\cdot(-1)+0.5=1+0+1+0.5=2.5
$$
$$
a_1=\sigma(2.5)
$$

若 $$\sigma=\mathrm{ReLU}$$，$$a_1=2.5$$。

## 6.3 第 2 个节点

$$
z_2=2\cdot1+1\cdot2+0\cdot(-1)+(-1)=2+2+0-1=3
$$
$$
a_2=\sigma(3)
$$

若 $$\sigma=\mathrm{ReLU}$$，$$a_2=3$$。

## 6.4 结果

$$
z=\begin{bmatrix}2.5\\3\end{bmatrix},\qquad
a=\begin{bmatrix}2.5\\3\end{bmatrix}
$$

参数量：权重 $$2\times3=6$$ 个，偏置 $$2$$ 个，总 $$8$$ 个。每个节点有 $$3$$ 个权重 + $$1$$ 个偏置。

---

# 7. 序列情形：$$T$$ 个 token 时参数量不变

Transformer 中输入是 $$X\in\mathbb{R}^{T\times d_x}$$（$$T$$ 个 token，每个 $$d_x$$ 维）。FFN 对**每个位置独立、共享同一组 $$W_1,b_1$$**：

$$
Z=XW_1+\mathbf{1}_T b_1^\top
$$

- $$Z\in\mathbb{R}^{T\times L_1}$$；
- 第 $$t$$ 行 $$Z_t$$ 就是第 $$t$$ 个 token 的 $$z$$；
- 参数量不随 $$T$$ 增加，仍是 $$d_xL_1+L_1$$。

所以“每个节点都和输入向量 $$x$$ 计算”在序列情形下是“每个节点和**每个 token 的输入向量**分别计算，权重共享”。

---

# 8. 现代 LLM 的 FFN 变体：SwiGLU 与无 bias

## 8.1 SwiGLU 结构

$$
\mathrm{FFN}(x)=(\mathrm{Swish}(xW_{\text{gate}})\odot xW_{\text{up}})W_{\text{down}}
$$

- $$W_{\text{gate}},W_{\text{up}}\in\mathbb{R}^{d_{\text{model}}\times d_{\text{ff}}}$$；
- $$W_{\text{down}}\in\mathbb{R}^{d_{\text{ff}}\times d_{\text{model}}}$$；
- 通常 $$d_{\text{ff}}=\frac{8}{3}d_{\text{model}}$$，以保持参数量与原始 FFN（$$d_{\text{ff}}=4d_{\text{model}}$$）相当。该比例出自 Shazeer (2020) *GLU Variants Improve Transformer*：SwiGLU 有 3 个矩阵，为与原始 FFN（2 个矩阵、$$4d_{\text{model}}$$）参数量相当，取 $$d_{\text{ff}}=\frac23\cdot4d_{\text{model}}=\frac83d_{\text{model}}$$。

## 8.2 无 bias 的实践

- LLaMA 的 attention 投影和 FFN 线性层**都不带 bias**；
- **修订说明**：这主要是工程实践/经验简化——去 bias 对性能影响很小，且有利于训练稳定性和效率。**不是**“由 RMSNorm 的可学习缩放替代偏置”。RMSNorm 只有逐维缩放 $$\gamma$$，没有加性平移 $$\beta$$，不能直接替代 bias。LLaMA 用的是 RMSNorm，不是 LayerNorm，不应混写为“RMSNorm/LayerNorm 替代”。
- 这不改变“每个隐藏节点对输入所有分量加权求和”的结构，只是省去了 $$b$$ 项。

## 8.3 与反向传播的衔接

若 $$b=0$$，反向传播中只需删去 $$g_b$$ 项；$$g_W=\delta a^\top$$ 不变。

---

# 9. Pre-LN 与 Post-LN：现代 LLM 为何以 Pre-LN 为主

## 9.1 结论

**是的，现代大语言模型（LLM）的架构选择已明确以 Pre-LN（Pre-LayerNorm）为主流。** 自 GPT-2 和 LLaMA 系列开始，Pre-LN 已成为绝大多数 Decoder-only 模型的默认归一化方案。这一转变并非因为 Pre-LN 在所有方面都更优越，而是它在**超深网络训练稳定性**上的决定性优势，使其成为大规模模型训练事实上的“安全默认值”。

## 9.2 核心差异：残差路径是否被“污染”

两者的根本区别在于 LayerNorm 相对于残差连接的位置，这直接决定了反向传播时梯度回传的路径。

**Post-LN（原始 Transformer 方案）**
$$
x_{l+1}=\mathrm{LN}(x_l+f_l(x_l))
$$
残差连接**必须穿过** LayerNorm 层才能继续向后传递。这意味着在反向传播时，梯度每经过一个 block，都必须乘以 LayerNorm 的导数 $$\mathrm{LN}'$$。

**Pre-LN（现代 LLM 主流方案）**
$$
x_{l+1}=x_l+f_l(\mathrm{LN}(x_l))
$$
残差连接是一条**无阻碍的恒等路径**（identity path），LayerNorm 只作用于旁路的子层输入 $$f_l$$ 上。在反向传播时，梯度可以沿着这条纯恒等路径，无衰减地从深层直接流向浅层嵌入。

## 9.3 Pre-LN 与 Post-LN 的完整公式对照

**Pre-LN（本文采用）**：
$$
X'=X+\mathrm{MHA}(\mathrm{LN}(X))
$$
$$
X_{\text{out}}=X'+\mathrm{FFN}(\mathrm{LN}(X'))
$$

**Post-LN（原始 Transformer）**：
$$
X'=\mathrm{LN}(X+\mathrm{MHA}(X))
$$
$$
X_{\text{out}}=\mathrm{LN}(X'+\mathrm{FFN}(X'))
$$

> **修订说明**：原稿在 1.2 节混用了两种写法：Attention 子层写成 Post-LN 的 $$X'=\mathrm{LN}(X+Z_{\text{out}})$$，FFN 子层写成 Pre-LN 的 $$X_{\text{out}}=X'+\mathrm{FFN}(\mathrm{LN}(X'))$$。本修订版统一采用 Pre-LN；Post-LN 的写法仅在本节作为对照列出。两者不可混用。

## 9.4 利弊权衡：稳定性 vs. 深层有效性

### Post-LN 的利与弊

**利：浅层网络性能更优，深层梯度信号更强**
- **浅层表现**：在层数较少（例如 6 层或更少）的 Transformer 中，Post-LN 一贯能取得比 Pre-LN 更好的性能。
- **深层梯度**：Post-LN 倾向于在**更高层（靠近输出）保留更大的梯度范数**，这有助于这些层进行有效训练。

**弊：深层网络梯度消失，训练极不稳定**
- **梯度消失**：由于残差路径被 LayerNorm “污染”，在深层网络中，梯度在向浅层回传时会逐渐消失，导致浅层参数难以得到有效更新。
- **训练脆弱**：这使得 Post-LN 在训练超过 10 层的 Transformer 时极易变得不稳定，甚至无法收敛，通常需要精细的学习率预热（warm-up）策略来勉强维持。

### Pre-LN 的利与弊

**利：训练稳定性极高，支持超深堆叠**
- **稳定梯度流**：恒等残差路径保证了梯度从损失函数到最浅层嵌入的顺畅传播，从根本上解决了深层网络的梯度消失问题，使训练过程对学习率等超参数远不敏感。
- **深度可扩展性**：这是 Pre-LN 成为主流的决定性因素。它使得研究者可以稳定地训练数百甚至上千层的 Transformer 网络。

**弊：深层贡献衰减与表示坍缩风险**
- **深层“惰性”**：Pre-LN 的稳定性是有代价的。研究发现，随着层数加深，**深层网络的梯度范数会逐渐减小**，导致这些深层的参数更新缓慢，对最终模型的贡献有限，甚至可以被剪枝而不太影响性能。这被称为 Pre-LN 的“深层有效性”问题。
- **表示坍缩**：有研究指出，Pre-LN 的特定残差结构可能导致“表示坍缩”（representation collapse）问题，限制了模型容量的充分利用。

## 9.5 总结与当前前沿

Pre-LN 的流行，是在“训练可行性”和“理论最优性”之间做出的务实选择。对于需要稳定训练的千亿参数级模型，Post-LN 的梯度消失风险是不可接受的，因此 Pre-LN 的稳定性成为首要考量。

不过，Pre-LN 并非完美终点，社区正在探索结合两者优势的改进方案：

- **Mix-LN**：在浅层使用 Post-LN 以保留更强的浅层梯度，在深层使用 Pre-LN 以保证稳定性，从而让整个网络的梯度分布更均匀。
- **KEEL**：通过将 Post-LN 中的残差路径替换为 Highway 式连接，旨在保留 Post-LN 深层梯度优势的同时，实现超过 1000 层的稳定训练。
- **Peri-LN / ResiDual**：这些方案通过同时归一化子层的输入和输出，或融合两种残差连接，来寻求稳定性和表达性之间更好的平衡。

总的来说，**Pre-LN 是当前大模型训练的“标准答案”**，因为它保证了“模型能训练起来”这个最根本的前提。而对其深层有效性不足的改进，则是当前架构研究的前沿方向之一。

---

# 10. 与第 9–10 天反向传播文档的衔接

## 10.1 FFN 是标准公式的“完全匹配”场景

原始 Transformer 的 FFN：

$$
\mathrm{FFN}(x)=\sigma(xW_1+b_1)W_2+b_2
$$

拆成两层：

- 第 1 层：$$z^{[1]}=xW_1+b_1$$，$$a^{[1]}=\sigma(z^{[1]})$$
- 第 2 层：$$z^{[2]}=a^{[1]}W_2+b_2$$，输出（若恒等）

这与文档中的 $$z^{[\ell]}=W^{[\ell]}a^{[\ell-1]}+b^{[\ell]}$$、$$a^{[\ell]}=S^{[\ell]}(z^{[\ell]})$$ **逐符号对应**。

> **FFN 子层的反向传播，就是第 9–10 天公式的直接套用。**

## 10.2 Attention 中的线性层也用同一套公式，但 Softmax 是例外

Attention 子层内部包含多个线性层：

- $$Q=XW^Q$$、$$K=XW^K$$、$$V=XW^V$$：都是仿射变换；
- 输出投影 $$Z_{\text{out}}=ZW^O$$：也是仿射变换。

这些线性层的 $$g_W=\delta a^\top$$、$$g_b=\delta$$ **仍然用文档公式**。但 Attention 的核心：

$$
Z=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

中间的 **Softmax 不是逐元素激活**，它的雅可比不是对角矩阵。文档在“延伸思考”中明确提过：

> 如果激活函数是 softmax（非逐元素），$$\delta$$ 递推会变；但 softmax + 交叉熵有特例，直接 $$\hat y-e_y$$。

Attention 里的 Softmax 不是输出层配交叉熵，而是对 $$QK^\top/\sqrt{d_k}$$ 做行归一化，所以**不能直接套 $$\hat y-e_y$$**，需要额外的 Softmax 雅可比推导。第 9–10 天文档没有展开这部分，只把它列为延伸思考。

所以精确说法是：

> **Attention 中的线性层（QKV、输出投影）用第 9–10 天的梯度公式；Attention 的 Softmax 部分需要额外处理，不属于“逐元素激活”的标准 $$\delta$$ 递推。**

## 10.3 残差和 LayerNorm 不在第 9–10 天范围内

Transformer block 还有：

- 残差：Pre-LN 中 $$X'=X+\mathrm{MHA}(\mathrm{LN}(X))$$，$$X_{\text{out}}=X'+\mathrm{FFN}(\mathrm{LN}(X'))$$
- LayerNorm：对每个 token 的向量做归一化

- **残差**：反向时梯度有两条路径（恒等路径 + 子层路径），文档只在延伸思考中提了一句“$$y=x+F(x)$$，反向有两条路径”，没有展开。
- **LayerNorm**：需要额外推导其雅可比，第 9–10 天完全没有涉及。Pre-LN 的恒等路径使梯度更容易回传，但 LayerNorm 内部的导数仍需单独推导。

所以完整 Transformer 的反向传播 = 第 9–10 天的 FFN/线性层公式 + Softmax 雅可比 + LayerNorm 反向 + 残差梯度组合。第 9–10 天只覆盖了其中**最基础、最通用的“仿射 + 逐元素激活”骨架**。

---

# 11. 全部问答速查表

| 问题 | 核心答案 |
|---|---|
| **$$Z$$ 通过激活函数得到 $$a$$，$$a$$ 传到下一层？** | 不准确。Pre-LN 中 $$Z$$ 先过输出投影 $$W^O$$，加残差得到 $$X'$$；LayerNorm 在进入子层之前做。注意力打分与加权机制内部的主要非线性是 softmax；整个 Attention 子层还含 LayerNorm，它也非线性，但不是逐元素激活。逐元素激活在 **FFN 内部**。 |
| **FFN 在哪？** | FFN 在**同一个 Transformer block 内、Attention 子层之后**，是“全连接 → 激活 → 全连接”的前馈子层，对每个位置独立作用。 |
| **下一层每个节点有多少权重和偏置？** | 若上层有 $$L$$ 个节点、下层有 $$n$$ 个节点，则**每个下层节点有 $$L$$ 个权重 + 1 个偏置**；整个层有 $$nL$$ 个权重 + $$n$$ 个偏置。 |
| **第 9–10 天反向传播主要讲的是 FFN 全连接部分？** | 基本正确。第 9–10 天讲“仿射 + 逐元素激活”的通用骨架，完全适用于 FFN；Attention 中的 QKV/输出投影等线性层也用同一套公式，但 Softmax、LayerNorm、残差需要额外推导。 |
| **FFN 第一层有 $$L_1$$ 个节点，输入 $$x\in\mathbb{R}^{d_x}$$，怎么计算？** | $$z=W_1x+b_1$$，$$a=\sigma(z)$$；$$W_1\in\mathbb{R}^{L_1\times d_x}$$（列向量约定），$$b_1\in\mathbb{R}^{L_1}$$。每个节点对输入 $$d_x$$ 个分量加权求和，加 1 个偏置，过激活。 |
| **FFN 第一层参数量？** | 权重 $$L_1\times d_x$$ 个，偏置 $$L_1$$ 个，总 $$L_1(d_x+1)$$。 |
| **每个节点有多少权重？** | 每个节点有 $$d_x$$ 个权重（对应输入所有分量）+ 1 个偏置。不是 $$L_1$$ 个，也不是 $$n$$ 个（除非 $$n=d_x$$）。 |
| **$$W_{j,k}$$ 一行代表一个节点的 $$d_x$$ 个权重？** | 列向量约定下，$$W_1\in\mathbb{R}^{L_1\times d_x}$$，**第 $$j$$ 行**是第 $$j$$ 个节点的 $$d_x$$ 个权重；行向量约定下，$$W_1\in\mathbb{R}^{d_x\times L_1}$$，**第 $$j$$ 列**才是。 |
| **序列 $$T$$ 个 token 时参数量会变吗？** | 不变。FFN 对每个位置独立、共享同一组 $$W_1,b_1$$；参数量仍是 $$d_xL_1+L_1$$。 |
| **Transformer 中 $$d_x$$ 和 $$L_1$$ 是什么？** | $$d_x=d_{\text{model}}$$，$$L_1=d_{\text{ff}}$$，通常 $$d_{\text{ff}}=4d_{\text{model}}$$（原始 FFN）或 $$\frac{8}{3}d_{\text{model}}$$（SwiGLU）。 |
| **现代 LLM 的 FFN 带 bias 吗？** | 很多不带。LLaMA 的 attention 投影和 FFN 线性层都无 bias；这是工程实践简化，**不是**由 RMSNorm 替代偏置。RMSNorm 只有缩放 $$\gamma$$，无加性平移 $$\beta$$。 |
| **SwiGLU 的第一层有几个权重矩阵？** | 两个：$$W_{\text{gate}}$$ 和 $$W_{\text{up}}$$，都是 $$d_{\text{model}}\times d_{\text{ff}}$$；然后逐元素乘，再经 $$W_{\text{down}}$$。 |
| **Attention 中的线性层反向传播用第 9–10 天公式吗？** | 是。QKV、输出投影的 $$g_W=\delta a^\top$$、$$g_b=\delta$$ 直接用；但 Attention 的 Softmax 需要额外雅可比，不在标准逐元素激活公式内。 |
| **LayerNorm 和残差的反向传播在第 9–10 天讲了吗？** | 没有。残差只在延伸思考提了一句“两条路径”；LayerNorm 完全未涉及。 |
| **现代 LLM 以 Pre-LN 还是 Post-LN 为主？** | **Pre-LN 为主**。GPT-2、LLaMA 等均采用 Pre-LN。 |
| **Pre-LN 与 Post-LN 的核心区别？** | Post-LN：$$x_{l+1}=\mathrm{LN}(x_l+f_l(x_l))$$，残差穿过 LayerNorm，梯度每层乘 $$\mathrm{LN}'$$。Pre-LN：$$x_{l+1}=x_l+f_l(\mathrm{LN}(x_l))$$，残差是恒等路径，梯度可无衰减回传。 |
| **Pre-LN 的优缺点？** | 优点：训练稳定，支持超深堆叠，对学习率不敏感。缺点：深层梯度范数衰减，深层贡献有限，可能有表示坍缩风险。 |
| **Post-LN 的优缺点？** | 优点：浅层网络性能更优，深层梯度信号更强。缺点：深层梯度消失，训练极不稳定，通常需要 warm-up。 |
| **有没有结合两者优点的方案？** | 有。Mix-LN（浅层 Post-LN、深层 Pre-LN）、KEEL（Highway 式残差）、Peri-LN/ResiDual 等。 |

---

# 12. 最终总结

1. **FFN 的位置**：FFN 不是 Attention 的“下一层网络”，而是同一个 Transformer block 的后半部分。一个 block = Attention 子层 + FFN 子层，各带残差；Pre-LN 中 LayerNorm 位于每个子层输入端。$$Z$$ 不直接过逐元素激活进入 FFN；$$Z$$ 先过输出投影 $$W^O$$、加残差得到 $$X'$$，再经 LayerNorm 进入 FFN。

2. **FFN 的结构**：原始 FFN 是 $$\sigma(xW_1+b_1)W_2+b_2$$，即“全连接 → 逐元素激活 → 全连接”。SwiGLU 变体是 $$(\mathrm{Swish}(xW_{\text{gate}})\odot xW_{\text{up}})W_{\text{down}}$$，常不带 bias；$$d_{\text{ff}}=\frac83d_{\text{model}}$$ 以保持参数量相当。

3. **全连接层参数计数**：输入维度 $$L$$、输出维度 $$n$$ 时，权重 $$n\times L$$ 个，偏置 $$n$$ 个；每个输出节点有 $$L$$ 个权重 + 1 个偏置。FFN 第一层：$$d_x=d_{\text{model}}$$，$$L_1=d_{\text{ff}}$$，权重 $$d_{\text{ff}}\times d_{\text{model}}$$，偏置 $$d_{\text{ff}}$$，总 $$d_{\text{ff}}(d_{\text{model}}+1)$$。注意符号 $$L$$ 在第 3 节表示输入维度，$$L_1$$ 在第 4 节表示 FFN 隐藏层节点数，二者不同。

4. **FFN 第一层逐节点计算**：第 $$j$$ 个节点 $$z_j=\sum_{k=1}^{d_x}W_{j,k}x_k+b_j$$，$$a_j=\sigma(z_j)$$。矩阵写法 $$z=W_1x+b_1$$，$$W_1\in\mathbb{R}^{L_1\times d_x}$$（列向量约定）。

5. **权重矩阵行列约定**：列向量约定 $$W\in\mathbb{R}^{L_1\times d_x}$$，一行对应一个节点的 $$d_x$$ 个权重；行向量约定 $$W\in\mathbb{R}^{d_x\times L_1}$$，一列对应一个节点。参数量都是 $$L_1d_x+L_1$$。本文第 2.1 节用行向量约定，第 3.4/4 节用列向量约定，切换处已加提示。

6. **序列情形**：$$X\in\mathbb{R}^{T\times d_x}$$，FFN 对每个位置独立、共享权重，$$Z=XW_1+\mathbf{1}_Tb_1^\top$$，参数量不随 $$T$$ 增加。

7. **现代 LLM 无 bias**：LLaMA 等线性层常省略 bias，这是工程实践简化，不是由 RMSNorm 替代偏置。RMSNorm 只有逐维缩放 $$\gamma$$，无加性平移 $$\beta$$；LLaMA 用 RMSNorm，不是 LayerNorm。

8. **Pre-LN 与 Post-LN**：现代 LLM 以 **Pre-LN** 为主。Post-LN 残差穿过 LayerNorm，深层梯度易消失，训练不稳定；Pre-LN 残差是恒等路径，训练稳定、支持超深堆叠，但深层梯度范数衰减、深层贡献有限。Mix-LN、KEEL、Peri-LN/ResiDual 等是结合两者优点的改进方向。

9. **与反向传播的衔接**：第 9–10 天的“仿射 + 逐元素激活”骨架完全适用于 FFN 和 Attention 中的线性层；Attention 的 Softmax、LayerNorm、残差需要额外推导。FFN 的反向传播就是 $$g_W=\delta a^\top$$、$$g_b=\delta$$、$$\delta^{[\ell]}=\mathrm{Diag}(\dot\sigma)W^{[\ell+1]\top}\delta^{[\ell+1]}$$ 的直接套用。Pre-LN 的恒等残差路径使梯度更容易回传，但 LayerNorm 内部导数仍需单独推导。

