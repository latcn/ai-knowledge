
### 一、奠基性综述与认知架构

这些论文为整个领域提供了理论框架和概念基石。

**1. 《The Rise and Potential of Large Language Model Based Agents》**（复旦NLP团队 & 米哈游，2023）

- **解决问题**：传统AI智能体多为“任务专用、泛化能力弱”，缺乏统一的通用设计范式。
- **方法**：首次提出 **“大脑-感知-行动”** 三大核心模块的通用框架。大脑以LLM为核心，负责记忆、推理与规划；感知模块处理多模态输入；行动模块负责工具使用与具身交互。
- **具体场景**：单智能体任务、多智能体协作、人机协作、智能体社群涌现。被誉为该领域“入门圣经”。

**2. 《A Survey on Large Language Model based Autonomous Agents》**（Wang et al., 2023）

- **解决问题**：LLM驱动的自主智能体研究分散，缺乏系统性的梳理框架。
- **方法**：从**构建、应用和评估**三个维度梳理研究，提出了包含**画像、记忆、规划、行动**四大模块的智能体架构标准模型。
- **具体场景**：通用智能体设计，覆盖各类LLM Agent的构建与评估。

**3. 《Agentic Large Language Models, a survey》**（Plaat et al., 2025）

- **解决问题**：智能体LLM的核心能力尚未被清晰定义和归类。
- **方法**：将智能体LLM的核心能力归纳为三个方面：**推理（reason）、行动（act）和交互（interact）**。
- **具体场景**：信息检索赋能工具使用、反思机制改善多智能体协作等。

**4. 《Agent Harness Engineering: A Survey》**（CMU、Yale、Amazon等）

- **解决问题**：Agent的最终表现不只取决于模型权重，还取决于外部的工程脚手架（Harness），但此前缺乏系统性研究。
- **方法**：将Agent Harness明确定位为模型外部的系统层，涵盖**执行、工具、上下文、生命周期、可观测性、验证和治理**七个维度（ETCLOVG框架）。
- **具体场景**：映射了170多个开源项目，适用于理解Agent工程实践。


### 二、推理与规划（Reasoning & Planning）

**5. 《ReAct: Synergizing Reasoning and Acting in Language Models》**（Yao et al., 普林斯顿 & Google Research，2022，ICLR 2023）

- **解决问题**：纯提示驱动的方案在多步推理、工具调用和动态环境交互中表现不足。
- **方法**：提出**ReAct范式**，将**推理（Reasoning）和行动（Acting）** 交错整合，构建 **“思考（Thought）-行动（Action）-观察（Observation）”** 的闭环机制。
- **具体场景**：问答、事实验证、文本游戏、网页导航等任务。已成为AI Agent领域最具基础性的范式。

**6. 《Chain-of-Thought Prompting Elicits Reasoning in Large Language Models》**（Wei et al., 2022）

- **解决问题**：LLM在复杂多步推理任务（如数学题、常识推理）中表现不佳。
- **方法**：提出**思维链（CoT）** 提示方法，让模型在给出最终答案前，显式输出中间推理步骤。
- **具体场景**：数学应用题、常识推理、符号推理等。

**7. 《Tree of Thoughts: Deliberate Problem Solving with Large Language Models》**（Yao et al., 2023）

- **解决问题**：CoT是线性推理，无法探索多条推理路径或回溯。
- **方法**：提出**思维树（ToT）** 框架，让LLM在推理过程中探索多个思维分支，并进行自我评估和回溯搜索。
- **具体场景**：需要系统性探索的复杂推理任务（如24点游戏、创意写作）。

**8. 《Self-Refine: Iterative Refinement with Self-Feedback》**（Madaan et al., 2023）

- **解决问题**：LLM一次生成的输出往往不够完美，缺乏自我改进能力。
- **方法**：提出**迭代自 refinemen**t 框架，让模型生成输出 → 自我反馈评价 → 根据反馈修正输出，循环往复。
- **具体场景**：代码生成、对话回复、文本摘要等需要多轮优化的任务。


### 三、工具使用（Tool Use）

**9. 《Toolformer: Language Models Can Teach Themselves to Use Tools》**（Schick et al., Meta, 2023）

- **解决问题**：LLM被限制在文本生成内，无法调用外部工具（如计算器、搜索引擎、日历）来增强能力。
- **方法**：让LLM**自我学习**何时调用何种工具，通过采样API调用示例并过滤有效结果来构建训练数据。
- **具体场景**：数学计算（调用计算器）、信息检索（调用搜索引擎）、日程管理（调用日历API）。

**10. 《API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs》**

- **解决问题**：缺乏评估LLM工具调用能力的标准化基准。
- **方法**：构建了一个包含**大量API**的评测基准，测试LLM在多种工具调用场景下的表现。
- **具体场景**：评估LLM在工具选择、参数生成、多工具协作等方面的能力。


### 四、记忆机制（Memory）

**11. 《MemGPT: Towards LLMs as Operating Systems》**（Packer et al., 2023）

- **解决问题**：LLM的上下文窗口有限，无法处理超长对话或文档，导致“记忆”丢失。
- **方法**：借鉴操作系统中的**虚拟内存**管理思想，将记忆分为**主上下文**和**外部上下文**，通过智能换入换出实现无限上下文。
- **具体场景**：长文档分析、长期对话代理、个性化助手等需要持久记忆的场景。

**12. 《Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory》**

- **解决问题**：生产环境中AI Agent需要可扩展、可靠的长时记忆能力。
- **方法**：提供了一套面向生产的**长时记忆**解决方案，支持记忆的存储、检索和更新。
- **具体场景**：企业级AI Agent部署，需要跨会话记忆用户偏好的应用。


### 五、多智能体系统（Multi-Agent Systems）

**13. 《AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation》**（Wu et al., 2023，COLM 2024）

- **解决问题**：单个LLM Agent能力有限，复杂任务需要多个Agent协作完成。
- **方法**：提出**多Agent对话框架**，支持多个LLM Agent通过对话协作，每个Agent可配置不同角色、能力和工具。
- **具体场景**：代码生成与调试（程序员Agent + 评审Agent）、数学问题求解、复杂决策等。

**14. 《MetaGPT: Meta Programming for Multi-Agent Collaborative Framework》**

- **解决问题**：多Agent协作缺乏结构化流程，容易陷入无效对话。
- **方法**：将**软件工程方法论**（如需求分析、架构设计、编码、测试）引入多Agent协作，让不同Agent扮演不同角色（产品经理、架构师、工程师等）。
- **具体场景**：自动化软件开发、项目协作等需要标准化流程的复杂任务。

**15. 《CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society》**

- **解决问题**：多Agent之间的自主对话和协作机制尚不清晰。
- **方法**：提出了**角色扮演**的多Agent通信框架，通过“AI助手”和“AI用户”的角色设定，探索LLM社会的涌现行为。
- **具体场景**：AI社会模拟、角色扮演对话、任务协作等。


### 六、评估与基准（Evaluation & Benchmarking）

**16. 《From LLM Reasoning to Autonomous AI Agents: A Comprehensive Review》**（2026）

- **解决问题**：LLM和AI Agent的评估标准不统一，难以横向比较。
- **方法**：提出了一个包含**约60个LLM和AI Agent基准测试**的分类体系，覆盖2019-2025年间多个领域。
- **具体场景**：为研究人员提供系统的评估框架选择指南。


### 七、安全与对齐（Safety & Alignment）

**17. 《Reflexion: Language Agents with Verbal Reinforcement Learning》**（Shinn et al., NeurIPS 2023）

- **解决问题**：AI Agent在任务失败后缺乏自我反思和改进的机制。
- **方法**：提出**Reflexion**机制，让Agent在失败后通过**语言反馈**进行自我反思，将失败经验转化为“记忆”用于后续改进。
- **具体场景**：复杂推理任务、代码生成调试、决策优化等需要从错误中学习的场景。


### 📚 延伸资源

- **GitHub论文列表**：[LLMAgentPapers](https://github.com/zjunlp/LLMAgentPapers)（必读论文合集）
- **Awesome-Agent-Papers**：追踪LLM和Agent最新论文
- **Awesome-Multi-Agent-Papers**：多智能体论文合集

以上论文构成了AI Agent领域的核心知识图谱，建议从综述论文（尤其是复旦的“大脑-感知-行动”框架）入手建立宏观认知，再按**推理规划 → 工具使用 → 记忆 → 多智能体**的路径深入阅读。