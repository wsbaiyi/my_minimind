```
Python 3.10.16
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

conda activate mind

channels:
  - defaults
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
show_channel_urls: True
```

# **项目包含**

- MiniMind-LLM结构的全部代码（Dense+MoE模型）。
- 包含Tokenizer分词器详细训练代码。
- 包含Pretrain、SFT、LoRA、RLHF-DPO、模型蒸馏的全过程训练代码。
- 收集、蒸馏、整理并清洗去重所有阶段的高质量数据集，且全部开源。
- 从0实现预训练、指令微调、LoRA、DPO强化学习，白盒模型蒸馏。关键算法几乎不依赖第三方封装的框架，且全部开源。
- 同时兼容`transformers`、`trl`、`peft`等第三方主流框架。
- 训练支持单机单卡、单机多卡(DDP、DeepSpeed)训练，支持wandb可视化训练流程。支持动态启停训练。
- 在第三方测评榜（C-Eval、C-MMLU、OpenBookQA等）进行模型测试。
- 实现Openai-Api协议的极简服务端，便于集成到第三方ChatUI使用（FastGPT、Open-WebUI等）。
- 基于streamlit实现最简聊天WebUI前端。
- 复现(蒸馏/RL)大型推理模型DeepSeek-R1的MiniMind-Reason模型，**数据+模型**全部开源

# 📌 快速开始

<details style="color:rgb(128,128,128)">
<summary>分享本人的软硬件配置（仅供参考）</summary>



* CPU: Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz
* RAM: 128 GB
* GPU: NVIDIA GeForce RTX 3090(24GB) * 8
* Ubuntu==20.04
* CUDA==12.1
* Python==3.10.16
* [requirements.txt](./requirements.txt)

</details>

```bash
git clone https://github.com/jingyaogong/minimind.git
```

## Ⅰ 测试已有模型效果

### 1.环境准备

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.下载模型

```bash
# MiniMind2放在minimind的根目录
git clone https://huggingface.co/jingyaogong/MiniMind2
```

### 3.启动WebUI

```bash
# 可能需要`python>=3.10` 安装 `pip install streamlit`
cd scripts
streamlit run web_demo.py
```

## 开始训练

### 预训练(Pretrain)（学知识）

这个过程是“无监督”的，即人类不需要在过程中做任何“有监督”的校正，而是由模型自己从大量文本中总结规律学习知识点。
模型此阶段目的只有一个：**学会词语接龙**。例如我们输入“秦始皇”四个字，它可以接龙“是中国的第一位皇帝”。

> ```bash
> python train_pretrain.py
> ```
>
> 训练后的模型权重文件默认每隔`100步`保存为 `pretrain_*.pth` 作为预训练的输出权重（其中*为模型的dimension，默认为512）

### 有监督微调(Supervised Fine-Tuning):（学对话方式）

SFT阶段就需要把半成品LLM施加一个自定义的聊天模板进行微调。
例如模型遇到这样的模板【问题->回答，问题->回答】后不再无脑接龙，而是意识到这是一段完整的对话结束。
称这个过程为指令微调，就如同让已经学富五车的「牛顿」先生适应21世纪智能手机的聊天习惯，学习屏幕左侧是对方消息，右侧是本人消息这个规律。
在训练时，MiniMind的指令和回答长度被截断在512，是为了节省显存空间。就像我们学习时，会先从短的文章开始，当学会写作200字作文后，800字文章也可以手到擒来。

> ```bash
> python train_full_sft.py
> ```
>
> 执行监督微调，得到 `full_sft_*.pth` 作为指令微调的输出权重（其中`full`即为全参数微调）
>
> <details style="color:rgb(128,128,128)">
> <summary>注：训练须知</summary>
>
> 
>
> 所有训练过程默认每隔100步保存1次参数到文件`./out/***.pth`（每次会覆盖掉旧权重文件）。
>
> 简单起见，此处只写明两个阶段训练过程。如需其它训练 (LoRA, 蒸馏, 强化学习, 微调推理等) 可参考下文【实验】小节的详细说明。

### 人类反馈强化学习(Reinforcement Learning from Human Feedback, RLHF)**

在前面的训练步骤中，模型已经具备了基本的对话能力，但是这样的能力完全基于单词接龙，缺少正反样例的激励。
模型此时尚未知什么回答是好的，什么是差的。我们希望它能够更符合人的偏好，降低让人类不满意答案的产生概率。
这个过程就像是让模型参加新的培训，从优秀员工的作为例子，消极员工作为反例，学习如何更好地回复。
此处使用的是RLHF系列之-直接偏好优化(Direct Preference Optimization, DPO)。
与PPO(Proximal Policy Optimization)这种需要奖励模型、价值模型的RL算法不同；
DPO通过推导PPO奖励模型的显式解，把在线奖励模型换成离线数据，Ref模型输出可以提前保存。
DPO性能几乎不变，只用跑 actor_model 和 ref_model 两个模型，大大节省显存开销和增加训练稳定性。

> 注：RLHF训练步骤**并非必须**，此步骤难以提升模型“智力”而通常仅用于提升模型的“礼貌”，有利（符合偏好、减少有害内容）也有弊（样本收集昂贵、反馈偏差、多样性损失）。

```bash
torchrun --nproc_per_node 1 train_dpo.py
# or
python train_dpo.py
```

> 训练后的模型权重文件默认每隔`100步`保存为: `rlhf_*.pth`（*
> 为模型具体dimension，每次保存时新文件会覆盖旧文件）

full_sft模型在简洁性和信息准确性方面表现更好；rlhf模型在回答中倾向于提供更多的背景信息，但信息准确性有待改进。
总的来说RLHF后的模型倾向于学习：说更多有礼貌但无用的废话讨好“对话”本身，而对信息准确性则有轻微损失。
天下没有免费的午餐，还需要继续提升RLHF数据集的质量，也要接受模型能力无法避免的损失(程度有轻重)。
DPO和在线PPO的区别在于reject和chosen都是离线准备的，和minimind模型本身的输出必然存在很大的分布差异。
通俗地说DPO算法使模型观看乒乓球世界冠军的打法「录像」进行RL，而不是像PPO一样请reward模型做「教练」纠正自己的打法进行RL

### 知识蒸馏(Knowledge Distillation, KD)

在前面的所有训练步骤中，模型已经完全具备了基本能力，通常可以学成出师了。
而知识蒸馏可以进一步优化模型的性能和效率，所谓知识蒸馏，即学生模型面向教师模型学习。
教师模型通常是经过充分训练的大模型，具有较高的准确性和泛化能力。
学生模型是一个较小的模型，目标是学习教师模型的行为，而不是直接从原始数据中学习。
在SFT学习中，模型的目标是拟合词Token分类硬标签（hard labels），即真实的类别标签（如 0 或 6400）。
在知识蒸馏中，教师模型的softmax概率分布被用作软标签（soft labels）。小模型仅学习软标签，并使用KL-Loss来优化模型的参数。
通俗地说，SFT直接学习老师给的解题答案。而KD过程相当于“打开”老师聪明的大脑，尽可能地模仿老师“大脑”思考问题的神经元状态。
例如，当老师模型计算`1+1=2`这个问题的时候，最后一层神经元a状态为0，神经元b状态为100，神经元c状态为-99...
学生模型通过大量数据，学习教师模型大脑内部的运转规律。这个过程即称之为：知识蒸馏。
知识蒸馏的目的只有一个：让小模型体积更小的同时效果更好。
然而随着LLM诞生和发展，模型蒸馏一词被广泛滥用，从而产生了“白盒/黑盒”知识蒸馏两个派别。
GPT-4这种闭源模型，由于无法获取其内部结构，因此只能面向它所输出的数据学习，这个过程称之为黑盒蒸馏，也是大模型时代最普遍的做法。
黑盒蒸馏与SFT过程完全一致，只不过数据是从大模型的输出收集，因此只需要准备数据并且进一步FT即可。
注意更改被加载的基础模型为`full_sft_*.pth`，即基于微调模型做进一步的蒸馏学习。
`./dataset/sft_1024.jsonl`与`./dataset/sft_2048.jsonl` 均收集自qwen2.5-7/72B-Instruct大模型，可直接用于SFT以获取Qwen的部分行为。

```bash
# 注意需要更改train_full_sft.py数据集路径，以及max_seq_len  
torchrun --nproc_per_node 1 train_full_sft.py
# or
python train_full_sft.py
```

> 训练后的模型权重文件默认每隔`100步`同样保存为: `full_sft_*.pth`（*为模型具体dimension，每次保存时新文件会覆盖旧文件）

此处应当着重介绍MiniMind实现的白盒蒸馏代码`train_distillation.py`，由于MiniMind同系列本身并不存在强大的教师模型，因此白盒蒸馏代码仅作为学习参考。

```bash
torchrun --nproc_per_node 1 train_distillation.py
# or
python train_distillation.py
```

### LoRA (Low-Rank Adaptation)

LoRA是一种高效的参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）方法，旨在通过低秩分解的方式对预训练模型进行微调。
相比于全参数微调（Full Fine-Tuning），LoRA 只需要更新少量的参数。
LoRA 的核心思想是：在模型的权重矩阵中引入低秩分解，仅对低秩部分进行更新，而保持原始预训练权重不变。
代码可见`./model/model_lora.py`和`train_lora.py`，完全从0实现LoRA流程，不依赖第三方库的封装。

```bash
torchrun --nproc_per_node 1 train_lora.py
# or
python train_lora.py
```

> 训练后的模型权重文件默认每隔`100步`保存为: `lora_xxx_*.pth`（*
> 为模型具体dimension，每次保存时新文件会覆盖旧文件）

非常多的人困惑，如何使模型学会自己私有领域的知识？如何准备数据集？如何迁移通用领域模型打造垂域模型？
这里举几个例子，对于通用模型，医学领域知识欠缺，可以尝试在原有模型基础上加入领域知识，以获得更好的性能。
同时，我们通常不希望学会领域知识的同时损失原有基础模型的其它能力，此时LoRA可以很好的改善这个问题。
只需要准备如下格式的对话数据集放置到`./dataset/lora_xxx.jsonl`，启动 `python train_lora.py`
训练即可得到`./out/lora/lora_xxx.pth`新模型权重。

此时【基础模型+LoRA模型】即可获得医疗场景模型增强的能力，相当于为基础模型增加了LoRA外挂，这个过程并不损失基础模型的本身能力。
我们可以通过`eval_model.py`进行模型评估测试。

### 训练推理模型 (Reasoning Model)

DeepSeek-R1实在太火了，几乎重新指明了未来LLM的新范式。
论文指出`>3B`的模型经历多次反复的冷启动和RL奖励训练才能获得肉眼可见的推理能力提升。
最快最稳妥最经济的做法，以及最近爆发的各种各样所谓的推理模型几乎都是直接面向数据进行蒸馏训练，
但由于缺乏技术含量，蒸馏派被RL派瞧不起（hhhh）。
本人迅速已经在Qwen系列1.5B小模型上进行了尝试，很快复现了Zero过程的数学推理能力。
然而一个遗憾的共识是：参数太小的模型直接通过冷启动SFT+GRPO几乎不可能获得任何推理效果。
MiniMind2第一时间只能坚定不移的选择做蒸馏派，日后基于0.1B模型的RL如果同样取得小小进展会更新此部分的训练方案。

做蒸馏需要准备的依然是和SFT阶段同样格式的数据即可，数据集来源已如上文介绍。数据格式例如：

```json lines
{
  "conversations": [
    {
      "role": "user",
      "content": "你好，我是小芳，很高兴认识你。"
    },
    {
      "role": "assistant",
      "content": "<think>\n你好！我是由中国的个人开发者独立开发的智能助手MiniMind-R1-Lite-Preview，很高兴为您提供服务！\n</think>\n<answer>\n你好！我是由中国的个人开发者独立开发的智能助手MiniMind-R1-Lite-Preview，很高兴为您提供服务！\n</answer>"
    }
  ]
}
```

推理模型R1的回复模板是：

```text
<think>\n思考过程\n</think>\n
<answer>\n最终回答\n</answer>
```

这在GRPO中通过设置规则奖励函数约束模型符合思考标签和回复标签（在冷启动靠前的阶段奖励值设置应该提高一些）

另一个问题是蒸馏过程虽然和SFT一样，但实验结果是模型难以每次都符合模板规范的回复，即脱离思考和回复标签约束。
这里的小技巧是增加标记位置token的损失惩罚，详见`train_distill_reason.py`:

```text
# 在 sp_ids 对应的位置增加额外的惩罚
...
loss_mask[sp_ids] = 10 # 惩罚系数
```

另另一个tips是由于推理数据由于只筛选了`<1024`长度的数据，其中多轮对话和英文数据偏少，
因此`r1_mix_1024.jsonl`进行了大约10k条多轮对话+英文数据的混合，防止模型遗忘严重。

脚本默认基于rlhf后的基模型做推理能力的蒸馏微调，下面直接启动训练即可：

```bash
torchrun --nproc_per_node 1 train_distill_reason.py
# or
python train_distill_reason.py
```

> 训练后的模型权重文件默认每隔`100步`保存为: `reason_*.pth`（*为模型具体dimension，每次保存时新文件会覆盖旧文件）



## 测试模型效果

确保需要测试的模型`*.pth`文件位于`./out/`目录下。
也可以直接去[此处](https://www.modelscope.cn/models/gongjy/MiniMind2-PyTorch/files)下载使用我训练的`*.pth`文件。

```bash
python eval_model.py --model_mode 1 # 默认为0：测试pretrain模型效果，设置为1：测试full_sft模型效果
```

<details style="color:rgb(128,128,128)">
<summary>注：测试须知</summary>



如需详情，查看`eval_model.py`脚本代码即可。model_mode分为 0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型，3: Reason模型

## Tokenizer

./scripts/train_tokenizer.py

```
# 一些自言自语
> 尽管minimind_tokenizer长度很小，编解码效率弱于qwen2、glm等中文友好型分词器。
> 但minimind模型选择了自己训练的minimind_tokenizer作为分词器，以保持整体参数轻量，避免编码层和计算层占比失衡，头重脚轻，因为minimind的词表大小只有6400。
> 且minimind在实际测试中没有出现过生僻词汇解码失败的情况，效果良好。
> 由于自定义词表压缩长度到6400，使得LLM总参数量最低只有25.8M。
> 训练数据`tokenizer_train.jsonl`均来自于`匠数大模型数据集`，这部分数据相对次要，如需训练可以自由选择。
```

## 训练数据

### Pretrain数据

尝试把[匠数大模型数据集](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)的中文部分提取出来，清洗出字符`<512`长度的大约1.6GB的语料直接拼接成预训练数据 

文件`pretrain_hq.jsonl` (1.6GB) 数据格式为

```bash
{"text": "如何才能摆脱拖延症？ 治愈拖延症并不容易，但以下建议可能有所帮助..."}
```

### SFT数据

`sft_mini_512.jsonl`(~1.2GB)。

所有sft文件 `sft_X.jsonl` 数据格式均为

```text
{
    "conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "再见"},
        {"role": "assistant", "content": "再见！"}
    ]
}
```

### RLHF数据

文件 `dpo.jsonl` 数据格式为

```text
{
  "chosen": [
    {"content": "Q", "role": "user"}, 
    {"content": "good answer", "role": "assistant"}
  ], 
  "rejected": [
    {"content": "Q", "role": "user"}, 
    {"content": "bad answer", "role": "assistant"}
  ]
}
```

![image-20250317133142181](../minimind/images/image-20250317133142181.png)

### 数据集下载

将下载的数据集文件放到`./dataset/`目录下（✨为推荐的必须项）

```bash
./dataset/
├── dpo.jsonl (909MB)
├── lora_identity.jsonl (22.8KB)
├── lora_medical.jsonl (34MB)
├── pretrain_hq.jsonl (1.6GB, ✨)
├── r1_mix_1024.jsonl (340MB)
├── sft_1024.jsonl (5.6GB)
├── sft_2048.jsonl (9GB)
├── sft_512.jsonl (7.5GB)
├── sft_mini_512.jsonl (1.2GB, ✨)
└── tokenizer_train.jsonl (1GB)
```

<details style="color:rgb(128,128,128)">
<summary>注：各数据集简介</summary>



* `dpo.jsonl` --RLHF阶段数据集
* `lora_identity.jsonl` --自我认知数据集（例如：你是谁？我是minimind...），推荐用于lora训练（亦可用于全参SFT，勿被名字局限）
* `lora_medical.jsonl` --医疗问答数据集，推荐用于lora训练（亦可用于全参SFT，勿被名字局限）
* `pretrain_hq.jsonl`✨ --预训练数据集，整合自jiangshu科技
* `r1_mix_1024.jsonl` --DeepSeek-R1-1.5B蒸馏数据，每条数据字符最大长度为1024（因此训练时设置max_seq_len=1024）
* `sft_1024.jsonl` --整合自Qwen2.5蒸馏数据（是sft_2048的子集），每条数据字符最大长度为1024（因此训练时设置max_seq_len=1024）
* `sft_2048.jsonl` --整合自Qwen2.5蒸馏数据，每条数据字符最大长度为2048（因此训练时设置max_seq_len=2048）
* `sft_512.jsonl` --整合自匠数科技SFT数据，每条数据字符最大长度为512（因此训练时设置max_seq_len=512）
* `sft_mini_512.jsonl`✨ --极简整合自匠数科技SFT数据+Qwen2.5蒸馏数据（用于快速训练Zero模型），每条数据字符最大长度为512（因此训练时设置max_seq_len=512）
* `tokenizer_train.jsonl` --均来自于`匠数大模型数据集`，这部分数据相对次要，（不推荐自己重复训练tokenizer，理由如上）如需自己训练tokenizer可以自由选择数据集。

#  Model Structure

![image-20250312163506339](../minimind/images/image-20250312163506339.png)

## GQA：Grouped Query Attention

为了减少计算量或参数数量，可能会共享键和值的头，即所谓的“Grouped Query Attention”（GQA）或者类似的变体。这种情况下，键和值的头数（n_kv_heads）可能少于查询的头数（n_heads）。



MiniMind-Dense（和[Llama3.1](https://ai.meta.com/blog/meta-llama-3-1/)一样）使用了Transformer的Decoder-Only结构，跟GPT-3的区别在于：

* 采用了GPT-3的预标准化方法，也就是在每个Transformer子层的输入上进行归一化，而不是在输出上。具体来说，使用的是RMSNorm归一化函数。
* 用SwiGLU激活函数替代了ReLU，这样做是为了提高性能。
* 像GPT-Neo一样，去掉了绝对位置嵌入，改用了旋转位置嵌入（RoPE），这样在处理超出训练长度的推理时效果更好。

---

MiniMind-MoE模型，它的结构基于Llama3和[Deepseek-V2/3](https://arxiv.org/pdf/2405.04434)中的MixFFN混合专家模块。

* DeepSeek-V2在前馈网络（FFN）方面，采用了更细粒度的专家分割和共享的专家隔离技术，以提高Experts的效果。

---

## Moe负载均衡

![image-20250312191644634](../minimind/images/image-20250312191644634.png)

### self.scatter_add(dim,index,src)，累加

![img](../minimind/images/v2-e68a940a7be07f7ab08899e357362d5f_1440w.jpg)

![img](../minimind/images/v2-e29b96c849bbfaa1fc55ded7f2913442_1440w.jpg)



![image-20250313100147640](../minimind/images/image-20250313100147640.png)



MiniMind的整体结构一致，只是在RoPE计算、推理函数和FFN层的代码上做了一些小调整。
其结构如下图（重绘版）：

![structure](../minimind/images/LLM-structure.png)
![structure-moe](../minimind/images/LLM-structure-moe.png)

修改模型配置见[./model/LMConfig.py](./model/LMConfig.py)。
参考模型参数版本见下表：

| Model Name        | params | len_vocab | rope_theta | n_layers | d_model | kv_heads | q_heads | share+route |
| ----------------- | ------ | --------- | ---------- | -------- | ------- | -------- | ------- | ----------- |
| MiniMind2-Small   | 26M    | 6400      | 1e6        | 8        | 512     | 2        | 8       | -           |
| MiniMind2-MoE     | 145M   | 6400      | 1e6        | 8        | 640     | 2        | 8       | 1+4         |
| MiniMind2         | 104M   | 6400      | 1e6        | 16       | 768     | 2        | 8       | -           |
| minimind-v1-small | 26M    | 6400      | 1e4        | 8        | 512     | 8        | 16      | -           |
| minimind-v1-moe   | 4×26M  | 6400      | 1e4        | 8        | 512     | 8        | 16      | 1+4         |
| minimind-v1       | 108M   | 6400      | 1e4        | 16       | 768     | 8        | 16      | -           |

# 模型参数设定

MiniMind设定small模型dim=512，n_layers=8来获取的「极小体积<->更好效果」的平衡。

# 项目文件说明

- 注：依据2024年9月20日的更新编写，未完

## images

Readme里的图片目录。

## model

模型文件夹。

### model/minimind_tokenizer

项目自定义的Tokenizer模型文件。

- model/minimind_tokenizer/merges.txt
  merges文件存放的是训练tokenizer阶段所得到的合并词表结果，就是tokenizer.json中，model.merges下的内容。
- model/minimind_tokenizer/tokenizer_config.json
  分词器的配置信息，定义了分词器的版本、额外添加的标记（tokens）、结构/代码和模型参数等信息，比如tokenizer_class指定使用的分词器类名以及model_max_length指定模型能够处理的最大序列长度 和 bos_token指定句首的标记等内容。
- model/minimind_tokenizer/tokenizer.json
  最终的分词器模型文件，包含了分词器的版本号、分词器的截断、填充策略、特殊标记、文本归一化的函数、预分词的策略或方法、分词器模型的类型、词汇表（vocab）和合并规则（merges）等信息。
- model/minimind_tokenizer/vocab.json
  词表文件，就是tokenizer.json中，model.vocab下的内容。

*注：分词器训练代码可见`train_tokenizer.py`*

### model/dataset.py

数据集定义文件，该文件定义了两个继承自Dataset的数据集类，分别是 PretrainDataset 和 SFTDataset，它们分别用于预训练任务和微调任务的数据加载和处理。

### model/LMConfig.py

模型配置文件，定义 LMConfig 类，继承自 PretrainedConfig。如果想修改模型参数，可以在这个文件里改。

主要包括如下内容：

- dim: 模型维度，默认为 512
- n_layers: Transformer 层数，默认为 8
- n_heads: 注意力头数，默认为 16
- n_kv_heads: KV 头数，默认为 8
- vocab_size: 词汇表大小，应于分词器保持一致，默认为 6400
- hidden_dim: 隐藏层维度，默认为 None
- multiple_of: 隐藏层维度的倍数，默认为 64
- norm_eps: 归一化层的 epsilon 值，默认为 1e-5
- max_seq_len: 最大序列长度，默认为 512，如果需要长文本对话支持，可以加大该值的设置
- dropout: Dropout 概率，默认为 0.0
- flash_attn: 是否使用 Flash Attention，默认为 True

*以下是 MOE（Mixture of Experts）的特定配置当 use_moe 为 False 时，以下配置无效*

- use_moe: 是否使用 MOE，默认为 False
- num_experts_per_tok：每个 token 选择的专家数量，默认为 2
- n_routed_experts=4, # 总的专家数量，默认为 4
- n_shared_experts: bool = True, # 是否使用共享专家，默认为 True
- scoring_func='softmax', # 评分函数，默认为 'softmax'
- aux_loss_alpha=0.01, # 辅助损失的 alpha 参数，默认为 0.01
- seq_aux=True, # 是否在序列级别上计算辅助损失，默认为 True
- norm_topk_prob=True, # 是否标准化 top-k 概率，默认为 True

### model/model.py

模型文件，定义了模型结构，包括多个子模块如 FeedForward、RMSNorm、MoEGate、MOEFeedForward、TransformerBlock 等，实现了前向传播计算、损失函数计算和通过逐步生成方式进行文本生成。

**主要内容总结：**

1. RMSNorm:
   - 实现 RMSNorm（Root Mean Square Layer Normalization）归一化，一种归一化方法，用于提高模型的稳定性和训练效果。
2. Attention:
   - 实现自注意力机制，包括 QKV 计算、缩放点积注意力、多头注意力等。
3. FeedForward:
   - 前馈神经网络，用于对输入数据进行非线性变换。
4. MoEGate:
   - 实现专家混合（MoE）的门控机制，用于在多个专家之间进行选择性信息传递。
5. MOEFeedForward:
   - 实现专家混合（MoE）的前馈神经网络。
6. TransformerBlock:
   - 实现 Transformer 的一个块，包含自注意力机制和前馈神经网络。
7. Transformer:
   - 实现整个 Transformer 模型，由多个 Transformer Block 组成的多层结构。包含词嵌入、位置编码、最终的输出层以及训练和推理方法。

**主要功能：**

1. 前向传播计算:
   - 通过 forward 方法，输入 tokens 或 input_ids 和 targets，进行前向传播计算，得到 logits 和 last_loss（如果提供 targets）。
2. 损失函数计算:
   - 使用 F.cross_entropy 计算损失函数，忽略索引为 -1 的标签。
3. 文本生成:
   - 通过 generate 方法，实现逐步生成方式进行文本生成，支持温度、top_k 等参数调整。
4. 评估答案:
   - 通过 eval_answer 方法，对给定的输入索引进行推理，得到最终的 logits。

## 0-eval_pretrain.py

测试预训练模型的接龙效果。
模型加载逻辑:
如果 model_from 为 1，则从本地路径加载自定义的 Transformer 模型。
如果 model_from 为 2，则使用 transformers 库中的预训练模型。

## 1-pretrain.py

### 功能概述

预训练脚本，执行预训练。
可以使用自定义的数据集进行预训练，训练过程中会动态调整学习率，并且支持分布式训练以提高训练效率。

### 使用、配置及功能说明

以下是该脚本的使用、配置和功能说明：

#### 单机多卡使用

- **`torchrun --nproc_per_node 2 1-pretrain.py`**: 运行脚本时需要使用`torchrun`命令，指定使用的GPU数量为2。

#### 参数配置

- **`lm_config = LMConfig()`**: 加载预定义的语言模型配置，具体配置内容在`model/LMConfig.py`文件中。
- **`out_dir = 'out'`**: 设置输出目录，默认为`out`文件夹。
- **`epochs = 20`**: 训练的轮数。
- **`batch_size = 64`**: 每个批次的数据量。
- **`learning_rate = 2e-4`**: 初始学习率。
- **`device = 'cuda:0' if torch.cuda.is_available() else 'cpu'`**: 选择GPU设备，如果没有GPU则使用CPU。
- **`dtype = 'bfloat16'`**: 数据类型，支持自动混合精度训练（AMP）。
- **`save_dir = os.path.join(out_dir)`**: 模型保存目录，默认为`out`文件夹。
- **`tokens_per_iter = batch_size \* max_seq_len`**: 每个迭代步的数据量。

#### 数据加载

- **`data_path_list = ['./dataset/pretrain_data.bin']`**: 训练数据的文件路径列表，默认为`./dataset/pretrain_data.bin`。
- **`num_workers = 8`**: 数据加载的线程数，可以根据系统CPU核心数调整。

#### 训练循环

- **`iter_per_epoch = len(train_loader)`**: 计算每个轮次的迭代步数。
- **`for epoch in range(epochs): train_epoch(epoch)`**: 进行多轮训练，每轮调用`train_epoch`函数进行训练。

#### 模型的保存频率

在训练过程中，模型的权重会每迭代1000步保存一次模型，以便后续检查点或恢复训练。

#### 使用已有权重再训练的说明

如果你已经有一个预训练模型的权重文件（例如`pretrain_model.pth`），并且你想继续在该模型基础上进行微调，可以按照以下步骤操作：

1. 加载已有权重

   ：

   ```
    ckp = f'{save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'
    
    state_dict = torch.load(ckp, map_location=device)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
   ```

   

注意，这部分是脚本中是注释掉的。 如果你想用已有权重继续训练，需要在脚本中找到上述代码，解除注释，才能使用现有的模型进行训练。
\2. **继续训练**：
解除注释后还是用`torchrun --nproc_per_node 2 1-pretrain.py`

## 2-eval.py

测试模型的对话效果。通过加载预训练后的模型，并让模型来回答内置在脚本中的一系列问题，以评估模型的对话效果。

```
        ckp = f'./out/full_sft_{lm_config.dim}{moe_path}.pth'
```



其中`ckp`是检查点的路径，用于加载预训练的模型权重。

## 3-full_sft.py

执行指令微调训练
这段代码实现了指令微调（Instruction Fine-tuning），主要用于训练一个语言模型。以下是代码的主要功能和配置参数说明：

### 配置参数

代码中可以配置的参数有：

- **模型加载**：

  ```
  model_from = 1  # 从权重加载，2使用transformers
  ```

  

  选择是否从本地权重文件加载模型（`model_from = 1`）或使用huggingface的Transformers库中的预训练模型（`model_from = 2`）。

- **配置本地权重文件路径**：

  ```
        ckp = f'./out/pretrain_{lm_config.dim}{moe_path}.pth'
  ```

  

  如果是用预训练权重训练，则可按默认配置，如果是想从已微调的模型继续微调，则需要指定ckp路径，比如`ckp = './out/full_sft_{lm_config.dim}{moe_path}.pth'`。

- **数据集和批处理**：

  ```
  epochs = 19  # 训练轮数
  batch_size = 40  # 每个batch的大小
  learning_rate = 1e-4  # 学习率
  gradient_accumulation_steps = 1  # 梯度累积步数
  ```

  

  控制数据集的使用和训练的批处理大小、学习率和梯度累积方式。

- **设备配置**：

  ```
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 选择GPU或CPU
  dtype = 'bfloat16' or 'float16'  # 数据类型
  ```

  

  指定训练设备和使用的数据类型（半精度浮点数）。

- **分布式训练**：

  ```
  ddp = int(os.environ.get("RANK", -1)) != -1  # 是否启用分布式训练
  ```

  

  如果环境变量中有`RANK`，则启用分布式训练（DDP）。

然后按Readme的描述运行脚本即可。

## 4-lora_sft.py

执行lora微调训练

## 5-dpo_train.py

执行DPO训练

## chat_openai_api.py

实现与OpenAI API类似的接口

## CODE_OF_CONDUCT.md

贡献者公约

## data_process.py

处理数据集，例如pretrain数据提前进行token-encoder、sft数据集抽离qa到csv文件

## eval_ceval.py

评估模型在ceval数据集上的表现

## export_model.py

可以导出模型到transformers格式，推送到huggingface。

## fast_infenence.py

使用 Streamlit 框架构建的交互式聊天应用程序，主要内容和功能的概述如下：
**定义的内容**

1. 模型和Tokenizer 加载：
   - 使用 `AutoModelForCausalLM` 和 `AutoTokenizer` 从 Hugging Face 的模型库中加载预训练的语言模型和对应的 tokenizer。
   - 通过 `st.cache_resource` 缓存模型和 tokenizer，以提高加载效率。
2. 生成配置：
   - 定义了温度（temperature）、top_k 等超参数用于控制生成的质量和多样性。
3. 聊天消息管理：
   - `clear_chat_messages` 和 `init_chat_messages` 函数分别用于清空和初始化聊天记录。
   - 使用 Streamlit 的会话状态（session state）来存储和管理对话历史。
4. 主要功能函数：
   - `main` 函数是应用程序的主入口，负责处理用户输入、模型生成响应以及界面交互。

**实现的功能**

1. 用户界面：
   - Streamlit 页面设置和标题显示。
   - 提供一个文本输入框供用户输入对话内容。
   - 使用聊天消息的 UI 组件展示历史对话记录。
2. 对话处理：
   - 接收用户的输入，并将其作为新消息添加到会话状态中。
   - 将当前对话历史转换为模型可理解的格式（通过 `tokenizer` 和 `apply_chat_template`）。
   - 使用预定义的超参数生成模型的响应。
   - 实时地将生成的回答部分展示在界面上，直到遇到 EOS 标记或达到最大序列长度。
3. 交互功能：
   - 提供一个按钮允许用户清空所有对话记录。
   - 通过 Streamlit 的回调机制实现动态更新界面内容和响应状态。

## LICENSE

项目使用Apache License许可证。

## my_openai_api.py

使用Flask框架构建的API服务器，用于处理与聊天模型相关的请求，包括生成聊天响应和计算文本的嵌入向量。

## README_en.md

项目说明文件（英语）。

## README.md

项目说明文件。

## requirements.txt

python环境依赖文件，列出了运行该项目所需的Python包及其版本。

## train_tokenizer.py

用于分词器训练。