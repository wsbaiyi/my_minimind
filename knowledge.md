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

# **项目包含**

- my_minimind-LLM结构的全部代码（Dense+MoE模型）。
- 包含Tokenizer分词器详细训练代码。
- 包含Pretrain、SFT、LoRA、RLHF-DPO、模型蒸馏的全过程训练代码。
- 收集、蒸馏、整理并清洗去重所有阶段的高质量数据集，且全部开源。
- 从0实现预训练、指令微调、LoRA、DPO强化学习，白盒模型蒸馏。关键算法几乎不依赖第三方封装的框架，且全部开源。
- 同时兼容`transformers`、`trl`、`peft`等第三方主流框架。
- 训练支持单机单卡、单机多卡(DDP、DeepSpeed)训练，支持wandb可视化训练流程。支持动态启停训练。
- 在第三方测评榜（C-Eval、C-MMLU、OpenBookQA等）进行模型测试。
- 实现Openai-Api协议的极简服务端，便于集成到第三方ChatUI使用（FastGPT、Open-WebUI等）。
- 基于streamlit实现最简聊天WebUI前端。
- 复现(蒸馏/RL)大型推理模型DeepSeek-R1的my_minimind-Reason模型，**数据+模型**全部开源

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
git clone https://github.com/jingyaogong/my_minimind.git

'2.2.2+cu121'
```

## Ⅰ 测试已有模型效果

### 1.环境准备

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.下载模型

```bash
# my_minimind2放在my_minimind的根目录
git clone https://huggingface.co/jingyaogong/my_minimind2
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
在训练时，my_minimind的指令和回答长度被截断在512，是为了节省显存空间。就像我们学习时，会先从短的文章开始，当学会写作200字作文后，800字文章也可以手到擒来。

> ```bash
> python train_full_sft.py
> ```
>
> 执行监督微调，得到 `full_sft_*.pth` 作为指令微调的输出权重（其中`full`即为全参数微调）
>
> 
> 所有训练过程默认每隔100步保存1次参数到文件`./out/***.pth`（每次会覆盖掉旧权重文件）。
>
> 简单起见，此处只写明两个阶段训练过程。如需其它训练 (LoRA, 蒸馏, 强化学习, 微调推理等) 可参考下文【实验】小节的详细说明。

### 人类反馈强化学习(Reinforcement Learning from Human Feedback, RLHF)

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
DPO和在线PPO的区别在于reject和chosen都是离线准备的，和my_minimind模型本身的输出必然存在很大的分布差异。
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

此处应当着重介绍my_minimind实现的白盒蒸馏代码`train_distillation.py`，由于my_minimind同系列本身并不存在强大的教师模型，因此白盒蒸馏代码仅作为学习参考。

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
my_minimind2第一时间只能坚定不移的选择做蒸馏派，日后基于0.1B模型的RL如果同样取得小小进展会更新此部分的训练方案。

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
      "content": "<think>\n你好！我是由中国的个人开发者独立开发的智能助手my_minimind-R1-Lite-Preview，很高兴为您提供服务！\n</think>\n<answer>\n你好！我是由中国的个人开发者独立开发的智能助手my_minimind-R1-Lite-Preview，很高兴为您提供服务！\n</answer>"
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
也可以直接去[此处](https://www.modelscope.cn/models/gongjy/my_minimind2-PyTorch/files)下载使用我训练的`*.pth`文件。

```bash
python eval_model.py --model_mode 1 # 默认为0：测试pretrain模型效果，设置为1：测试full_sft模型效果 0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型，3: Reason模型

python eval_vlm.py --model_mode 1 # 0: Pretrain模型，1: SFT模型，2: SFT-多图模型 (beta拓展)

```


如需详情，查看`eval_model.py`脚本代码即可。model_mode分为 0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型，3: Reason模型

## Tokenizer

./scripts/train_tokenizer.py

```
# 一些自言自语
> 尽管my_minimind_tokenizer长度很小，编解码效率弱于qwen2、glm等中文友好型分词器。
> 但my_minimind模型选择了自己训练的my_minimind_tokenizer作为分词器，以保持整体参数轻量，避免编码层和计算层占比失衡，头重脚轻，因为my_minimind的词表大小只有6400。
> 且my_minimind在实际测试中没有出现过生僻词汇解码失败的情况，效果良好。
> 由于自定义词表压缩长度到6400，使得LLM总参数量最低只有25.8M。
> 训练数据`tokenizer_train.jsonl`均来自于`匠数大模型数据集`，这部分数据相对次要，如需训练可以自由选择。
```

## 训练数据

### Pretrain数据

尝试把[匠数大模型数据集](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)的中文部分提取出来，清洗出字符`<512`长度的大约1.6GB的语料直接拼接成预训练数据 

文件`pretrain_hq.jsonl` (1.6GB) 数据格式为

```bash
{"text": "如何才能摆脱拖延症？ 治愈拖延症并不容易，但以下建议可能有所帮助..."}

{"text": "<s>如何才能摆脱拖延症？ 治愈拖延症并不容易，但以下建议可能有所帮助...</s>"}
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

# <s>system
# 你是 MiniMind，是一个有用的人工智能助手。</s>
# <s>user
# 请用一段话描述阿里巴巴集团的企业文化。</s>
# <s>assistant
# 阿里巴巴集团的企业文化以“客户第一、员工第二、股东第三”为核心价值观，强调“让天下没有难做的生意”的使命。</s>
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



## chosen
<s>system
你是 MiniMind，是一个有用的人工智能助手。</s>
<s>user
Q</s>
<s>assistant
good answers>


## rejected
<s>system
你是 MiniMind，是一个有用的人工智能助手。</s>
<s>user
Q</s>
<s>assistant
bad answer</s>
```

![image-20250317133142181](../my_minimind/images/image-20250317133142181.png)

https://www.bilibili.com/list/watchlater?oid=1201309534&bvid=BV1GF4m1L7Nt&spm_id_from=333.1365.top_right_bar_window_view_later.content.click

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
* `lora_identity.jsonl` --自我认知数据集（例如：你是谁？我是my_minimind...），推荐用于lora训练（亦可用于全参SFT，勿被名字局限）
* `lora_medical.jsonl` --医疗问答数据集，推荐用于lora训练（亦可用于全参SFT，勿被名字局限）
* `pretrain_hq.jsonl`✨ --预训练数据集，整合自jiangshu科技
* `r1_mix_1024.jsonl` --DeepSeek-R1-1.5B蒸馏数据，每条数据字符最大长度为1024（因此训练时设置max_seq_len=1024）
* `sft_1024.jsonl` --整合自Qwen2.5蒸馏数据（是sft_2048的子集），每条数据字符最大长度为1024（因此训练时设置max_seq_len=1024）
* `sft_2048.jsonl` --整合自Qwen2.5蒸馏数据，每条数据字符最大长度为2048（因此训练时设置max_seq_len=2048）
* `sft_512.jsonl` --整合自匠数科技SFT数据，每条数据字符最大长度为512（因此训练时设置max_seq_len=512）
* `sft_mini_512.jsonl`✨ --极简整合自匠数科技SFT数据+Qwen2.5蒸馏数据（用于快速训练Zero模型），每条数据字符最大长度为512（因此训练时设置max_seq_len=512）
* `tokenizer_train.jsonl` --均来自于`匠数大模型数据集`，这部分数据相对次要，（不推荐自己重复训练tokenizer，理由如上）如需自己训练tokenizer可以自由选择数据集。

#  Model Structure

## minimind

minimind的整体结构一致，只是在RoPE计算、推理函数和FFN层的代码上做了一些小调整。
其结构如下图（重绘版）：

![structure](../my_minimind/images/LLM-structure.png)
![structure-moe](../my_minimind/images/LLM-structure-moe.png)

## minimind-V

![VLM-structure](pic/VLM-structure.png)

![VLM-structure-moe](pic/VLM-structure-moe.png)

# 知识点

## 输入数据和loss mask

在预训练（Pre-training）和监督微调（Supervised Fine-Tuning, SFT）阶段，输入数据的结构、内容和目标存在显著差异。以下是两者的关键区别：

---

### **1. 预训练（Pre-training）的输入数据**
#### **核心目标**  
学习通用的语言或跨模态表示（如LLM的语言规则、多模态对齐）。  
**数据特点**：  
- **规模大**：海量无标注或弱标注数据（如Common Crawl、维基百科、多模态数据集）。  
- **无监督/自监督**：依赖任务设计自动生成标签（如Next Token Prediction、MLM）。  

#### **输入格式示例**  
- **纯文本模型（如GPT）**：  
  ```text
  连续文本（如："The cat sat on the mat. It was..."）
  ```
  - **处理方式**：  
    - 输入 = 全文（或截断后的片段）。  
    - 标签 = 向右移一位的输入（预测下一个token）。  
    - **Loss Mask**：忽略`[PAD]`和因果掩码（未来token）。  

- **多模态模型（如LLaVA）**：  
  ```text
  [IMG_Emb]图像特征[文本描述："A cat on the mat"]
  ```
  - **处理方式**：  
    - 输入 = 图像嵌入 + 文本描述。  
    - 标签 = 文本部分（图像嵌入不计算损失）。  
    - **Loss Mask**：掩盖图像嵌入和`[PAD]`。  

#### **关键特征**  
✅ 数据量大、覆盖广（通用领域）。  
✅ 输入为连续文本或原始多模态数据（无明确指令）。  
✅ 标签通过任务设计自动生成（如预测下一个token）。  

---

### **2. 监督微调（SFT）的输入数据**
#### **核心目标**  
使模型适应特定任务（如指令遵循、对话生成）。  
**数据特点**：  
- **规模小**：高质量人工标注或合成的任务数据（如Alpaca、ShareGPT）。  
- **有监督**：输入-输出对需明确对应（如指令-回答）。  

#### **输入格式示例**  
- **单轮指令（如Alpaca）**：  
  ```text
  [USER]: 解释牛顿第一定律  
  [ASSISTANT]: 牛顿第一定律指出...
  ```
  - **处理方式**：  
    - 输入 = 用户指令 + 助手回答前缀（`[ASSISTANT]:`）。  
    - 标签 = 助手的完整回答（仅计算这部分损失）。  
    - **Loss Mask**：掩盖用户指令和`[PAD]`。  

- **多模态对话（如LLaVA-SFT）**：  
  ```text
  [IMG_Emb]图像特征[USER]: 图片里有什么？[ASSISTANT]: 一只猫
  ```
  - **处理方式**：  
    - 输入 = 图像嵌入 + 用户问题 + 助手回答前缀。  
    - 标签 = 助手回答部分（`"一只猫"`）。  
    - **Loss Mask**：掩盖图像嵌入、用户问题和`[PAD]`。  

#### **关键特征**  
✅ 数据量小但质量高（任务相关）。  
✅ 输入为结构化对话或指令（明确划分`用户/助手`角色）。  
✅ 标签是人工编写的目标输出（如答案、对话回复）。  

---

### **3. 核心区别总结**
| **维度**      | **预训练（Pre-training）**        | **监督微调（SFT）**           |
| ------------- | --------------------------------- | ----------------------------- |
| **数据规模**  | 海量（TB级）                      | 少量（GB级）                  |
| **数据来源**  | 无标注原始数据                    | 人工标注/合成的任务数据       |
| **输入格式**  | 连续文本或多模态原始数据          | 结构化指令/对话（含角色标记） |
| **标签生成**  | 自监督（如预测下一个token）       | 人工标注的理想输出            |
| **Loss Mask** | 掩盖填充符、未来token（因果掩码） | 掩盖用户输入、填充符          |
| **目标**      | 学习通用表示                      | 适应特定任务行为              |

---

### **4. 典型案例对比**
#### **预训练输入（GPT风格）**  
```text
输入: "The cat sat on the"
标签: " cat sat on the mat"
```
- 模型学习预测下一个词（无明确任务）。  

#### **SFT输入（Alpaca风格）**  
```text
输入: "[USER]: 写一首关于猫的诗\n[ASSISTANT]:"
标签: "小猫轻轻跳，阳光..."
```
- 模型学习根据指令生成诗歌（任务明确）。  

---



## 项目

### **命令行参数argparse**

```
import argparse

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='minimind argument')
    parser.add_argument('--model',type=int,default=0)
    parser.add_argument('--dtype',type=str,default='bfloat16')	
    
    # 运行脚本时加上 --use_wandb，argparse 会将与该参数对应的变量值设置为 True。
	# 默认值为 False: 如果你在命令行中没有包含这个参数，argparse 会将与该参数对应的变量值默认为 False
	parser.add_argument("--use_wandb", action="store_true")
	
	
    args=parser.parse_args()
    print(args.model)
```



## LLaVA 架构详细解释

LLaVA 的核心思想是将一个强大的预训练语言模型与一个预训练的视觉编码器相结合，并通过一个连接模块（投影层）来实现跨模态的对齐。其主要组成部分包括：

1. **视觉编码器 (Vision Encoder)**
2. **投影层 (Projection Layer)**
3. **大型语言模型 (Large Language Model)**

下面我们详细介绍每个部分及其作用。

### 1. 视觉编码器 (Vision Encoder)

- **作用：** 视觉编码器的作用是将输入的图像转换成一系列具有代表性的视觉特征向量。
- **实现：** LLaVA 通常使用预训练好的视觉模型作为其视觉编码器，例如 CLIP (Contrastive Language–Image Pre-training) 的 ViT (Vision Transformer) 部分。CLIP 在大量的图像-文本对上进行训练，使其能够学习到图像和文本之间的关联性，生成的图像特征对语言信息具有很好的兼容性。
- **工作原理：** 给定一张输入图像，视觉编码器（如 ViT）会将图像分割成一系列图像块（patches），然后通过多层 Transformer 结构处理这些图像块，最终输出一系列视觉特征向量（也称为图像 token 或视觉 embedding）。这些特征向量捕捉了图像中的内容、对象、场景等信息。
- **训练状态：** 在 LLaVA 的训练过程中，**视觉编码器的参数通常是冻结的，即不进行更新。这是因为预训练的视觉编码器已经具备了强大的图像理解能力，冻结它可以保留这些能力，并减少需要训练的参数数量。**

### 2. 投影层 (Projection Layer)

- **作用：** 投影层是 LLaVA 架构中的关键连接模块，它的作用是将视觉编码器输出的视觉特征向量转换到与大型语言模型输入 embedding 空间兼容的维度和表示形式。
- **实现：** 投影层通常由一个简单的神经网络构成，例如一个或多个线性层（Linear Layer）或者一个 MLP (多层感知机)。
- **工作原理：** 视觉编码器输出的视觉特征向量的维度和表示形式可能与大型语言模型期望的输入 embedding 不同。投影层接收视觉特征向量作为输入，并通过学习一个映射函数，将其转换为与语言模型 embedding 具有相同维度的向量。这样，视觉信息就可以被“翻译”成语言模型能够理解的格式。
- **训练状态：** 投影层的参数是**可训练**的。在 LLaVA 的训练过程中，投影层会学习如何有效地将视觉特征映射到语言空间，以便语言模型能够利用这些视觉信息进行后续的任务（如生成描述、回答问题等）。

### 3. 大型语言模型 (Large Language Model)

- **作用：** 大型语言模型是 LLaVA 的核心处理单元，它负责理解输入的文本指令和视觉特征，并生成相应的文本输出。

- **实现：** LLaVA 使用强大的预训练大型语言模型，例如 LLaMA、Vicuna 等。这些模型在海量文本数据上进行训练，具备强大的语言理解、生成和推理能力。

- **工作原理：** 大型语言模型接收两种类型的输入：

  - **文本输入：** 用户提出的问题或指令的文本 embedding。

  - 视觉输入： 经过投影层转换后的视觉特征向量，这些向量被视为特殊的“视觉 token”，与文本 token 一起输入到语言模型中。

    语言模型将这些混合的 token 序列作为输入，通过其 Transformer 结构进行处理，理解图像和文本之间的关系，并根据输入生成连贯、相关的文本响应。

- **训练状态：** 在 LLaVA 的训练过程中，大型语言模型的参数可以是**完全冻结**的，或者进行**部分微调**（例如使用 LoRA 等参数高效微调技术）。冻结语言模型可以大幅减少训练计算量和内存需求，但可能限制其适应新任务的能力；部分微调则可以在保持效率的同时，提高模型在多模态任务上的性能。LLaVA 的原始版本通常是冻结语言模型并训练投影层和少量其他参数。

### 整体工作流程

当用户向 LLaVA 输入一张图像和一个文本问题时，整个流程如下：

1. **图像处理：** 输入图像首先由**视觉编码器**处理，生成一系列视觉特征向量。
2. **特征转换：** 这些视觉特征向量通过**投影层**进行转换，使其与语言模型的输入 embedding 空间对齐。
3. **输入序列构建：** 转换后的视觉特征向量（作为视觉 token）与用户输入的文本问题（作为文本 token 的 embedding）拼接起来，形成一个混合的输入序列。
4. **语言模型处理：** 这个混合的输入序列被送入**大型语言模型**。语言模型根据这个序列理解图像内容和用户意图。
5. **文本生成：** 大型语言模型根据其理解生成文本响应，例如图像描述、问题答案等。

### 总结

LLaVA 的架构通过巧妙地结合预训练的视觉编码器和大型语言模型，并通过一个可训练的投影层连接它们，使其能够有效地处理和理解多模态信息。这种架构设计充分利用了现有预训练模型的强大能力，并通过参数高效的方式实现了视觉和语言的融合，为构建强大的多模态 AI 助手奠定了基础。

## BERT 架构详解

BERT (Bidirectional Encoder Representations from Transformers) 是一种基于 Transformer 架构的预训练语言表示模型。它通过在大型无标注文本语料库上进行预训练，学习通用的语言表示，然后在特定的下游任务上进行微调，取得了显著的效果。BERT 的核心在于其双向性和基于 Transformer 的深度编码器结构。

### 1. 整体架构

BERT 的核心是一个多层的双向 Transformer **编码器 (Encoder)** 栈。与原始 Transformer 模型包含编码器和解码器不同，BERT 只使用了 Transformer 的编码器部分。这意味着 BERT 旨在理解输入文本的上下文信息，而不是生成序列。

- **多层结构:** BERT 有两种主要尺寸：
  - **BERT-Base:** 包含 12 个 Transformer 编码器层 (L=12)，隐藏层维度为 768 (H=768)，自注意力头数量为 12 (A=12)，总参数量约 1.1 亿。
  - **BERT-Large:** 包含 24 个 Transformer 编码器层 (L=24)，隐藏层维度为 1024 (H=1024)，自注意力头数量为 16 (A=16)，总参数量约 3.4 亿。
- **Transformer 编码器层:** 每个编码器层由两个子层组成：
  - **多头自注意力 (Multi-Head Self-Attention):** 允许模型在处理每个词时，同时关注输入序列中所有词的信息，捕捉词与词之间的依赖关系。"多头"机制使得模型可以从不同的表示子空间学习信息。
  - **前馈神经网络 (Feed-Forward Neural Network):** 一个简单的、位置独立的全连接前馈网络，通常包含两个线性层和一个激活函数 (如 GELU)。

- **残差连接 (Residual Connections):** 每个子层的输入会通过残差连接直接加到该子层的输出上，有助于缓解梯度消失问题，使得训练更深层的网络成为可能。
- **层归一化 (Layer Normalization):** 在每个子层的输出之后进行层归一化，有助于稳定训练过程。

### 2. 输入表示 (Input Representation)

BERT 的输入不仅仅是词的嵌入，而是由三种嵌入向量相加而成，以捕捉更丰富的信息：

- **词嵌入 (Token Embeddings):** 使用 WordPiece 或类似的子词切分方法将输入文本切分成词元 (tokens)。每个词元会被映射到一个固定维度的向量。BERT 使用一个包含特殊词元的词汇表，例如：
  - `[CLS]`: 加在每个输入序列的开头。这个词元对应的最终隐藏状态通常用于分类任务。
  - `[SEP]`: 用于分隔两个句子，例如在句子对任务 (如问答、自然语言推断) 中。
  - `[MASK]`: 用于 MLM 预训练任务，表示被遮盖的词元。
- **段嵌入 (Segment Embeddings):** 用于区分输入中的不同句子。例如，在句子对任务中，第一个句子的所有词元会加上句子 A 的嵌入，第二个句子的所有词元会加上句子 B 的嵌入。对于单句输入，所有词元都使用句子 A 的嵌入。
- **位置嵌入 (Position Embeddings):** 由于 Transformer 本身不包含序列顺序的信息 (不像 RNN)，BERT 引入了位置嵌入来学习词元在序列中的位置信息。BERT 使用的是**可学习的**位置嵌入，而不是原始 Transformer 中的正弦/余弦位置编码。每个位置 (从 0 到最大序列长度) 都有一个对应的嵌入向量。

最终的输入表示是这三种嵌入向量的**逐元素相加**。

```
Input Embedding = Token Embedding + Segment Embedding + Position Embedding
```

### 3. 预训练任务 (Pre-training Tasks)

BERT 的强大能力很大程度上归功于其创新的预训练策略，它通过两个无监督任务在大量文本数据上进行训练：

- **遮盖语言模型 (Masked Language Model - MLM):**

  - **动机:** 为了实现真正的双向表示学习。传统的从左到右或从右到左的语言模型只能利用单向的上下文。如果简单地让模型预测所有词，模型可以通过"看到自己"来轻易预测目标词，无法学到深层上下文。

  - **方法:** 随机遮盖 (Mask) 输入序列中一定比例 (通常是 15%) 的词元，然后让模型基于**所有**未被遮盖的词元 (即左右上下文) 来预测这些被遮盖的原始词元。

  - **细节:**

    - 80% 的概率，将选中的词元替换为 `[MASK]`。

    - 10% 的概率，将选中的词元替换为**随机**的其他词元。

    - 10% 的概率，保持选中的词元不变。

      这样做是为了减少预训练和微调之间的不匹配 (因为微调时输入中没有 [MASK] 标记)，并迫使模型学习每个词元的分布式表示。

  - **目标:** 最小化模型预测的被遮盖词元与原始词元之间的交叉熵损失。

- **下一句预测 (Next Sentence Prediction - NSP):**

  - **动机:** 许多重要的下游任务 (如问答 QA, 自然语言推断 NLI) 需要理解句子之间的关系，而 MLM 任务本身无法直接捕捉这种关系。
  - **方法:** 给定两个句子 A 和 B，模型需要预测句子 B 是否是句子 A 在原始文本中的**实际下一句** (IsNext) 还是一个**随机**的句子 (NotNext)。
  - **构造样本:**
    - 50% 的样本，B 是 A 的实际下一句。
    - 50% 的样本，B 是从语料库中随机选择的句子。
  - **目标:** 通常使用 `[CLS]` 词元对应的最终隐藏状态，通过一个简单的分类器来预测 `IsNext` 或 `NotNext`，并最小化二分类交叉熵损失。

BERT 的预训练过程同时优化 MLM 和 NSP 两个任务的联合损失。

### 4. 微调 (Fine-tuning)

经过预训练后，BERT 模型学习到了通用的语言表示。为了将其应用于特定的下游任务，需要进行微调：

1. **加载预训练模型:** 使用预训练好的 BERT 模型参数初始化模型。
2. **修改输出层:** 根据具体任务调整模型的输出层。例如：
   - **分类任务 (如情感分析、NLI):** 在 `[CLS]` 词元对应的最终隐藏状态上添加一个线性分类层。
   - **序列标注任务 (如命名实体识别 NER):** 在每个词元对应的最终隐藏状态上添加一个线性分类层。
   - **问答任务 (如 SQuAD):** 预测答案在原文中的起始和结束位置，通常需要两个独立的线性层分别预测开始和结束概率。
3. **端到端训练:** 使用特定任务的标注数据，对整个模型 (包括 BERT 的参数和新添加的输出层) 进行端到端的训练。微调通常使用较小的学习率，并且只需要较少的训练轮数 (Epochs)。

### 总结

BERT 的架构核心是基于 Transformer 的多层双向编码器。其创新的输入表示 (结合词、段、位置嵌入) 和预训练任务 (MLM 和 NSP) 使其能够学习到深层次、上下文感知的语言表示。通过在大规模无标注数据上预训练，然后在特定任务上进行微调，BERT 及其变种在众多自然语言处理任务上取得了突破性的进展。

## RMSNorm

![image-20250312163506339](../my_minimind/images/image-20250312163506339.png)

## GQA：Grouped Query Attention

为了减少计算量或参数数量，可能会共享键和值的头，即所谓的“Grouped Query Attention”（GQA）或者类似的变体。这种情况下，键和值的头数（n_kv_heads）可能少于查询的头数（n_heads）。



my_minimind-Dense（和[Llama3.1](https://ai.meta.com/blog/meta-llama-3-1/)一样）使用了Transformer的Decoder-Only结构，跟GPT-3的区别在于：

* 采用了GPT-3的预标准化方法，也就是在每个Transformer子层的输入上进行归一化，而不是在输出上。具体来说，使用的是RMSNorm归一化函数。
* 用SwiGLU激活函数替代了ReLU，这样做是为了提高性能。
* 像GPT-Neo一样，去掉了绝对位置嵌入，改用了旋转位置嵌入（RoPE），这样在处理超出训练长度的推理时效果更好。

---

my_minimind-MoE模型，它的结构基于Llama3和[Deepseek-V2/3](https://arxiv.org/pdf/2405.04434)中的MixFFN混合专家模块。

* DeepSeek-V2在前馈网络（FFN）方面，采用了更细粒度的专家分割和共享的专家隔离技术，以提高Experts的效果。

---

## PretrainedConfig

在Hugging Face的Transformers库中，**`PretrainedConfig`** 是一个核心类，用于存储和管理预训练模型的配置信息。它的主要作用是为模型定义结构参数和超参数，确保模型在加载、保存和初始化时保持一致性和可复现性。

## Moe负载均衡

```
设 N 是专家的总数，对于一个批次的数据：
f_i表示专家 i 接收到的输入单元的比例（占批次中所有输入单元的比例）。ce
p_i表示门控网络为专家 i 输出的平均路由概率（对批次中所有输入单元取平均）。score_mean



序列级别负载均衡 (Sequence-Level Load Balancing)
旨在将一个序列中不同位置的元素（token）的处理负载分散到不同的专家上。

样本级别负载均衡 (Sample-Level Load Balancing)
旨在将不同的完整输入样本分配给不同的专家进行处理
```

![image-20250312191644634](../my_minimind/images/image-20250312191644634.png)

### self.scatter_add(dim,index,src)

累加

![img](../my_minimind/images/v2-e68a940a7be07f7ab08899e357362d5f_1440w.jpg)

![img](../my_minimind/images/v2-e29b96c849bbfaa1fc55ded7f2913442_1440w.jpg)



![image-20250313100147640](../my_minimind/images/image-20250313100147640.png)

## RoPE（Rotary Position Embedding）

### 简单了解

R(theta)就是旋转矩阵

![image-20250318110543999](../my_minimind/images/image-20250318110543999.png)

![image-20250318110603931](../my_minimind/images/image-20250318110603931.png)

多维

看作是钟表；m和n是token的位置

![image-20250318110639726](../my_minimind/images/image-20250318110639726.png)

### 具体公式

https://www.bilibili.com/video/BV12x42127Pb?spm_id_from=333.788.videopod.sections&vd_source=edb614e9f3e817577f46a2e9deeca011

![image-20250416100708654](pic/image-20250416100708654.png)

![image-20250416100830215](pic/image-20250416100830215.png)

![image-20250416100923520](pic/image-20250416100923520.png)

### 设计公式验证

![image-20250416101224278](pic/image-20250416101224278.png)

![image-20250416101251052](pic/image-20250416101251052.png)

### 二维到多维

![image-20250416101402998](pic/image-20250416101402998.png)

![image-20250416101649364](pic/image-20250416101649364.png)

#### **代码实现**

```
不需要按照相邻两两组合奇偶位置；因为神经元是无序的，不依赖维度顺序

chatglm中为了高效地实现 RoPE，把向量在内存中重新排列成前后两部分，而不是严格按照奇偶位置。这样做是为了充分利用硬件的并行计算能力。
```

![image-20250416102436015](pic/image-20250416102436015.png)

## LoRA（Low-Rank Adaptation of Large Language Models）

### lora

peft (Parameter-Efficient Fine-Tuning)

 从2019年的Adapter, 到2021年的Prefix-tuning，在到现在2023最常用的微调方法LoRA。

![image-20250329103055851](D:/RL-study/pic/image-20250329103055851.png)

反向传播时仅更新Lora权重矩阵

更新后Lora矩阵加到原始权重矩阵上完成更新

![image-20250318103129964](../my_minimind/images/image-20250318103129964.png)

r远远小于M,N，因此降低了参数量

![image-20250318103544107](../my_minimind/images/image-20250318103544107.png)

$$
h=W_0x+\bigtriangleup W_x=W_0x+BAx
$$

```
参数A初始化为random高斯分布，参数B初始化为0；
好处是没有在一开始引入噪声
B确保了微调从预训练模型的状态开始，避免了初始的干扰，同时也为低秩矩阵在训练过程中学习有效的任务特定适应提供了基础。
使用随机高斯分布初始化矩阵 A，则为低秩更新 (BA) 在训练过程中学习有意义的变化提供了必要的多样性和非零值
```

### Qlora

QLoRA 是一种高效的大型语言模型 (LLM) 微调技术，它建立在 LoRA (Low-Rank Adaptation) 的基础之上，并通过引入量化进一步降低了内存消耗。

QLoRA 的核心思想是：

1. **将预训练的基础模型量化到低精度**（通常是 4 位，如 4-bit NormalFloat），以极大减少模型的内存占用。
2. **引入低秩适配器 (LoRA Adapters)**，这些适配器是一些小的、可训练的矩阵，注入到基础模型的特定层（主要是线性层）中。
3. **仅训练这些 LoRA 适配器**，而基础模型的量化权重在微调过程中保持冻结不变。
4. **在更高精度下进行 LoRA 适配器的计算和优化**（通常是 FP16 或 bfloat16），以保持训练的数值稳定性。

**它是如何工作的？**

QLoRA 的工作原理可以分解为以下几个关键部分：

1. **基础模型的 4 位量化**: 这是 QLoRA 节省内存的主要手段。预训练好的 LLM 的权重被量化到 4 位精度。QLoRA 提出了一种名为 **NormalFloat (NF4)** 的 4 位数据类型，它针对权重通常遵循正态分布的特点进行了优化，能在 4 位下更有效地表示权重值。量化后的权重存储在显存中。
2. **LoRA 适配器**: 与原始 LoRA 一样，QLoRA 在基础模型的每个被选中的线性层 W0 中，加入两个小的矩阵 A 和 B。原始的权重更新 ΔW 被约束为 BA，其中 B∈Rd×r，A∈Rr×k，r 是远小于 d 和 k 的秩。在 QLoRA 中，这些矩阵 A 和 B **不被量化**，它们以更高精度（如 FP16 或 bfloat16）进行训练。
3. **前向传播**: 在前向传播时，输入数据与基础模型的量化权重矩阵相乘 (QW0×x)，同时与 LoRA 适配器矩阵相乘 ((BA)×x)。为了执行计算，量化权重 QW0 需要在计算时动态地反量化回更高精度（例如 FP16）。然后，反量化后的基础权重结果与 LoRA 适配器的计算结果相加。整个计算流程大致是：输入 -> 反量化 QW0 到 FP16 -> 计算 FP16 的 W0×x -> 计算 FP16 的 (BA)×x -> 将两个结果相加 -> 输出（可能再被量化）。
4. **反向传播**: 梯度只相对于 LoRA 适配器矩阵 A 和 B 计算。由于基础模型的权重是冻结的，它们不需要计算梯度，这大大减少了计算量和显存需求。
5. **优化器**: 优化器仅更新 LoRA 适配器矩阵 A 和 B 的参数。这些参数数量远少于基础模型的参数数量。

**QLoRA 的创新点:**

除了结合量化和 LoRA，QLoRA 论文还提出了几个关键的技术来确保在低显存下的训练可行性和精度：

- **4-bit NormalFloat (NF4)**: 一种新的信息理论上最优的 4 位量化数据类型，尤其适合正态分布的权重。
- **双量化 (Double Quantization, DQ)**: 对量化常数（如 scale 和 zero_point）本身也进行量化。虽然节省的内存不多，但它进一步减少了量化相关的额外内存开销。
- **分页优化器 (Paged Optimizers)**: 利用 NVIDIA Unified Memory 的特性，将优化器的状态（如 Adam 的动量和方差）分页到 CPU 内存，只在需要时加载到 GPU，避免了优化器状态占用过多 GPU 显存导致 OOM (Out of Memory)。
- **梯度检查点 (Gradient Checkpointing)**: 一种标准的内存优化技术，通过在反向传播时重新计算一部分前向传播的结果来节省显存，QLoRA 更积极地使用了它。

**为什么 QLoRA 能高效微调大模型？**

- **极低的显存占用**: 基础模型量化到 4 位是内存效率的关键。
- **参数高效**: 仅训练 LoRA 适配器，训练参数量极少。
- **计算高效**: 利用了高度优化的 4 位计算核（如 `bitsandbytes` 提供的 CUDA Kernel）。
- **精度保持**: NF4 量化、双量化和在更高精度下训练 LoRA 适配器有助于在大幅降低内存的同时保持可接受的精度。

### 代码

实现 QLoRA 微调通常需要结合使用以下库：

- `transformers`: 用于加载预训练模型和分词器，以及提供 `Trainer` API 进行训练。
- `peft` (Parameter-Efficient Fine-tuning): 提供了 LoRA 等参数高效微调技术的实现，能够轻松地将 LoRA 适配器注入到基础模型中。
- `bitsandbytes`: 提供了高效的 8 位和 4 位量化功能，以及相应的优化器和 CUDA Kernel。

```python
# 首先确保安装必要的库
# pip install transformers peft bitsandbytes accelerate datasets

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# --- 配置 ---
# 选择你要微调的模型，这里以一个较小的模型为例，大模型原理相同
model_name = "facebook/opt-125m" # 替换为你需要微调的模型名称，如 "meta-llama/Llama-2-7b-hf"
dataset_name = "Abirate/english_quotes" # 示例数据集
output_dir = "./qlora-finetuning-demo"
lora_r = 16         # LoRA 的秩
lora_alpha = 32     # LoRA 的缩放因子
lora_dropout = 0.05 # LoRA 层的 dropout
# 通常需要根据模型结构确定 target_modules，这里是 OPT 模型的线性层名称
# 对于 Llama 模型，通常是 ['q_proj', 'k_proj', 'v_proj', 'o_proj'] 等
target_modules = ["q_proj", "v_proj"]

# --- 1. 加载分词器 ---
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 设置 padding token，许多模型没有默认的 padding token，需要手动设置
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- 2. 加载基础模型并进行 4 位量化 ---
# load_in_4bit=True 告诉 transformers 使用 bitsandbytes 加载 4 位量化模型
# bnb_4bit_compute_dtype 指定计算时的数据类型 (通常与 FP16 或 BF16 配合)
# bnb_4bit_quant_type 指定 4 位量化类型，nf4 是 bitsandbytes 提出的 NormalFloat 4bit
# device_map='auto' 自动将模型加载到可用的设备 (GPU)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, # 如果 GPU 支持 BF16，优先使用；否则用 torch.float16
    bnb_4bit_quant_type="nf4",
    device_map="auto"
)

# 3. 准备模型进行 k-bit (这里是 4-bit) 训练
# prepare_model_for_kbit_training 函数会进行一些处理，例如启用梯度检查点、处理 LayerNorm 等
model = prepare_model_for_kbit_training(model)

# --- 4. 配置 LoRA ---
lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none", # 通常对 bias 不使用 LoRA
    task_type="CAUSAL_LM", # 任务类型是因果语言模型
    target_modules=target_modules # 指定将 LoRA 注入到哪些层
)

# --- 5. 将 LoRA 适配器注入到模型中 ---
# get_peft_model 会用 LoRA 适配器包装基础模型，并标记 LoRA 参数为可训练
model = get_peft_model(model, lora_config)

# 打印模型的可训练参数量，你会发现相对于总参数量非常少
model.print_trainable_parameters()

# --- 6. 准备数据集 ---
dataset = load_dataset(dataset_name, split="train")

# 数据预处理函数
def preprocess_function(examples):
    # 将文本和结束符拼接，作为语言模型的输入
    # 这里简单拼接所有 quotes，实际应用中需要更精细的样本处理
    text = [q + tokenizer.eos_token for q in examples["quote"]]
    # 对文本进行分词和编码
    tokenized_examples = tokenizer(
        text,
        max_length=256, # 设置最大序列长度
        padding="max_length",
        truncation=True,
        return_attention_mask=False # 对于 causal LM 通常不需要 attention mask
    )
    # 语言模型的标签是输入的下一个 token，所以直接复制 input_ids
    tokenized_examples["labels"] = tokenized_examples["input_ids"].copy()
    return tokenized_examples

# 应用预处理
dataset = dataset.map(preprocess_function, batched=True, remove_columns=["quote", "author"])

# --- 7. 设置训练参数 ---
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4, # 根据你的显存调整 batch size
    gradient_accumulation_steps=4, # 梯度累积步数，模拟更大的 batch size (4 * 4 = 16)
    learning_rate=2e-4,
    num_train_epochs=1, # 训练 epoch 数
    logging_steps=10, # 每多少步记录一次日志
    save_steps=100, # 每多少步保存一次检查点 (LoRA 适配器)
    fp16=False,      # 如果你的 GPU 不支持 BF16，并且 bnb_4bit_compute_dtype 设置为 FP16，则这里设为 True
    bf16=True,       # 如果你的 GPU 支持 BF16 (如 Ampere 或更新架构)，并且 bnb_4bit_compute_dtype 设置为 BF16，则这里设为 True
    gradient_checkpointing=True, # 启用梯度检查点，进一步节省显存（会增加计算时间）
    # optim="paged_adamw_8bit", # QLoRA 论文中使用了分页优化器，这里可以选择
    report_to="none" # 不向任何地方报告日志
)

# --- 8. 创建 Trainer 并开始训练 ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer, # Trainer 也需要 tokenizer
    # data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # 如果需要特殊的 collator
)

print("开始训练...")
trainer.train()
print("训练完成.")

# --- 9. 保存 LoRA 适配器 ---
# 训练完成后，保存的只有 LoRA 适配器的权重，而不是整个模型
trainer.model.save_pretrained(output_dir)
print(f"LoRA 适配器已保存到 {output_dir}")

# --- 推理示例 (加载基础模型和 LoRA 适配器) ---
print("\n加载 LoRA 适配器进行推理...")
# 首先加载原始的 4 位量化基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    device_map="auto"
)

# 然后加载保存的 LoRA 适配器到基础模型上
from peft import PeftModel
model_for_inference = PeftModel.from_pretrained(base_model, output_dir)

# 将模型切换到评估模式
model_for_inference.eval()

# 进行推理
prompt = "The meaning of life is"
inputs = tokenizer(prompt, return_tensors="pt").to(model_for_inference.device)

with torch.no_grad():
    outputs = model_for_inference.generate(**inputs, max_new_tokens=50, num_return_sequences=1)

print("\n加载适配器后的模型推理结果:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```



## ViT

https://www.bilibili.com/video/BV15P4y137jb?spm_id_from=333.788.videopod.sections&vd_source=edb614e9f3e817577f46a2e9deeca011

```
图片大小：224*224*3   w*h*c

图片切割成许多个patch 16*16；每个Patch看作是一个token；经过线性投影成token id 即embeding层

N:(224*224)/(16*16)=196个patch
D:16*16*3=768 每个patch的大小
输入数据：196*768
投影矩阵E：768*768
投影后：196*768
添加开始标志[class]，则最后的输入：197*768
```

![image-20250319160756977](../my_minimind/images/image-20250319160756977.png)

## clip **Contrastive Language–Image Pretraining**（对比语言-图像预训练）

```
zero shot----···+-

1 对比预训练
图片I_i对应文字T_i


2 从label创建数据集分类器
labels代表正样本，因为对角线都是正样本


3 zero-shot 预测
```

![image-20250319161854468](../my_minimind/images/image-20250319161854468.png)

## 混合精度scaler

### 主要阶段

混合精度训练（通常指结合使用 FP32 和 FP16）的核心思想是在计算过程中智能地切换数据类型，以利用 FP16 的高速和低显存优势，同时保留 FP32 的高精度和数值稳定性，特别是在对数值范围敏感的环节。

这里以前向传播、损失计算、反向传播和权重更新这几个主要阶段来描述精度的变化：

**1. 前向传播 (Forward Pass)**

- **输入数据**: 通常，输入数据最初是以 FP32 精度加载的。
- **模型层内部计算**: 在使用支持混合精度的框架（如 PyTorch 的 `autocast`）时，前向传播过程中大部分计算（如矩阵乘法、卷积等）会在 **FP16** 精度下执行。

- **部分保持 FP32 的计算**: 并不是所有的操作都适合用 FP16 进行。一些对数值范围或精度要求较高的操作（例如 Softmax、Layer Normalization、归约操作 Reduce Sum/Mean 等）可能会被框架自动保持在 **FP32** 精度进行计算，即使它们位于 `autocast` 区域内。这是为了确保数值稳定性，防止溢出或精度损失过大。
- **中间结果**: 在前向传播过程中产生的中间张量会根据执行的操作类型，以 FP16 或 FP32 的精度存储。
- **模型输出**: 模型最终的输出通常会被转换回 **FP32** 精度，以便进行后续的损失计算，因为损失函数对输入精度通常比较敏感。

**总结前向传播：** 输入 FP32 -> 大部分核心计算在 FP16 (有自动类型转换) -> 部分稳定敏感操作在 FP32 -> 中间结果 FP16/FP32 混杂 -> 模型输出 FP32。

**2. 损失计算 (Loss Calculation)**

- 损失函数通常接收 FP32 格式的模型输出和标签。
- 损失值本身通常在 **FP32** 精度下计算和存储。这是为了保持损失值的精度，尤其是当损失值很小时，FP32 能提供更大的动态范围，避免下溢。

**总结损失计算：** 使用 FP32 模型输出和 FP32 标签 -> 计算得到 FP32 的损失值。

**3. 反向传播 (Backward Pass)**

- **梯度缩放 (Gradient Scaling)**: 在反向传播开始之前，混合精度训练会引入梯度缩放。FP32 的损失值会乘以一个较大的**标量缩放因子**（通常用 `GradScaler` 管理）。
- **计算缩放后的梯度**: 反向传播从这个**缩放后的损失**开始计算梯度。根据链式法则，计算得到的每个梯度都会被这个缩放因子放大。
- **梯度计算精度**: 大部分梯度的计算会沿用前向传播时的精度，即在 **FP16** 精度下进行计算，因为梯度也被放大了，不容易发生下溢。
- **梯度累积 (如果使用)**: 如果使用了梯度累积，累积的梯度通常会在 **FP32** 精度下进行累加，以保证累加过程的精度。
- **反缩放 (Unscaling)**: 在将梯度用于更新模型权重之前，累积的、**缩放后的梯度**需要被除以相同的缩放因子，以恢复其原始的数值范围。这一步通常在 **FP32** 精度下进行，得到**未缩放的梯度**。

**总结反向传播：** FP32 损失 -> FP32 损失乘以缩放因子 -> 反向计算得到**缩放后**的梯度（主要在 FP16 计算）-> 累积梯度（通常在 FP32）-> **反缩放**得到**未缩放的梯度**（在 FP32）

**4. 权重更新 (Weight Update)**

- 为了保持模型的长期稳定性和精度，模型的主权重通常会以 **FP32** 精度存储和维护。
- 优化器会使用**未缩放的 FP32 梯度**（可能经过梯度裁剪）来更新这些 **FP32 主权重**。权重的更新计算（例如 SGD 的 `weight = weight - learning_rate * gradient`）也在 FP32 精度下执行。
- 在更新完 FP32 主权重后，通常会将这些更新后的 FP32 权重复制并转换为 **FP16** 格式，用于下一轮前向传播时的快速计算（即所谓的 FP16 工作权重）。

**总结权重更新：** 使用 FP32 梯度更新 FP32 主权重 -> 将更新后的 FP32 权重转换为 FP16 工作权重用于下一轮前向。



**总的来说，混合精度训练中的精度变化是一个动态和混合的过程：**

- 核心计算（前向和反向传播中的大部分）利用 **FP16** 加速。
- 数值敏感的操作和损失计算保持 **FP32** 稳定性。
- 通过**梯度缩放**，放大 FP16 的梯度，防止下溢。
- **FP32 主权重**用于保持模型的精确状态，并在更新时使用 **FP32 梯度**。
- **FP16 工作权重**用于前向传播时的 FP16 计算。

### 图解

![image-20250319192044309](../my_minimind/images/image-20250319192044309.png)

loss计算时，梯度一般很小，超过FP16的范围，因此采用scale缩放

![image-20250319192106215](../my_minimind/images/image-20250319192106215.png)

scaler.update()更新scale比例

![image-20250319191755309](../my_minimind/images/image-20250319191755309.png)

### 原理和代码

```
为什么需要梯度缩放和这个“scalar”？

使用 FP16 进行计算可以带来显著的性能提升，但 FP16 的数值表示范围比 FP32 小。在深度学习训练的反向传播过程中，计算得到的梯度值可能会非常小。如果直接使用 FP16 来存储和处理这些小梯度的值，它们可能会因为数值太小而变成零（即发生 下溢 (underflow)），导致模型无法正确地更新权重，从而影响模型的收敛和最终的精度。

为了解决这个问题，梯度缩放被引入。它的核心思想是在反向传播计算梯度之前，先将损失 (loss) 乘以一个较大的缩放因子（一个标量）。根据链式法则，乘以损失的缩放因子会自动将所有的梯度也按相同的比例放大。这样，原来非常小的梯度值在放大后就落在了 FP16 可以有效表示的范围内，避免了下溢。

在优化器更新模型权重时，再将这些放大后的梯度除以相同的缩放因子，恢复其原始大小，用于更新 FP32 精度的模型权重副本（为了保持权重的精确性，通常会维护一份 FP32 的主权重）。

所以，“混合精度 scalar”实际上是指用于梯度缩放的那个标量缩放因子。
```

现代深度学习框架提供了自动混合精度 (Automatic Mixed Precision, AMP) 的功能，极大地简化了在代码中使用混合精度和梯度缩放的过程。在 PyTorch 中，主要使用 `torch.cuda.amp.autocast` 和 `torch.cuda.amp.GradScaler`

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# 1. 定义模型、损失函数和优化器
model = nn.Linear(10, 1).cuda() # 将模型放到支持FP16计算的设备上，如GPU
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 2. 创建 GradScaler 实例
# GradScaler 负责管理缩放因子，并在反向传播和优化器步骤中应用缩放和反缩放
scaler = GradScaler()

# 假设有一些输入数据和标签
input_data = torch.randn(64, 10).cuda()
labels = torch.randn(64, 1).cuda()

# 训练循环
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # 3. 在需要使用混合精度的前向传播部分使用 autocast 上下文管理器
    # 在 autocast 区域内，PyTorch会自动选择合适的精度（FP16或FP32）进行计算
    with autocast():
        outputs = model(input_data)
        loss = criterion(outputs, labels)

    # 4. 使用 scaler.scale() 来缩放损失
    # 这一步会将损失值乘以当前的缩放因子
    scaler.scale(loss).backward()

    # 5. 使用 scaler.step() 来更新优化器参数
    # 在调用 optimizer.step() 之前，scaler 会先检查梯度是否包含 inf 或 NaN（溢出或无效值）
    # 如果没有，scaler 会将缩放后的梯度反缩放回原始大小，然后调用 optimizer.step() 进行参数更新
    # 如果有 inf 或 NaN，scaler 会跳过当前的 optimizer.step()，并调整缩放因子以尝试避免未来的溢出
    scaler.step(optimizer)

    # 6. 使用 scaler.update() 来更新缩放因子
    # scaler 会根据之前的梯度检查结果来动态调整缩放因子，以便在下一次迭代中使用
    scaler.update()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
```

### minimind的scalar

```python
with ctx:  # 使用自动混合精度（如果启用）
    res = model(X)  # 前向传播
    loss = loss_fct(
        res.logits.view(-1, res.logits.size(-1)),
        Y.view(-1)
    ).view(Y.size())
    loss = (loss * loss_mask).sum() / loss_mask.sum()  # 计算掩码损失
    loss += res.aux_loss  # 添加辅助损失（如果有）
    loss = loss / args.accumulation_steps  # 梯度累积

scaler.scale(loss).backward()  # 反向传播（带梯度缩放）

# 梯度累积完成后更新模型参数
if (step + 1) % args.accumulation_steps == 0:
    scaler.unscale_(optimizer) # 反缩放
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪通常应该应用于未缩放的梯度
    scaler.step(optimizer)  # 更新参数；由于前面对梯度进行了操作，所以不再对梯度反缩放
    scaler.update()  # 更新梯度缩放器
    optimizer.zero_grad(set_to_none=True)  # 清空梯度

    
# 当你需要在调用 scaler.step(optimizer) 之前对梯度进行操作（例如梯度裁剪、手动修改梯度等），并且这些操作要求梯度是未缩放的，你就需要显式地调用 scaler.unscale_(optimizer) 来提前完成反缩放。
# 当你计算完缩放后的梯度后，没有任何需要未缩放梯度的中间操作，直接进行优化器步进，那么 scaler.step(optimizer) 会负责在内部完成反缩放和更新参数的整个流程。
```

这段代码片段展示了在深度学习训练中同时使用了**混合精度训练 (Mixed Precision Training)**、**梯度缩放 (Gradient Scaling)**、**梯度累积 (Gradient Accumulation)** 和**梯度裁剪 (Gradient Clipping)**。

在 `accumulation_steps` 步后才更新参数的原因主要在于**梯度累积**。

这里解释一下各个部分的协同工作：

1. **梯度累积 (`accumulation_steps`)**:

   - 梯度累积是一种在不增加硬件显存占用的情况下，模拟更大 batch size 训练的技术。
   - 它通过在多个小的 forward/backward 步骤中累积梯度，而不是在每一步都更新模型权重。
   - 累积 `accumulation_steps` 步的梯度，相当于使用了 `batch_size * accumulation_steps` 的有效 batch size。

2. **混合精度与梯度缩放 (`scaler.scale(loss).backward()`, `GradScaler`)**:

   - 如前所述，混合精度训练使用 FP16 计算加速，但可能导致小梯度下溢。
   - `GradScaler` 和 `scaler.scale(loss)` 是为了解决 FP16 下溢问题而引入的梯度缩放机制。它将损失放大，从而放大梯度，使其在 FP16 的范围内。
   - `scaler.scale(loss).backward()` 计算的是**缩放后的梯度**。这些缩放后的梯度会在每个小步骤中累积起来。

3. **参数更新 (`scaler.step(optimizer)`)**:

   - 优化器的 `step()` 方法是根据当前累积的梯度来更新模型权重的关键步骤。

   - 在使用 `GradScaler` 时，需要调用 `scaler.step(optimizer)` 来代替标准的 `optimizer.step()`。

   - ```
     scaler.step(optimizer)
     ```

      在内部会执行几个重要操作：

     - **反缩放 (Unscaling)**: 在更新权重之前，`scaler` 会检查累积的梯度是否有 `inf` 或 `NaN` 值。如果没有，它会将累积的**缩放后的梯度**除以当前的缩放因子，得到**原始尺度的梯度**。
     - **应用梯度**: 使用反缩放后的梯度来更新模型的 FP32 主权重。
     - **处理溢出**: 如果检测到 `inf` 或 `NaN`，说明当前的缩放因子可能太高导致溢出，`scaler.step` 会跳过这次权重更新，并通知 `scaler.update()` 调整缩放因子。

4. **更新梯度缩放器 (`scaler.update()`)**:

   - `scaler.update()` 会根据前一步 `scaler.step()` 的结果来调整缩放因子。如果之前没有溢出，缩放因子可能会增大；如果发生了溢出，缩放因子会减小。这是为了动态地找到一个合适的缩放因子。

5. **梯度裁剪 (`torch.nn.utils.clip_grad_norm_`)**:

   - 梯度裁剪用于限制梯度的最大范数，防止梯度爆炸。
   - **重要的一点是，梯度裁剪通常应该应用于未缩放的梯度。**

**为什么在 `accumulation_steps` 后更新参数？**

结合以上几点，原因就很清楚了：

- **`scaler.step(optimizer)` 需要操作的是累积完成的总梯度。** 它负责对累积了多个小批量梯度的结果进行反缩放和应用到权重上。如果在每个小步骤都调用 `scaler.step`，那么每次都只用到了一个小批量的梯度，这违背了梯度累积的初衷，也无法正确地进行缩放和反缩放。
- **`scaler.update()` 的缩放因子调整是基于累积了一定步数（即一个有效 batch size）后的梯度情况。** 在每个小步骤后更新缩放因子是不准确的。
- **梯度裁剪 (`clip_grad_norm_`) 必须在 `scaler.unscale_(optimizer)` 之后进行。** 因为梯度裁剪的阈值是针对未缩放的梯度设定的。在累积 `accumulation_steps` 步后，我们首先通过 `scaler.unscale_(optimizer)` 获取未缩放的累积梯度，然后进行裁剪，最后再由 `scaler.step(optimizer)` 应用这些裁剪后的、已经处理了缩放问题的梯度。

**代码中的流程总结：**

在代码中，`scaler.scale(loss).backward()` 在每个小步都会执行，计算并累加缩放后的梯度。

每当累积满 `args.accumulation_steps` 步后，执行一次完整的优化步骤：

1. `scaler.unscale_(optimizer)`: 将当前累积的缩放后的梯度反缩放回原始大小。**这一步是进行梯度裁剪前必须的。**
2. `torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)`: 对反缩放后的累积梯度进行裁剪。
3. `scaler.step(optimizer)`: 使用裁剪后的梯度更新模型参数，并内部处理 FP16 溢出检测。
4. `scaler.update()`: 根据 `scaler.step` 的结果调整缩放因子。
5. `optimizer.zero_grad(set_to_none=True)`: 清空累积的梯度，为下一轮梯度累积做准备。

因此，将参数更新放在 `accumulation_steps` 后是为了确保优化器操作的是经过完整累积、正确缩放和处理后的梯度，从而实现模拟大 batch size 训练、利用混合精度加速并保持训练稳定。

## 显存占用（混合精度，FP16和FP32）

### 模型参数

### 优化器

为什么不用fp16，因为存在大量的小值操作（梯度计算后乘以一个很小的学习率），可能会丢失精度

![image-20250319183417425](../my_minimind/images/image-20250319183417425.png)

adam优化器

![image-20250319183329645](../my_minimind/images/image-20250319183329645.png)

### 梯度值

![image-20250319184506157](../my_minimind/images/image-20250319184506157.png)

### 激活值

https://zhuanlan.zhihu.com/p/673916177

**激活值：需要在前向传播时保存中间值，便于反向传播计算**

![image-20250319184302628](../my_minimind/images/image-20250319184302628.png)

![image-20250319184416752](../my_minimind/images/image-20250319184416752.png)

![image-20250319185604434](../my_minimind/images/image-20250319185604434.png)



### 总占用

![image-20250319184636424](../my_minimind/images/image-20250319184636424.png)

## Adam和AdamW

https://www.bilibili.com/video/BV1NZ421s75D/?spm_id_from=333.1387.upload.video_card.click&vd_source=edb614e9f3e817577f46a2e9deeca011

![image-20250319193214757](../my_minimind/images/image-20250319193214757.png)

![image-20250319193355233](../my_minimind/images/image-20250319193355233.png)

![image-20250319193931216](../my_minimind/images/image-20250319193931216.png)

-w，weight decay权重衰减，防止参数过大，提高模型泛化能力

![image-20250319194454645](../my_minimind/images/image-20250319194454645.png)

**L2正则和权重衰减不同**

![image-20250319194046711](../my_minimind/images/image-20250319194046711.png)

保存梯度指数平滑值V和保存梯度平方指数平滑值S两个参数，float32存储，因此是原参数的4被

![image-20250319194718141](../my_minimind/images/image-20250319194718141.png)

## 量化

减小模型大小和显存占用

浮点数转为整数型计算

### 图解

#### 量化和反量化：对称量化和非对称量化

![image-20250319201402966](../my_minimind/images/image-20250319201402966.png)

![image-20250319201722534](../my_minimind/images/image-20250319201722534.png)

#### 神经网络量化

![image-20250319202219301](../my_minimind/images/image-20250319202219301.png)

#### 动态量化

量化参数：zero_point，scale

输入fp32，输出fp32，每层动态保存int8权重；**每次输出fp32**

![image-20250319204420378](../my_minimind/images/image-20250319204420378.png)

#### 静态量化

每层输出int8，利用代表性数据得到每层的量化参数，以后每层就固定使用这些参数；**有误差**

![image-20250319204400829](../my_minimind/images/image-20250319204400829.png)

#### 量化感知训练

![image-20250319205245822](../my_minimind/images/image-20250319205245822.png)

#### LLM.int8

![image-20250319210306174](../my_minimind/images/image-20250319210306174.png)

```
# hugging face 模型量化步骤
bnb_config=BitsAndBytesConfig(load_in_8bit=True)
model=AutoModelForCausalLM.from_pretrained(model_id,device_map='auto',quantization_config=bnb_config)
```

#### QLoRA 4bit 量化 NormalFloat4 量化

4bit总共有16类

![image-20250319211516594](../my_minimind/images/image-20250319211516594.png)

![image-20250319211626597](../my_minimind/images/image-20250319211626597.png)

查表，和哪个值最接近得到索引

![image-20250319212318899](../my_minimind/images/image-20250319212318899.png)

分块量化：QLoRA每64个值作为一个块进行NF4 4-bit量化

![image-20250319213148921](../my_minimind/images/image-20250319213148921.png)

**NF4量化后不能直接计算，只能反量化为浮点型进行计算**

![image-20250319211429968](../my_minimind/images/image-20250319211429968.png)

### 原理

量化是指将模型的权重（Weights）和/或激活值（Activations）从高精度浮点格式（如 FP32，32位浮点数；或 FP16，16位半精度浮点数）转换为低精度定点或整数格式（如 INT8，8位整数；INT4，4位整数；甚至更低）。

**减小模型体积**

**加速推理**

**降低能耗**

**量化的核心思想:**

量化的基本思想是将一个范围内的浮点数值，映射到另一个范围内的整数数值。这个映射关系通常通过**缩放因子 (scale)** 和**零点 (zero_point)** 来定义。

最常见的量化方法是**仿射量化 (Affine Quantization)**，其公式为：

​														Q=round(R/scale)+zero_point

其中：

- R: 原始的浮点数 (Real value)。
- Q: 量化后的整数 (Quantized value)。
- scale: 缩放因子，决定了浮点范围如何映射到整数范围。
- zero_point: 零点，一个整数偏移，确保浮点数 0.0 能够精确地映射到一个整数值，这对于保留激活值中的零非常重要（零激活在神经网络中很常见）。
- round(): 取整函数，将结果转换为整数。

![image-20250423232815868](pic/image-20250423232815868.png)

**量化的类型（按何时进行）：**

1. **训练后量化 (Post-Training Quantization, PTQ)**: 在模型训练完成后进行。相对简单快速，不需要重新训练。通常需要少量校准数据来确定 scale 和 zero_point。可能会有一定的精度损失。
2. **量化感知训练 (Quantization-Aware Training, QAT)**: 在训练过程中模拟量化的影响。在模型中插入模拟量化和反量化的节点，让模型在训练时就适应量化带来的数值误差。通常能获得比 PTQ 更好的精度，但需要修改训练流程。

大模型中 PTQ 更常用，因为它不需要巨大的计算资源重新训练，且配合一些先进的 PTQ 技术（如 AWQ, GPTQ）也能达到很好的效果。

**大模型中的反量化 (Dequantization)**

反量化是将量化后的低精度整数值，转换回原始的高精度浮点格式的过程

### 代码

**使用针对大模型的量化库 (如 `bitsandbytes`)**

对于 LLMs，由于其规模和特性（如大量的线性层），常常使用专门的量化库。`bitsandbytes` 是一个非常流行的库，提供了 8 位和 4 位量化功能，并且优化了相应的 CUDA Kernel 以加速计算。

```python
# 需要先安装 bitsandbytes 和 accelerate
# pip install bitsandbytes accelerate transformers

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 模型名称 (例如 Llama-2-7b)
model_name = "meta-llama/Llama-2-7b-hf" # 替换为你想加载的模型

# 以 8 位精度加载模型
# load_in_8bit=True 会使用 bitsandbytes 进行 8 位量化加载
# device_map='auto' 会自动将模型加载到可用的设备上 (如 GPU)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True, # 核心参数：启用 8 位加载
    device_map='auto',
    torch_dtype=torch.float16 # 通常与 FP16 或 BF16 配合使用
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"模型加载完成，使用了 8 位量化: {model_name}")
print("模型参数类型示例:", model.lm_head.weight.dtype) # 可能会显示 BitsAndBytes 8-bit 类型

# 现在可以直接使用 model 进行推理，计算会在 8 位进行 (通过 bitsandbytes 的 kernel)
# 输入和输出处理通常仍然涉及 FP16/FP32 的转换，由 bitsandbytes 在底层处理反量化和量化

# 示例推理
prompt = "请写一个关于人工智能的短故事："
inputs = tokenizer(prompt, return_tensors="pt").to(model.device) # 将输入放到模型所在的设备上

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n模型生成结果:")
print(response)

# 使用 4 位量化类似，只需将 load_in_8bit=True 改为 load_in_4bit=True
# model_4bit = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     load_in_4bit=True, # 启用 4 位加载
#     device_map='auto',
#     torch_dtype=torch.float16
# )
```

这里的量化和反量化细节主要由 `bitsandbytes` 库在底层处理，包括权重的加载、存储和计算时的动态反量化（或直接使用量化 Kernel）。

## 大模型分布式DP

### 原理

**为什么需要大模型分布式？**

主要原因有三个：

1. **内存限制 (Memory Limit):**
   - **模型参数 (Model Parameters):** 大模型的参数量巨大，即使使用较低精度（如 FP16），其存储所需的内存也可能超过单个 GPU 的显存。
   - **梯度 (Gradients):** 训练时需要存储与参数量相同的梯度。
   - **优化器状态 (Optimizer States):** Adam 等优化器需要存储额外的状态（如动量、方差），通常是参数量的 2-4 倍。
   - **中间激活值 (Intermediate Activations):** 前向计算时产生的中间结果，反向传播时需要用到，也非常占用内存。
   - **数据批次 (Batch Data):** 输入数据本身也需要占用显存。 当这些总和超过单个设备的最大内存时，就必须进行分布式处理。
2. **计算限制 (Compute Limit):**
   - 即使模型能勉强放入单个设备的内存，训练或推理所需的计算量也极其庞大，导致耗时过长。分布式可以将计算任务并行化，显著缩短时间。
3. **数据限制 (Data Limit):**
   - 训练大模型通常需要海量数据。将数据分布到不同的机器上，可以提高数据加载和处理的效率。

**DP:data parallel数据并行**

**模型并行 (Model Parallelism, MP):**

- **原理:** 将**模型本身**（层、模块或甚至层内的张量）分割到不同的设备上。数据在这些设备之间流动，按顺序通过模型的不同部分。
- 主要类型:
  - **张量并行 (Tensor Parallelism, TP):** 将层内的**张量**（如权重矩阵）分割到多个设备上。例如，一个大的权重矩阵可以按列或按行分割，矩阵乘法操作也随之分解并涉及设备间的通信。这解决了单个层计算或存储需求过大的问题。
  - **流水线并行 (Pipeline Parallelism, PP):** 将模型的**不同层或连续的几层**分配到不同的设备上，形成一个处理数据的流水线。数据批次被进一步分成更小的“微批次”（micro-batches）。当第一个微批次完成第一阶段的计算并传递给第二阶段时，第一阶段就可以开始处理第二个微批次，从而重叠计算和通信。
- 工作流程 (以 PP 为例):
  1. 将模型层分组，分配到不同的设备（阶段）。
  2. 将一个 mini-batch 数据分成多个 micro-batches。
  3. micro-batches 依次进入流水线的第一个阶段设备，计算完成后传递给下一个阶段设备。
  4. 不同阶段的设备可以同时处理不同的 micro-batch，形成流水线效应。
  5. 反向传播也沿着流水线反向进行。
- **优点:** 可以训练模型本身大于单个设备内存的模型。
- 缺点:
  - **实现复杂:** 需要仔细切分模型和管理数据流。
  - **流水线气泡 (Pipeline Bubble):** 在流水线启动和关闭时，部分设备可能会空闲，效率不高。
  - **计算/通信平衡:** 需要平衡各阶段的计算负载和设备间的通信开销。

**专家并行 (Expert Parallelism, EP):**

- **原理:** 主要用于 Mixture-of-Experts (MoE) 模型。MoE 模型包含多个独立的“专家”网络。不同的数据输入会通过一个“门控网络”（gating network）被路由到一个或多个特定的专家进行处理。专家并行是将这些不同的专家分散到不同的设备上。
- **优点:** 可以显著增加模型的总参数量，但每个数据点只激活其中一部分专家，计算效率相对较高。
- **缺点:** 实现复杂，负载均衡是挑战（确保专家被均匀访问）

### DP:data parallel数据并行

```
原理：
将相同完整的模型副本复制到不同的设备上。
通过一个cpu进程将batch数据分割成多个mini-batch
每个设备接收不同批次的数据mini-batch进行前向传播和反向传播，计算出各自的梯度。
然后，通过通信操作将所有设备的梯度进行平均或求和，再用这个平均梯度更新所有设备上的模型参数，保持模型副本的一致性。
（具体操作是所有梯度集中到gpu0上操作，操作完后将更新的梯度广播给其他的gpu）

使用All-Reduce等通信操作，同步所有设备的梯度（例如，求和或平均）。

不足：
通信量大
每个gpu需要保存完整的模型副本
单进程，多线程，只能利用一个cpu（适用于一机cpu多卡gpu）
GPU0通信量大
```

![image-20250319220151303](../my_minimind/images/image-20250319220151303.png)

### DDP:distributed data parallel 分布式数据并行

#### 集群通信方式ring_allreduce

```
ring_allreduce第一阶段: scatter-reduce（有一个数据如a,b,c满了就结束这一阶段） 
```

![image-20250319221019146](../my_minimind/images/image-20250319221019146.png)



```
ring_allreduce第二阶段: allgather（将满的数据发送给下一个gpu，直至所有gpu的梯度全部处理）
```

![image-20250319221051513](../my_minimind/images/image-20250319221051513.png)

#### 过程

```
ddp多进程

原理：
将相同完整的模型副本复制到不同的设备上。
通过一个cpu进程将batch数据分割成多个mini-batch
每个设备接收不同批次的数据mini-batch进行前向传播和反向传播，计算出各自的梯度。
然后，通过通信操作ring-reduce同步各个gpu的梯度，使得梯度保持一致，保持模型副本的一致性。
（具体操作是所有梯度集中到gpu0上操作，操作完后将更新的梯度广播给其他的gpu）

使用All-Reduce等通信操作，同步所有设备的梯度（例如，求和或平均）。

不足:
每个gpu需要存储完整的优化器，显存可能不足。deepspeed可以解决
```

![image-20250319220654786](../my_minimind/images/image-20250319220654786.png)

```
具体细节：
各个gpu的梯度用桶反向保存，因为输出层的梯度最先计算，这样就能边传输边计算。
每满一个桶就进行ring allreduce
```

![image-20250424101943190](pic/image-20250424101943190.png)

![image-20250319221259577](../my_minimind/images/image-20250319221259577.png)

#### 代码

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# 1. 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

# 2. 定义一个简单的模拟数据集
class DummyDataset(Dataset):
    def __init__(self, num_samples, input_dim, output_dim):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim
        # 生成随机数据和标签
        self.inputs = torch.randn(num_samples, input_dim)
        self.labels = torch.randn(num_samples, output_dim)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# 3. 分布式训练函数
def train(rank, world_size, model, train_loader, optimizer, criterion, epoch):
    # 设置当前进程使用的GPU
    torch.cuda.set_device(rank)
    model.cuda(rank)

    # 使用 DDP 包装模型
    # find_unused_parameters=True might be needed if not all parameters receive gradients
    # (e.g., when using gradient checkpointing or freezing layers)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # 设置 epoch 采样器，确保每个 epoch 的数据顺序不同
    train_loader.sampler.set_epoch(epoch)

    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.cuda(rank), labels.cuda(rank)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 每隔一段时间打印损失
        if i % 10 == 9 and rank == 0: # 只在主进程打印
            print(f"Epoch {epoch+1}, Rank {rank}, Step {i+1}, Loss: {running_loss / 10:.4f}")
            running_loss = 0.0

# 4. 主函数，负责分布式环境初始化和启动训练
def main():
    # 获取当前进程的全局 rank (进程编号) 和总进程数
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"Rank {rank}/{world_size} is initializing...")

    # 初始化分布式环境
    # backend: 使用的通信后端 (nccl for GPUs, gloo for CPUs)
    # init_method: 初始化方式 (env:// 表示从环境变量获取 master_addr 和 master_port)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    print(f"Rank {rank}/{world_size} initialization done.")

    # 设置参数
    input_dim = 10
    output_dim = 2
    num_samples_per_rank = 100 # 每个进程处理的数据量
    total_samples = num_samples_per_rank * world_size
    batch_size = 16
    epochs = 5
    learning_rate = 0.01

    # 创建数据集和数据加载器
    # 注意：DistributedSampler 会确保每个进程只加载总数据的一部分
    dataset = DummyDataset(total_samples, input_dim, output_dim)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2) # num_workers 可以根据需要设置

    # 创建模型、损失函数和优化器
    model = SimpleModel(input_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # 开始训练
    print(f"Rank {rank}/{world_size} starting training...")
    for epoch in range(epochs):
        train(rank, world_size, model, train_loader, optimizer, criterion, epoch)

    print(f"Rank {rank}/{world_size} training finished.")

    # 清理分布式环境
    dist.destroy_process_group()
    print(f"Rank {rank}/{world_size} destroyed process group.")

if __name__ == "__main__":
    # 如何运行这个脚本:
    # 假设你想在两块 GPU 上运行 (world_size=2)
    # 使用 torchrun 命令 (推荐，PyTorch 1.9+):
    # torchrun --nproc_per_node=2 your_script_name.py
    # 或者使用旧的 torch.distributed.launch:
    # python -m torch.distributed.launch --nproc_per_node=2 your_script_name.py

    # 注意: 这个脚本需要在支持分布式训练的环境中运行 (多块GPU或多台机器)
    # 如果在单 GPU 环境运行 torchrun --nproc_per_node=1 your_script_name.py 也可以工作
    # 但要运行在多 GPU 环境，确保你的机器有多块 GPU 并且正确配置了驱动。
    # 实际运行时，torchrun 会设置 RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT 等环境变量
    # 这些环境变量会被 dist.init_process_group 读取。
    main()
```



### DeepSpeed ZeRO-1 (zero redundancy optimizer 零冗余优化器)

反向传播参数：1

梯度收集：1

**广播梯度-->更新参数**

大大减少了显存：仅发送给单一GPU

```
网络参数和梯度 FP16
优化器梯度，一阶，二阶，参数（main weight） FP32

由于占用最大的是优化器，因此每个GPU保存对应网络层的优化器Adam（FP32数据）

过程：
前向传播
后向传播：
首先每个gpu计算所有的梯度fp16，将梯度fp16集中发给对应处理这层的gpu更新梯度；
这样每个gpu就有自己负责的网络层最新的梯度fp16，也有自己负责的网络层的优化器
将梯度fp16缩放至fp32，得到一阶二阶动量，更新优化器得到最新的fp32参数
更新各自优化器对应的网络参数fp16，广播给其他gpu


大大减少了显存占用量，但是没有减少通讯量
```

![image-20250319222321451](../my_minimind/images/image-20250319222321451.png)

![image-20250319222303536](../my_minimind/images/image-20250319222303536.png)

### DeepSpeed ZeRO-2

反向传播参数：1

梯度收集：1

```
每个gpu负责更新各自的网络层的参数，因此只需要保存各自网络层的梯度

过程：
前向传播
后向传播：
首先每个gpu计算所有的梯度fp16，将梯度fp16集中发给对应处理这层的gpu更新梯度；
不处理这层的gpu发送完梯度fp16后立即释放（不占用显存）
这样每个gpu就有自己负责的网络层最新的梯度fp16，也有自己负责的网络层的优化器
将梯度fp16缩放至fp32，得到一阶二阶动量，更新优化器得到最新的fp32参数
更新各自优化器对应的网络参数fp16，广播给其他gpu


大大减少了显存占用量，但是没有减少通讯量
```

![image-20250319224601012](../my_minimind/images/image-20250319224601012.png)

#### 代码

**步骤：**

1. **创建 DeepSpeed 配置 JSON 文件**
2. **创建 Python 训练脚本**
3. **使用 DeepSpeed 命令运行脚本**

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler # DeepSpeed handles sampler for you

# 1. 定义一个简单的模拟模型 (可以换成你的大模型)
class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# 2. 定义一个简单的模拟数据集
class DummyDataset(Dataset):
    def __init__(self, num_samples, input_dim, output_dim):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim
        # 生成随机数据和标签
        self.inputs = torch.randn(num_samples, input_dim)
        self.labels = torch.randn(num_samples, output_dim)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# 3. 主训练函数
def main():
    # 初始化 DeepSpeed 环境
    # 注意: DeepSpeed 会自动处理 PyTorch 分布式环境的初始化
    # 命令行参数 --local_rank 会由 deepspeed 启动命令传入
    deepspeed.init_distributed()

    # 获取当前的 rank (进程编号) 和 world_size (总进程数)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    print(f"Rank {rank}/{world_size} is initializing...")

    # 设置参数
    input_dim = 10
    hidden_dim = 256 # 可以增大以模拟更大的模型
    output_dim = 2
    num_samples_per_rank = 100 # 每个进程处理的数据量
    total_samples = num_samples_per_rank * world_size
    # batch_size 和 accumulation_steps 会从 JSON 文件加载或 auto 推断
    epochs = 5
    learning_rate = 0.001 # 注意: 实际学习率可能会因为优化器配置和 global_batch_size 变化

    # 创建模型
    model = SimpleModel(input_dim, hidden_dim, output_dim)

    # 创建数据集和数据加载器
    # DeepSpeed 的 initialize 函数会返回包装好的 dataloader，
    # 它内部会使用 DistributedSampler
    dataset = DummyDataset(total_samples, input_dim, output_dim)

    # 损失函数
    criterion = nn.MSELoss()

    # === DeepSpeed 初始化 ===
    # 传入模型、优化器、数据加载器、损失函数以及配置文件路径
    # deepspeed.initialize 会返回 engine, optimizer, data_loader, lr_scheduler
    # DeepSpeed 会根据 JSON 配置自动包装模型、优化器和数据加载器
    # 这里我们先创建标准的 optimizer，DeepSpeed 内部会用它的 fused 优化器代替
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # model: PyTorch 模型
    # optimizer: 标准 PyTorch 优化器 (DeepSpeed 会包装它)
    # args: 可以通过命令行或字典传入的额外参数，这里用 config= 传入 JSON 路径
    # lr_scheduler: (可选) 标准 PyTorch 学习率调度器
    # dist_init_required: DeepSpeed >= 0.4.0 默认是 True，会自动初始化分布式环境
    engine, optimizer, train_loader, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=None, # 可以通过 args 传入命令行参数，这里用 config= 指定 JSON
        config="ds_config.json",
        training_data=dataset, # DeepSpeed 可以直接从 dataset 创建 dataloader
        # training_dataloader=train_loader, # 也可以传入已有的 dataloader
        lr_scheduler=None,
        dist_init_required=True # DeepSpeed >= 0.4.0 默认 True
    )

    print(f"Rank {rank}/{world_size} DeepSpeed engine initialized.")

    # === 训练循环 ===
    print(f"Rank {rank}/{world_size} starting training...")
    for epoch in range(epochs):
        # DeepSpeed 的 train_loader 内部使用了 DistributedSampler，
        # 所以需要像 PyTorch DDP 一样设置 epoch
        if hasattr(train_loader.sampler, 'set_epoch'):
             train_loader.sampler.set_epoch(epoch)

        engine.train() # DeepSpeed engine 设置模型为训练模式

        total_loss = 0.0
        step_count = 0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            # 数据会自动移动到对应的 GPU 上
            inputs, labels = inputs.to(engine.device), labels.to(engine.device)

            # 前向传播
            outputs = engine(inputs)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播 (使用 engine.backward)
            # DeepSpeed 会根据 ZeRO 配置处理梯度计算和同步
            engine.backward(loss)

            # 优化器步进 (使用 engine.step)
            # DeepSpeed 会根据 ZeRO 配置处理参数更新和分片
            engine.step()

            total_loss += loss.item() * inputs.size(0) # 累加样本损失
            step_count += inputs.size(0) # 累加样本数

            # 每隔一段时间打印损失 (只在主进程打印)
            if (i + 1) % 10 == 0 and rank == 0:
                 print(f"Epoch {epoch+1}, Rank {rank}, Step {i+1}, Loss: {total_loss / step_count:.4f}")
                 total_loss = 0.0
                 step_count = 0


    print(f"Rank {rank}/{world_size} training finished.")

    # === 保存和加载检查点 (可选) ===
    # DeepSpeed 的检查点会保存模型、优化器、学习率调度器和 ZeRO 状态
    checkpoint_dir = "./deepspeed_checkpoint"
    # 只在主进程保存
    if rank == 0:
        print(f"Saving checkpoint to {checkpoint_dir}")
        engine.save_checkpoint(checkpoint_dir)

    # 在所有进程同步，确保保存完成后再继续 (可选，但推荐)
    torch.distributed.barrier()

    # 加载检查点 (在所有进程上进行)
    # print(f"Loading checkpoint from {checkpoint_dir}")
    # # tag 是检查点的名称，默认为 global step
    # # load_module_strict=False 如果模型结构有微小变化，可以设置为 False
    # engine.load_checkpoint(checkpoint_dir, load_module_strict=True)
    # print(f"Rank {rank}/{world_size} checkpoint loaded.")


if __name__ == "__main__":
    main()
```

**Step 3: 使用 DeepSpeed 命令运行脚本**

在终端中，使用 `deepspeed` 命令来启动训练。假设你的脚本名为 `train_ds.py`，配置文件名为 `ds_config.json`。

例如，如果你想在两块 GPU 上运行：

Bash

```
deepspeed --num_gpus=2 train_ds.py --deepspeed_config ds_config.json
```

**命令解释:**

- `deepspeed`: DeepSpeed 提供的启动命令。
- `--num_gpus=2`: 指定使用两块 GPU。DeepSpeed 会启动两个进程，每个进程绑定一个 GPU。你可以根据你的机器拥有的 GPU 数量修改这个参数。
- `train_ds.py`: 你要运行的 Python 训练脚本。
- `--deepspeed_config ds_config.json`: 指定 DeepSpeed 配置文件的路径。

如果你想在多台机器上运行，启动命令会更复杂，通常涉及 `--num_nodes`、`--node_rank`、`--master_addr` 和 `--master_port` 参数，并且需要在所有机器上执行相应的命令。

### DeepSpeed ZeRO-3

由于每个GPU独占一层parameter，因此前向传播和反向传播时需要对应GPU广播对应参数：2次

梯度收集：1

```
每个gpu负责更新各自的网络层的参数，因此只需要保存各自网络层的梯度；同时只保存各自网络层的参数

过程：

前向传播：
由于gpu只保存一部分网络参数，需要广播；其他gpu收到参数并计算后立即释放网络参数（不占用显存）
后向传播：
同样需要网络参数广播；其他gpu收到参数并计算后立即释放网络参数（不占用显存）
每个gpu计算所有的梯度fp16，将梯度fp16集中发给对应处理这层的gpu更新梯度；
不处理这层的gpu发送完梯度fp16后立即释放（不占用显存）
这样每个gpu就有自己负责的网络层最新的梯度fp16，也有自己负责的网络层的优化器
将梯度fp16缩放至fp32，得到一阶二阶动量，更新优化器得到最新的fp32参数
更新各自优化器对应的网络参数fp16，广播给其他gpu


大大减少了显存占用量，但是通讯量变成1.5倍
```

![image-20250424105044052](pic/image-20250424105044052.png)

![image-20250319224314692](../my_minimind/images/image-20250319224314692.png)

### 显存节省分析

os：zero1

os+g：zero2   共享gradient

os+g+p：zero3   共享gradient，parameter

![image-20250319224347437](../my_minimind/images/image-20250319224347437.png)

### Accelerate使用

使用Accelerate集成deepspeed，采用单机多卡的形式进行分布式训练

```
Accelerate使用步骤
初始化accelerate对象accelerator = Accelerator()
调用prepare方法对model、dataloader、optimizer、lr_schedluer进行预处理
删除掉代码中关于gpu的操作，比如.cuda()、.to(device)等，让accelerate自行判断硬件设备的分配
将loss.backbard()替换为accelerate.backward(loss)
当使用超过1片GPU进行分布式训练时，在主进程中使用gather方法收集其他几个进程的数据，然后在计算准确率等指标

之后，只需要配置下accelerate的config文件，使用accelerate launch --config_file default_config.yaml train.py启动脚本开始训练啦！


Accelerate：在无需大幅修改代码的情况下完成并行化。同时还支持DeepSpeed的多种ZeRO策略，简直不要太爽。

代码效率高，支持无论是单机单卡还是多机多卡适配同一套代码。
允许在单个实例上训练更大的数据集：Accelerate 还可以使 DataLoaders 更高效。这是通过自定义采样器实现的，它可以在训练期间自动将部分批次发送到不同的设备，从而允许每个设备只需要储存数据的一部分，而不是一次将数据复制四份存入内存。
支持DeepSpeed：无需更改代码，只用配置文件即可对DeepSpeed开箱即用
```



## 梯度检查点gradient checkpoint/激活值检查点activation checkpoint  

节省显存

反向传播时神经网络默认保存所有梯度，gradient checkpoint可以选择性保存一些梯度来节省显存，未保存的梯度可以计算得到

![image-20250319231033758](../my_minimind/images/image-20250319231033758.png)

## VLLM（**Very Large Language Model Inference Framework**）

**高效大语言模型推理框架**

解决KV cache浪费显存的问题

### PagedAttention：键值缓存的分页管理**

类似操作系统

![image-20250320110026574](../my_minimind/images/image-20250320110026574.png)

![image-20250320110128888](../my_minimind/images/image-20250320110128888.png)

- **问题背景**：传统注意力机制在处理长序列时，键值（KV）缓存需预分配连续显存，导致显存碎片化，限制并发请求数和吞吐量。
- 解决方案
  - **分页机制**：将 KV 缓存划分为固定大小的“块”（类似操作系统内存分页），按需动态分配物理块，消除显存碎片。
  - **逻辑块映射表**：记录序列中每个 token 对应的物理块地址，支持非连续显存的高效访问。
- **效果**：显存利用率提升 **4-5 倍**，支持更长上下文（如 16K tokens）和更高并发请求。

### sharing kv cache

vLLM的SamplingParameter里有个参数n， n: Number of output sequences to return for the given prompt. 业务场景：我在生成训练数据时经常用，比如Prompt是针对给定文本，提出一个问题。n设置为2。vLLM会给你返回两个output。

例如下面给出翻译的两种输出

![image-20250320110742116](../my_minimind/images/image-20250320110742116.png)

### **连续批处理（Continuous Batching）**

- **动态请求调度**：将多个用户请求的 tokens 打包为统一批次，实时动态调整批次大小，避免传统静态批处理的等待延迟。
- **优势**：GPU 利用率提升 **5-10 倍**，尤其适合流式输出场景。

![image-20250319232758455](../my_minimind/images/image-20250319232758455.png)

## Flash Attention

HBM 是 **High Bandwidth Memory** 的缩写，中文称为**高带宽内存**。它是一种用于高性能计算和图形处理的高性能内存技术，主要用于 GPU（图形处理器）、AI 加速器和数据中心等领域。HBM 通过将内存芯片堆叠在一起，并与处理器通过高密度互连技术直接连接，显著提高了内存带宽和能效。

![image-20250320150459317](../my_minimind/images/image-20250320150459317.png)

![image-20250320150630354](../my_minimind/images/image-20250320150630354.png)

1. **分块（Tiling）**
   将输入序列分为多个小块（例如每块 64-128 个 token），每次仅处理一小块，避免一次性加载整个 QKᵀ 矩阵。
2. **在线 Softmax 修正**
   在分块计算 Softmax 时，动态调整每块的统计量（如最大值和求和项），确保全局结果的数值稳定性。
3. **重计算（Recomputation）**
   反向传播时，通过存储少量元数据（如随机数种子）重新生成中间结果，避免显存占用。

![image-20250320152939108](../my_minimind/images/image-20250320152939108.png)



![image-20250320152958609](../my_minimind/images/image-20250320152958609.png)

![image-20250320163729777](../my_minimind/images/image-20250320163729777.png)

 ![image-20250320163644708](../my_minimind/images/image-20250320163644708.png)

![image-20250320153020336](../my_minimind/images/image-20250320153020336.png)

```
算法流程：
Qi  Br,d
Ki,Vi  Bc,d

m_ij：存储第i行第j列的小块每一行的最大值
P_ij：存储第i行第j列的小块每一行的exp
l_ij：存储第i行第j列的小块每一行的exp总和
O_i：存储第i行的Output
m_i_new：存储第i行0到j列每一行的最大值
l_i_new：存储第i行0到j列每一行的总和（-m_i_new因为指数函数结果很大，为了缩小数据）

第十二行代码：第二个括号的第一部分计算的是以前的O的总和，第二部分计算的的当前新算出来的O，乘以外面的逆矩阵相当于除以sum
```

# 项目文件说明

## train_dpo.py

```
得到model和ref_model
两个操作一致，都是从sft_model加载得到
但是ref_model需要require_grad=False并且eval()

# 选出model 输出对应y的概率
模型的输出 是logits shape: (batch_size, seq_len, vocab_size)
y是labels shape: (batch_size, seq_len)
取出对应y的位置的logits的值probs shape: (batch_size, seq_len)；就是策略选择的概率

# 选出ref_model对应y的概率
# 选出actor_model对应y的概率

# 带入dpo loss公式计算loss
```

## train_distill_reason.py

```
# sp_ids是一个与Y.view(-1)形状相同的bool形状的张量，表示Y中每个元素是否在后续的列表中
sp_ids = torch.isin(Y.view(-1),torch.tensor(start_of_think_ids + end_of_think_ids+ start_of_answer_ids + end_of_answer_ids).to(args.device)) 

# <think></think><answer></answer>这些token id的loss设置为10                                           
loss_mask[sp_ids] = 10      
```

## 视觉模型

```
为了得到MiniMind-V，我们只需要完成这2件事即可：

借助擅长翻译图片的 "外语词典" ，把图片从 "外国语言" 翻译为模型便于理解的 "LLM语言"
训练微调LLM，使其和 "外语词典" 度过磨合期，从而更好的理解图片
"外语词典" 称之为Visual Encoder模型。 和LlaVA、Qwen-VL等视觉语言模型类似，MiniMind-V同样选用开源Clip系列模型作为Visual Encoder。 
具体使用clip-vit-base-patch16， 一种基于 ViT-B/16 架构的经典Visual Encoder用于描述图像文本信息。 
输入的图像尺寸为224x224，因为划分的Patch是16×16，所以会产生14*14=196个token作为encoder编码层的输入， 最终产生1×768维的嵌入向量用于和文本对计算误差。 
我们并不需要最终嵌入表示，因此只取encoder层的输出，也就是VIT核心主干的输出特征即可。 
它拿到前一层维度196×768大小的特征，我们把它作为196个visual token输入MiniMind-V。 
与LLM的结合在获取图像encoder特征后，一方面需要把768维度的visual token对齐到LLM的文本token， 
另一方面，要将图像特征映射到与文本embedding相同的空间，即文本token和原生的视觉token需要磨合并不能直接地一视同仁， 可以称之为跨模态的特征对齐。
LlaVA-1使用简单的无偏线性变换完成了这一操作，效果很不错，MiniMind-V同样如此。





下面，我们简单讨论MiniMind-V的外部输入输出的变化。

VLM的输入依然是一段文本，其中包含特殊的占位符。 在计算文本嵌入后，可以将图像编码器生成的向量投影到该占位符对应的嵌入部分，替换掉原先的占位符embedding。 例如：

<image>\n这个图像中有什么内容？
在minimind-v中，使用196个字符组成的 @@@...@@@ 占位符代替图像，之所以是196个字符，前面有所提及： 任何图像都被clip模型encoder为196×768维的token， 因此minimind-v的prompt为：

@@@......@@@\n这个图片描述的是什么内容？
计算完embedding和projection，并对图像部分token替换后整个计算过程到输出则和LLM部分没有任何区别。
```

![minimind-v-input](pic/minimind-v-input.png)

![llava-structure](pic/llava-structure.png)

## VLMDataset

```
输入：图像内容\n<image>
chat template：替换<image>为标记的特殊字符串（196个'@'对应ViT的196个图像块,这些位置是模型的输出）
图像内容\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@(196个'@)

```

## model_vlm.py

```
VisionProj类将视觉编码器输出维度投影到语言模型隐层维度


查找替换位置 (find_indices)
使用 unfold 创建滑动窗口，检测文本中连续 image_ids 的位置（如 @@@@...）。

投影图像嵌入
通过 self.vision_proj 将视觉特征（如 CLIP 的 768 维）投影到语言模型维度（如 512 维）。

嵌入替换
对每个批次的嵌入，根据 image_indices 找到的位置，用图像嵌入替换文本中的特殊标记。
```

## train_pretrain_vlm.py

```
# 冻结除 vision_proj 外的所有参数
视觉编码器的参数通常是冻结的，即不进行更新。这是因为预训练的视觉编码器已经具备了强大的图像理解能力，冻结它可以保留这些能力，并减少需要训练的参数数量。

# 解冻最后两层transformers block的参数
LLaVA 的原始版本通常是冻结语言模型并训练投影层和少量其他参数。
```

# LLM-book

## 训练过程

大语言模型的训练过程可以分为**大规模预训练**和**指令微调与人类对齐**两个阶段

经过大规模数据**预训练**后的语言模型已经具备较强的模型能力，能够编码丰

富的世界知识，但是由于预训练任务形式所限，这些模型更擅长于文本补全，并不适合直接解决具体的任务。

**“指令微调”**（也叫做有监督微调，Supervised Fine-tuning, SFT），通过使用任务输入与输出的配对数据进行模型训练，可以使得语言模型较好地掌握通过问答形式进行任务求解的能力。

基于人类反馈的强化学习对齐方法 **RLHF**（Reinforcement Learning from Human Feedback），在指令微调后使用强化学习加强模型的对齐能力。

训练一个符合人类价值观的奖励模型（Reward Model）。

## 预训练

### **可扩展的训练技术**

#### **3D** **并行训练**

即数据并行（DataParallelism）、流水线并行（Pipeline Parallelism）和张量并行（Tensor Parallelism）。

![image-20250329110122884](pic/image-20250329110122884.png)

```
• 数据并行. 数据并行是一种提高训练吞吐量的方法，它将模型参数和优化器
状态复制到多个 GPU 上，然后将训练数据平均分配到这些 GPU 上。这样，每个GPU 只需要处理分配给它的数据，然后执行前向传播和反向传播以获取梯度。当所有 GPU 都执行完毕后，该策略会将不同 GPU 的梯度进行平均，以得到整体的梯度来统一更新所有 GPU 上的模型参数。
• 流水线并行. 流水线并行旨在将大语言模型不同层的参数分配到不同的GPU 上。在实践中，可以将 Transformer 连续的层加载到同一 GPU 上，以减少
GPU 之间传输隐藏状态或梯度的成本。例如，在图 6.3 (d) 中Transformer 的第 1-2层部署在 1 号 GPU，将 3-4 层部署在 2 号 GPU。然而，朴素的流水线调度并不能达到真正的并行效果。以图 6.3 (d) 为例，1 号 GPU 在前向传播后需要等待 2 号 GPU反向传播的结果才能进行梯度传播，因此整个流程是“1 号前向-2 号前向-2 号反向-1 号反向”的串行操作，大大降低了 GPU 的利用率。为了解决这一问题，流水线并行通常需要配合梯度累积（Gradient Accumulation）技术进行优化。该技术的主要思想是，计算一个批次的梯度后不立刻更新模型参数，而是累积几个批次后再更新，这样便可以在不增加显存消耗的情况下模拟更大的批次。在流水线并行中使用了梯度累积后，1 号卡前向传播完第一个批次后，便可以不用等待，继续传播第二个和后续的批次，从而提高了流水线的效率。
• 张量并行. 张量并行与流水线并行是两种将大模型参数加载到多个 GPU 上
的训练技术。流水线并行侧重于将模型的不同层分配到不同的 GPU 上。相较之下，张量并行的分配粒度更细，它进一步分解了模型的参数张量（即参数矩阵），以便更高效地利用多个 GPU 的并行计算能力。具体地，对于大语言模型中的某个矩阵乘法操作 𝑾𝑯，参数矩阵 𝑾 可以按列分成两个子矩阵 𝑾1 和 𝑾2，进而原式可以表示为 [𝑾1𝑯, 𝑾2𝑯]。然后，可以将参数矩阵 𝑾1 和 𝑾2 放置在两张不同的 GPU上，然后并行地执行两个矩阵乘法操作，最后通过跨 GPU 通信将两个 GPU 的输出组合成最终结果。常见的张量并行策略是分解模型注意力层的 𝑾𝑄，𝑾𝐾，𝑾𝑉，𝑾𝑂 矩阵参数和前馈网络层的 𝑾𝑈，𝑾𝐷 矩阵参数。
```

#### **零冗余优化器**

零冗余优化器（Zero Redundancy Optimizer, ZeRO）**主要用于解决数据并行中的模型冗余问题**，即每张 GPU 均需要复制一份模型参数。在图 6.3 (a) 中可以看到，数据并行时每个 GPU 都需要存储大语言模型的相同副本，包括模型参数和优化器参数等。对于每个 GPU，在模型传播到某一层时，其他层的模型和优化器参数并不参数计算，这导致了严重的显存冗余现象，同时也限制了每个 GPU 可以支持的前向传播数据量，降低了训练效率。为了解决这个问题，**ZeRO 技术仅在每个 GPU 上保留部分模型参数和优化器参数，当需要时再从其它 GPU 中读取**。如图 6.3 (b) 所示，模型被均分在两张 GPU 上，当需要使用第一层计算时，两张卡分别从对方获取相应的模型参数进行计算，使用完之后便可以释放相应显存，从而降低了显存冗余度。

#### **激活重计算**

激活重计算（Activation Recomputation），也称为梯度检查点（Gradient Checkpointing），是一种用于优化反向传播时显存占用的技术。具体来说，给定一个待优化函数 *𝒀* = *𝑿𝑾*，在反向传播时需要 *𝑿* 的值才能计算 *𝑾* 的导数，所以在前向传播时需要保留这些 *𝑿*（通常被称为激活值）。然而，保存每一层所有的激活值需要占用大量的显存资源（具体的显存占用见第 6.4.4 节）。因此，激活重计算技术在前向传播期间仅保留部分的激活值，然后在反向传播时重新计算这些激活值，以达到节约显存的目的，但是同时也会引入额外的计算开销。在大语言模型的训练过程中，激活重计算的常见方法是将 Transformer 的每一层的输入保存下来，然后在反向传播时计算对应层内的激活值。

#### **混合精度训练**

通过同时使用半精度浮点数（2 个字节）和单精度浮点数（4 个字节）进行运算，以实现显存开销减半、训练效率翻倍的效果。为了保证表示精度，需要保留原始 32 位模型的参数副本。但在训练过程中，会先将这些 32 位参数转换为 16 位参数，随后以 16 位精度执行前向传播和反向传播等操作，最后在参数更新时再对 32 位模型进行优化。由于在模型训练中前向传播和反向传播占用了绝大部分优化时间，混合精度训练因而能够显著提升模型的训练效率。

### **模型参数量计算与效率分析**

#### **参数量计算**

由于当前主流的大模型普遍采用因果解码器架构，因此下面以 LLaMA 模型为范例，深入剖析其参数数量计算方式。首先，假设词表大小为 *𝑉*，模型包含 *𝐿* 层解码器，中间状态的维度大小为 *𝐻*，前馈网络层的中间状态维度大小为 *𝐻* ′。我们主要关注计算以下几个部分的参数量：

![image-20250329111758960](pic/image-20250329111758960.png)

![image-20250329111810677](pic/image-20250329111810677.png)

#### **训练运算量估计**

模型训练运算量指的是模型在训练过程中，需要进行的浮点运算次数（FloatingPoint Operations, FLOP）。这里的浮点运算包括浮点数的加减乘除运算，以及浮点数的指数函数，对数函数，三角函数等运算操作。使用 Transformer 架构进行训练的运算量主要集中在多头注意力计算和线性变换计算。相比之下，归一化、输出映射和旋转位置编码计算所需的运算量较少，而输入编码层则无需计算，因此后续的分析中省略了这些部分。在分析多头注意力和线性变换的运算量时，我们进一步设定以下参数：模型总参数量为 *𝑃*，批处理大小为 *𝐵*，输入序列长度为 *𝑇*，因此训练词元总数为 *𝐶* = *𝐵𝑇*；多头注意力机制包含 *𝑁* 个头，每个头的维度为 *𝐷*，因此和中间状态维度 *𝐻* 满足关系 *𝐻* = *𝑁𝐷*。其它定义与参数量计算一节 6.4.1 保持一致。

**矩阵乘法运算量**

矩阵 *𝑨* ∈ R *𝑛*×*𝑚* 和矩阵 *𝑩* ∈ R *𝑚*×*𝑝* 相乘所需的运算量为 2*𝑛𝑚 𝑝*。

```
每一行乘以每一列需要m次，相加需要m次
每一行乘以矩阵B总共需要2mp次
A乘以矩阵B总共需要2nmp次
```

![image-20250329140112459](pic/image-20250329140112459.png)

```
后向传播的运算量大致为前向传的两倍：
考虑到 Transformer 结构中大多数运算为二元运算（如两个矩阵相乘），需要分别计算损失对两个矩
阵的梯度（对权重梯度；对输入梯度 ），因此需要两倍的运算量。
对输入梯度：
即使输入数据本身不需要更新（如训练集中的样本），其梯度仍需计算，以确保前一层（如更靠近输入的层）的参数能够接收到正确的梯度信号。
```

![image-20250329140146945](pic/image-20250329140146945.png)

```
激活重计算技术（Activation Recomputation，也称为梯度检查点）的核心目的是通过牺牲计算时间来减少内存占用。在反向传播时“多计算一次”的本质是：由于前向传播阶段未保存全部中间激活值，反向传播时需要重新计算这些丢弃的激活值，导致额外的计算开销
相当于重新进行了一次前向传播

经计算确实多头注意力的运算量为线性变换的T/6H
```

![](pic/image-20250329140206549.png)

#### **训练显存估计**

接下来讨论如何估计模型在训练中需要的显存资源占用，主要可以分为三个部分：模型参数与优化器、训练中需要保存的激活值和其他显存占用。

**模型参数与优化器**

模型参数和模型梯度通常以 16 位浮点数存储，

而 Adam 或 AdamW优化器则需要额外存储 32 位浮点数的模型参数、动量参数以及动量二阶矩参数。

不使用 *ZeRO* 优化技术*.* 在这种情况下，由于一个 16 位浮点数需要 2 字节，一个 32 位浮点数需要 4 字节，因此模型参数和模型梯度各需要 2*𝑃* 字节的显存，Adam 优化器的模型参数、动量参数以及动量二阶矩参数则各需要 4*𝑃* 字节的显存。通过对于这些显存占用进行累和，每张 GPU 上会需要使用 (2+2+4+4+4) ·*𝑃* = 16*𝑃*字节的显存用于存储模型参数与优化器。

**训练激活值的显存占用**

在大模型的训练期间，前向传播中需要保留每层的激活值（中间状态），来用于后续反向传播中计算梯度并更新模型参数

![image-20250319185604434](../my_minimind/images/image-20250319185604434.png)

```
FFN层需要保存swiGLU的输出

可能还需要dropout
```

![image-20250329144837370](pic/image-20250329144837370.png)

![image-20250329144904486](pic/image-20250329144904486.png)

**SwiGLU激活函数**

![image-20250329145711802](pic/image-20250329145711802.png)

## 指令微调

### LoRA

低秩适配（Low-Rank Adaptation, LoRA）微调技术

![image-20250329150002237](pic/image-20250329150002237.png)

模型在针对特定任务进行适配时，参数矩阵往往是过参数化（Over-parametrized）的，其存在一个较低的内在秩。

![image-20250329150317476](pic/image-20250329150317476.png)

### **LoRA显存占用**

这里假设 LoRA 需要训练的参数量为 *𝑃*_LoRA，模型原始参数为P

![image-20250329152625122](pic/image-20250329152625122.png)

## **人类对齐**

基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）

尽管大语言模型在下游任务中表现出优秀的性能，这些模型有时会出现错误或具有危害性的行为，例如无法正确遵循指令、生成虚假信息、以及产生有害、有误导性以及带有偏见的表达。

旨在保证大语言模型的行为与人类期望和价值观相一致

RLHF 首先需要收集人类对于不同模型输出的偏好，然后使用收集到的人类反馈数据训练奖励模型，最后基于奖励模型使用强化学习算法（例如 Proximal Policy Optimization, PPO ）微调大语言模型。

![image-20250329153526380](pic/image-20250329153526380.png)

### **PPO** **介绍**

近端策略优化（Proximal Policy Optimization, PPO）算法是强化学习领域的一种重要优化方法，主要用于训练能够根据外部环境状态做出行为决策的策略模型。PPO 算法在策略梯度算法的基础上，主要使用优势估计来更加准确的评估决策轨迹能获得的奖励，使用了重要性采样来进行离线策略训练。此外，为了保证重要性采样的稳定性，PPO 算法通过在目标函数中加入了梯度裁剪以及相关的惩罚项来减小采样误差。为了能够实现上述优化过程，PPO 在策略模型和奖励模型的基础上，还引入了参考模型和评价模型

#### 优势估计

![image-20250329154039720](pic/image-20250329154039720.png)

在 PPO 的优势函数中，通过将决策的奖励与期望奖励做差，产生较低奖励的决策将会得到一个负的优势值，而产生较高奖励的决策会得到一个正的优势值。这些相对较差的决策就会被抑制，同时鼓励策略模型产生收益更高的决策。因此，优势函数可以帮助策略模型学习在众多决策中做出更好的选择。

#### 重要性采样

![image-20250329154209810](pic/image-20250329154209810.png)

![image-20250329154326683](pic/image-20250329154326683.png)

#### 基于梯度裁剪的目标函数

![image-20250329154430517](pic/image-20250329154430517.png)

#### KL散度

```
第一项：鼓励策略向高优势动作方向更新（最大化回报）。
第二项（KL惩罚项）：限制策略变化的幅度（最小化KL散度）。
参数 β：控制“最大化回报”与“限制策略变化”之间的权衡。

D_kl>d_target:说明策略变化过大，需增大β，加强惩罚力度。使得更新幅度减小
D_kl<d_target:说明策略变化过小，可减小β，允许更大更新。使得更新幅度增大
```

![image-20250329154826308](pic/image-20250329154826308.png)

#### 算法流程

![image-20250329155220877](pic/image-20250329155220877.png)

### **基于监督微调的对齐方法(非强化学习的对齐方法)DPO**

尽管 RLHF 已被证明是一种较为有效的语言模型对齐技术，但是它也存在一些局限性。首先，在 RLHF 的训练过程中，需要同时维护和更新多个模型，这些模型包括**策略模型、奖励模型、参考模型以及评价模型**。这不仅会占用大量的内存资源，而且整个算法的执行过程也相对复杂。此外，RLHF 中常用的近端策略优化算法在优化过程中的稳定性欠佳，对超参数的取值较为敏感，这进一步增加了模型训练的难度和不确定性。

**直接偏好优化（Direct Preference Optimization, DPO）**是一种不需要强化学习的对齐算法。由于去除了复杂的强化学习算法，DPO 可以通过与有监督微调相似的复杂度实现模型对齐，不再需要在训练过程中针对大语言模型进行采样，同时超参数的选择更加容易。

```
DPO 算法的主要思想是在强化学习的目标函数中建立决策函数与奖励函数之间的关系，以规避奖励建模的过程。形式化地，DPO 算法首先需要找到奖励函数 𝑟(𝑥, 𝑦) 与决策函数 𝜋𝜃 (𝑦|𝑥)之间的关系，即使用 𝜋𝜃 (𝑦|𝑥) 来表示 𝑟(𝑥, 𝑦)。
```

![image-20250329160443713](pic/image-20250329160443713.png)

与 RLHF 算法相比，DPO 算法没有采用强化学习算法来训练奖励模型，而是通过监督微调的方式对于语言模型进行训练。与传统有监督微调方法不同，DPO 算法中不仅训练模型生成符合人类偏好的内容，同时降低模型生成不符合人类偏好内容的概率。相比于强化学习算法 PPO，DPO 在训练过程中只需要加载策略模型和参考模型，并不用加载奖励模型和评价模型。因此，DPO 算法占用的资源更少、运行效率更高，并且具有较好的对齐性能，在实践中得到了广泛应用。

### RLHF和SFT的对比

![image-20250329161007295](pic/image-20250329161007295.png)

在 RLHF 中，我们首先学习一个奖励模型，然后利用该奖励模型通过强化学习算法（如 PPO）来改进大语言模型。而在 SFT 中，我们则采用了 Teacher-Forcing 的方法，直接优化模型对实例输出的预测概率。从本质上说，SFT 所采用的这种词元级别的训练方式是一种“行为克隆”。它利用教师的行为数据（即每个步骤的目标词元）作为监督标签，来直接训练大语言模型模仿教师的行为。在实现上，SFT 主要依赖于序列到序列的监督损失来优化模型，而 RLHF 则主要关于 SFT 和 RLHF 的进一步讨论过强化学习方法来实现大模型与人类价值观的对齐。本质上来说，为了学习教师的生成策略，SFT 采用了基于示例数据的“局部”优化方式，即词元级别的损失函数。作为对比，RLHF 则采用了涉及人类偏好的“全局”优化方式，即文本级别的损失函数。

#### SFT的缺点

关于 SFT，其作用在于“解锁”大语言模型的能力，而非向大语言模型“注入”新能力。当待学习的标注指令数据超出了大语言模型的知识或能力范围，例如训练大语言模型回答关于模型未知事实的问题时，可能会加重模型的**幻象**（Hallucination）行为。

作为一种基于行为克隆的学习方法，SFT 旨在模仿构建标注数据的教师的行为，而无法在这一过程中进行有效的行为探索。然而，标注者在写作风格、创作水平和主题偏好等方面经常存在一定的差异，这些都会使得标注数据中出现不一致的数据特征，进而影响 SFT 的学习性能。因此，在 SFT 阶段，**高质量的指令数据**（而非数量）是影响大语言模型训练的主要因素。

#### RLHF的缺点

RLHF 通过对比模型的输出数据（区分“好”输出与“坏”输出）来指导大语言模型学习正确的生成策略，它不再强迫大语言模型模仿教师的示例数据，因此可以缓解上述提到的 SFT 所导致的幻象问题。

RLHF继承了经典强化学习算法的缺点，**如样本学习效率低和训练过程不稳定等问题。**

RLHF 的过程通常会持续多轮，这是一个复杂的迭代优化过程，其中**涉及了很多重要细节的设定**（例如提示选择、奖励模型训练、PPO的超参数设置以及训练过程中对超参数的调整），都会影响整个模型的性能，对于精确的高效复现提出了较大挑战。

#### 总结

总的来说，SFT 特别适合预训练后增强模型的性能，具有实现简单、快速高效等优点；而 RLHF 可在此基础上规避可能的有害行为并进一步提高模型性能，但是实现较为困难，不易进行高效优化。

## 模型部署

### 模型蒸馏

![image-20250329162702259](pic/image-20250329162702259.png)

模型蒸馏（Model Distillation）的目标是将复杂模型（称为教师模型）包含的知识迁移到简单模型（称为学生模型）中，从而实现复杂模型的压缩。

模型蒸馏的核心思想是，引入额外的损失函数（称为蒸馏损失函数），训练学生模型的输出尽可能接近教师模型的输出。

![image-20250329162935542](pic/image-20250329162935542.png)

![image-20250329163008380](pic/image-20250329163008380.png)

相较于最终的预测分布，中间层特征提供了更为丰富的模型信息，有助于在模型蒸馏过程中实现更为有效的知识迁移。

# Deepseek

## deepseek-v3

```
可读性差，语言混杂
```

pre-training remarkably stable

two architecture：MLA(multi-head latent attention) Moe(mixture of experts)

two strategies：auxiliary-loss-free + MTP(multi-token-prediction training objective)

FP8 mixed precision

### MLA 多头隐式注意力机制 ：inference

```
用低维的向量或矩阵取代原始的kv cache里的kv；仅cache c
```

#### **kv-cache**：空间换时间

```
inference阶段才有kv-cache

attentions实际是矩阵相乘得到correlation


QK^T由于是因果模型，第一个token看不到后面的输出，需要mask掩码
不需要每次重复计算K,V，直接concat复用
由于只需要attention的最后一行，只需要计算Q的最后一行，因此q不需要cache
```

![image-20250414222734715](pic/image-20250414222734715.png)

#### low-rank kv

```
压缩到低维存储，类似lora
使用时再投影恢复
```

![image-20250414223520322](pic/image-20250414223520322.png)

```
仅cache c
```

![image-20250414223746942](pic/image-20250414223746942.png)

```
##trick  abosorb 参数矩阵
做推理的时候，不用计算k,v；直接用一个参数矩阵代替两个参数矩阵相乘，减少参数量
```

![image-20250414224008978](pic/image-20250414224008978.png)

```
同时对q也进行了压缩
但是和kv不是同一个参数矩阵
```

![image-20250414224224489](pic/image-20250414224224489.png)

#### **rotary position embeding**（absolute+relative）

绝对位置：对长句子不好（扩展性差）

相对位置：计算复杂

```
relative可能会改变角度和长度
rope仅会改变角度
```

![image-20250414224905015](pic/image-20250414224905015.png)

```
rope给qk乘上一个旋转矩阵，每个旋转矩阵代表token的absolute位置，attn的结果代表relative位置

RoPE（Rotary Position Embedding）的核心思想是通过复数空间旋转将位置信息编码到Query和Key中
```

![image-20250414225155043](pic/image-20250414225155043.png)

#### 解决kv cache和rope问题（解耦）

```
取q和k的小部分apply rope；再和另一部分未经过rope的数据concat；计算
结合了rope位置编码的优点和low-rank参数量小的优点

低秩压缩
矩阵的秩为r，则可分解为两个小矩阵的乘积；这种分解使得参数数量减少，从而降低计算复杂度。但低秩矩阵可能丢失部分信息，因此需要设计方法确保关键特征（如位置编码）不受影响

MLA在推理的时候不想缓存完整的kv cache，只想缓存压缩的ckv，但这样的话需要在升维后再重新计算rope，它也不想，所以干脆把q和k拆成两部分，一部分是位置编码的，一部分是无关位置编码的。带位置编码的k可以缓存，比缓存全部维度都带位置的k强

如下图，kv cache存储的是k_rope和潜在向量c
```

![image-20250414231551311](pic/image-20250414231551311.png)

**rope分量独自生成**

k的rope由h直接计算，不经过低秩压缩

q的rope由潜在向量c生成

关键点：rope仅作用于未被低秩压缩破坏的原始输入或中间表示，避免位置编码与压缩路径的冲突

**非rope通过低秩压缩生成**

q，k，v的非rope由低秩压缩路径生成

不携带位置信息，仅编码内容特征

**拼接后计算attn**

拼接q，k的rope和非rope，计算

```
W_qr,W_kr是rope旋转矩阵
```



![img](images/v2-db832a57ffb06b6d23b55bdf26750eac_1440w.webp)

![image-20250418103706299](images/image-20250418103706299.png)

![image-20250418115334070](images/image-20250418115334070.png)

### Moe：training

#### **origin moe**

```
dense moe计算所有的ffn加权平均
sparse moe计算top k的加权平均


moe好处：仅计算少数的expert，花费少；每个expert可以学习专长特征
```

![image-20250415094553846](pic/image-20250415094553846.png)

#### deepseek moe

```
改进:
1.划分多个expert
减少中间维度，增加expert数量，因此参数数量大致不变
类似model ensemble, 降低varience，提高模型性能

2.添加shared expert
学习公共的知识
```

![image-20250415094958016](pic/image-20250415094958016.png)

#### auxiliary-loss-free load balancing 

负载均衡

```
MoE 训练中一个关键挑战是负载均衡——某些专家可能被过度使用，而其他专家则被冷落。这不仅会导致计算资源的浪费（如部分设备闲置），还可能引发路由崩溃（Routing Collapse），即模型仅依赖少数专家，无法充分利用所有专家能力。

传统解决方案通过引入辅助损失（Auxiliary Loss） 强制均衡专家负载，但这种方法存在明显缺陷：辅助损失会向模型注入干扰梯度，与主任务（如语言建模）的优化目标冲突，导致模型性能下降。为此，本文提出了一种全新的无辅助损失负载均衡策略（Loss-Free Balancing），通过动态调整路由得分偏置（Bias）实现负载均衡，同时避免引入额外梯度干扰。



在softmax前加一个bias
如果希望增大某个expert的token数，则加上一个较大的b增大概率
```

![image-20250415112541701](pic/image-20250415112541701.png)

![image-20250416232032597](pic/image-20250416232032597.png)

```
核心思想
Auxiliary-Loss-Free Load Balancing的核心是动态调整专家路由得分的偏置，而非依赖辅助损失。具体步骤：
偏置注入：在 Top-K 路由前，为每个专家的原始路由得分 s添加动态偏置b，生成偏置得分s+b
偏置更新：根据近期负载情况调整偏置，高负载专家的偏置被抑制，低负载专家的偏置被提升
```

![img](pic/dd8364f455c7e8f0ab9023edbcedcaee.png)

### MTP：enhance model

```
优点：
1. trianing efficience
2. reasoning

应用 MTP 机制的主要目的是提升基础模型的性能，因此在实际推理阶段可以不使用 MTP 模块，基础模型能够独立完成正常推理
```

#### 以前的mtp

```
deepseek将inference阶段的EAGLE的causal模型替换train阶段的meata MTP的parallel heads模型
```

![image-20250415113246227](pic/image-20250415113246227.png)

#### llm train和Inference

```
train阶段每次都是用ground truth来训练
存在的问题：
1.每次用truth训练
2.每次预测一个token（短视）
3.training signal差 


实际上trian过程是并行的auto regressive，指将一整个句子输入，获得一整个输出；为了保证causal，需要mask
本质上还是预测一个token
```

![image-20250415113756534](pic/image-20250415113756534.png)

```
inference阶段每次将上一次预测的作为输入
```

![image-20250415113949336](pic/image-20250415113949336.png)

#### meta MTP

![image-20250415114525346](pic/image-20250415114525346.png)

```
hard transition：比如5预测A

one token prediction：此时hard transition的比例是1/7，权重是0.14
three token prediction：此时hard transition的比例是2/7，权重是0.28；可以学到更多的hard transition

缺点：
parallel heads：1预测多个token 2，3，4，违背了causal（因为3和2应该是有因果的）
```

![image-20250415114856619](pic/image-20250415114856619.png)

#### speculative decoding

```
有两种形式：
1. small LLM
2. heads（self)
```

![image-20250415151412706](pic/image-20250415151412706.png)

```
用一个大模型和一个小模型

token先给小模型处理，用大模型验证，并用大模型重新处理小模型预测错误的token
```

![image-20250415151500307](pic/image-20250415151500307.png)

![image-20250415151735664](pic/image-20250415151735664.png)

```
将小模型嵌入大模型，也就是用自己的heads作为小模型，有两者形式：
Medusa: 并行预测
EAGLE: casual预测
```

![image-20250415151952344](pic/image-20250415151952344.png)

![image-20250415120628021](pic/image-20250415120628021.png)

#### deepseek MTP

```
deeepseek MTP有两个MTP module
将main model输出之前的feature输送到MTP model1，这样MTP model1既有以前token的预测输出也有当前token的embed
最后计算三个Loss的平均值

inference不需要MTP module


使用MTP提高了planning规划和reasoning推理能力
planning: main model输送的feature包含对feature token有帮助的信息，具有planning
reasoning


首先，MTP 目标通过增加训练信号的密度可能提高数据利用效率；其次，它使模型能够提前规划表征，从而更准确地预测后续 token。
DeepSeek-V3 采用顺序预测方式，并在每个预测层级保持完整的因果关系链。
```

![image-20250415153414784](pic/image-20250415153414784.png)

![image-20250418105726712](images/image-20250418105726712.png)

![image-20250418105826356](images/image-20250418105826356.png)

## deepseek-R1

### top-down

```
COT: chain of thought

AGI：artificial general intelligence 通用人工智能

deepseek-R1注重提高reasoning ability
```

![image-20250415154636605](pic/image-20250415154636605.png)

### 两个版本

```
pre-training
post-training:SFT+RLHF

deepseek-R1-zero:不用SFT,直接使用RLHF
可读性差，语言混杂

deepseek-R1:少量SFT，接着RLHF
cold-start data + multi-stage+training + bootstraping

贡献:
post-training
distillation
```

### RL

#### sl和RL

```
AGI：artificial general intelligence 通用人工智能

sl memorise：无法超过人类
RL generize：可以超过人类
```

![image-20250415161800384](pic/image-20250415161800384.png)

```
RL和SFT的对比

sft:训练一个NN，让pred越接近ground truth，每一步都有ground truth

rl：仅在最后才给出reward
policy based 预测采取哪个action后reward大 （player)
value based 评价当前state action的好坏 (coach)
actor-critic= policy based  + value based 
```

![image-20250415162256736](pic/image-20250415162256736.png)

```
RL的特点
```

![image-20250415163102770](pic/image-20250415163102770.png)

#### alphaGo和alphaZero

```
alphaGo
本质上是search算法，只是用到了rl
stage1 supervised训练一个初始模型
stage2 self-play产生data，并训练一个policy model
stage3 使用data训练value model

alphaZero 将stage1的supervised去掉了；超过了人类水平；证明不需要人类的先验知识
```

![image-20250415163931405](pic/image-20250415163931405.png)

### background

#### scaling law

```
模型越大，性能越强

GPT: 相比bert less structure，性能极限更高 scale up（data,model,compute)
```

![image-20250415172011098](pic/image-20250415172011098.png)

#### openAI训练的范式

```
pre-training
post-training:sft,reward model,ppo



DPO不需要reward model

deepseek只有sft和rule-based
rule-based：对reward model的调整，使结果符合rule，防止出现reward-hacking
```

![image-20250415173527412](pic/image-20250415173527412.png)

![image-20250415174501708](pic/image-20250415174501708.png)

#### emergent ability

```
涌现:当模型超过一定大小，性能会极大提高
```

### COT：chain of thought

```
早期的llm随着模型增大，性能并不是一致增大，scaling law失效了（这是因为没有cot，没有system2思考）
```

![image-20250415181610559](pic/image-20250415181610559.png)

#### 早期：prompt engineering

```
test time阶段使用的：不参与模型的架构

zero-shot：不提供范例
few-shot：提供少数范例；不包含cot
few-shot-cot：提供少数范例；但是需要人工添加cot
zero-shot-cot：answer提示词step by step

COT prompting：最强性能
```

![image-20250415175833136](pic/image-20250415175833136.png)

```
answer-only：就是few-shot

cot prompting：在answer里添加cot和step by step
```

![image-20250415180438469](pic/image-20250415180438469.png)

#### SFT阶段：收集COT标注的数据训练大模型产生COT

```
self-consistent：COT prompting+Q给llm生成多个A,使用majority voting选择正确的A；
缺点是在inference阶段人为添加cot

训练模型产生cot数据？

SFT阶段训练模型产生cot
self-training：将self-consistent生产的QA对作为新的COT example；在train阶段训练模型产生cot
大模型生成数据-->筛选出high confidence数据-->训练大模型生成cot数据



rlhf阶段训练模型产生cot
```

![image-20250415180853865](pic/image-20250415180853865.png)

#### RL阶段：训练大模型产生更好的COT

```
RL选取的reward model:outcome reward model orm和precess reward model prm
reward model本质是评价cot的每一步的好坏
 

COT的reward model有两种
在rl框架下reward model
在test tiem compute框架下verifier（verifier是一个llm，可以判断生成的多个cot哪个最好）
```

![image-20250415181610559](pic/image-20250415181610559.png)

```
test time compute 的verifier


train阶段
训练generator，使用generator生成cot，将cot和人类标注的好坏的数据来训练verifier
inference阶段
Q->generator->多个s->verifier->生成多个p，选择最大的p
由于是在inference/test阶段进行的，所有叫test time compute
类似self-consistent：不同的是对所有的预测s用verifier大模型得到score，最后选择最大的score作为结果；而不是用majority vote


实验证明test time compute可以增强模型的性能
```

![image-20250415182325807](pic/image-20250415182325807.png)

```
验证cot是否正确： 

RL
final answer：直接计算最后的答案好不好；选出yes的cot
orm：outcome reward model  对整个cot打分；选出最好的cot
prm：process reward model  对cot的每一个step打分；选出每一步的最佳step，根据此step生成下一step
```

![image-20250415183505052](pic/image-20250415183505052.png)

```
deepseek-R1使用的是 SFT+final answer RL+majority voting
orm 和prm的结果明显好于deepseek-R1
可能是因为deepseek-R1使用的是GRPO，而且参数量是657B,由于emerging ability 和scaling law导致性能更好
```

![image-20250415183956081](pic/image-20250415183956081.png)

### R1的训练

```
round1: reasoning


round2: reasoning+general
reasoning data是由round1产生的data传给V3打分得到的（这一过程叫rejection sampling；拒绝采样「只有那些正确且推理清晰的输出才会被保留，质量较低的输出会被丢弃」）



其训练流程，简言之 是微调 → RL → 微调 → RL，具体而言是
1：首先收集了数千个冷启动数据来微调 DeepSeek-V3-Base 模型
2：随后，进行类似 DeepSeek-R1-Zero 的面向推理的强化学习
3：当强化学习过程接近收敛时，通过对 RL 检查点进行拒绝采样，结合 DeepSeek-V3 在写作、事实问答和自我认知等领域的监督数据，创建新的 SFT 数据，然后重新训练DeepSeek-V3-Base 模型
4：在用新数据微调后，检查点会经历额外的 RL 过程——且会考虑到所有场景的提示
经过这些步骤后，最终一个称为 DeepSeek-R1 的checkpoint，其性能可以与 OpenAI-o1-1217 相媲美

SFT (数千条cold start data)
RL/GRPO
SFT (结合rejection sampling，80w的推理和非推理数据)
RL/GRPO
```

![image-20250415191159995](pic/image-20250415191159995.png)

![image-20250417230350656](images/image-20250417230350656.png)

### rule-based reward

规则奖励建模(准确性奖励 + 格式奖励)：不用训练专门的偏好奖励模型

奖励是训练信号的来源，它决定了强化学习的优化方向

为了训练DeepSeek-R1-Zero，作者采用了一个基于规则的奖励系统(rule-based reward)，主要由两种类型的奖励组成：

**准确性Accuracy奖励**：准确性奖励模型评估响应是否正确
例如，对于具有确定性结果的数学问题，模型需要以指定格式（例如，框内）提供最终答案，从而实现基于规则的正确性验证——毕竟数学问题的答案具有明确的结果确定性与唯一性，对就是对，错就是错
**格式奖励**
除了准确性奖励模型外，作者还采用了格式奖励模型，比如要求模型在'<think>'和'</think>'标签之间放置CoT思考过程
那么，系统会检查模型输出是否正确地将推理过程包含在<think>…</think>标签内，并将最终答案包裹在<answer>…</answer>标签中
若格式正确，则模型可以获得奖励

**prompt格式输出**

```
<think>推理轨迹COT</think>，answer/response
```

![image-20250417225858096](images/image-20250417225858096.png)

![image-20250417225929873](images/image-20250417225929873.png)

### distilling

```
具有强推理能力的大模型蒸馏后推理能力仍强

实际上R1蒸馏版模型，是R1的数据集 去微调其他家的模型
不是其他家的模型 来教R1
```

# AI agent

![image-20250423161539410](pic/image-20250423161539410.png)

```
三个阶段：
reasoning/planing
tools
memory
```

## prompt engineering

### ReAct

```
reason+act

thought
search
```

![image-20250417202603436](D:/RL-study/pic/image-20250417202603436.png)

### reflextion

```
thought
search
reflextion（类似memory)
```

![](D:/RL-study/pic/image-20250417203035968.png)

## 最重要的能力-ToolCall

也叫function call

![image-20250423162128045](pic/image-20250423162128045.png)



## MCP

```
类似接口
```

![image-20250417211529486](D:/RL-study/pic/image-20250417211529486.png)





# api-key

```
import requests

#   chatmoss openai
def askllm(content):
    url = "https://api.aihao123.cn/luomacode-api/open-api/v1/chat/completions"
    headers={
        'Content-Type':'application/json',
        'Authorization':'Bearer sk-un661ob05nl5o89smn0i89cpi7idf00v6506nkg0ib1morul'
    }
    data={
        "model": "gpt-3.5-turbo",
        'messages':[
            {
            'role':'user',
            'content':content
            },
        ]
    }
    response=requests.post(url,headers=headers,json=data)
    return response.json()



# api2d openai
def askllm(content):
    url = "https://openai.api2d.net/v1/chat/completions"
    headers={
        'Content-Type':'application/json',
        'Authorization':'Bearer fk231350-Ii7dp2ZelILsfhGBun8kPGjGa7EC5X6I'
    }
    data={
        "model": "gpt-3.5-turbo",
        'messages':[
            {
            'role':'user',
            'content':content
            },
        ]
    }
    response=requests.post(url,headers=headers,json=data)
    return response.json()
```





# LLM学习路径

## 编程基础

### web框架

Gradio和Streamlit，快速构建前端页面

### 数据分析和可视化

pandas，matplotlib,seaborn

### 数据库

mysql，redis

