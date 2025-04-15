import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
import ast
from PIL import Image
from .model_vlm import MiniMindVLM


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# 多模态数据集
class VLMDataset(Dataset):
    def __init__(self, jsonl_path, images_path, tokenizer, preprocess=None, max_length=512,
                 image_special_token='@' * 196):
        # vit-base-patch16视觉编码器，因此用16*16=196个imagetoken
        super().__init__()
        self.samples = self.load_data(jsonl_path)
        self.images_path = images_path

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_token = image_special_token
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
                # 数据集格式如下：
                # {"conversations": 
                #  [
                #   {"role": "user", "content": "提供给定图像的简要描述。\n<image>"}, 
                #   {"role": "assistant", "content": "橄榄油是自由使用的健康成分。"}
                #  ], 
                #  "image": "GCC_train_002582585.jpg"
                # }
                
        return samples

    def _create_chat_prompt(self, conversations):
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content'].replace('<image>', self.image_token)})
        # 下面是构造的对话模板
        # <s>system
        # 你是 MiniMind，是一个有用的人工智能助手。</s>
        # <s>user
        # 提供给定图像的简要描述。
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@</s>
        # <s>assistant
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_paths = sample['image']
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = self._generate_loss_mask(input_ids)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        image_tensors = []
        for image_name in image_paths.split(','):
            image_name = image_name.strip()

            # image对象
            image = Image.open(f'{self.images_path}/{image_name}')

            # 
            image_tensor = MiniMindVLM.image2tensor(image, self.preprocess)
            image_tensors.append(image_tensor)
        image_tensors = torch.stack(image_tensors, dim=0)

        # image_tensors.shape (B, 1, C, H, W)
        return X, Y, loss_mask, image_tensors


# llm数据集
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        # start=1 是可选参数，用于指定索引的起始值，默认值为 0。
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 构建输入文本，添加开始和结束标签bos_token和eos_token
        # {"text": "如何才能摆脱拖延症？ 治愈拖延症并不容易，但以下建议可能有所帮助..."}  
        # {"text": "<s>如何才能摆脱拖延症？ 治愈拖延症并不容易，但以下建议可能有所帮助...</s>"}
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # print(encoding.input_ids.shape)
        # torch.Size([1, 512]) 因此squeeze
        # 移除了所有长度为 1 的维度。
        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        # loss_mask和Y的位置一致，不是pad的位置为1
        # 对齐预测位置
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask

# SFT，添加bos_id和eos_id
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)

        # SFT
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())

                # 下面是reason distill时的数据格式 由于数据中加入了<think>与<answer>标签，所以回答时也会按照相应的格式回答
                #      {"role": "assistant", "content": 
                #          "<think>\n嗯，用户让我用一段话描述阿里巴巴集团的企业文化。\n</think>\n
                #          <answer>\n阿里巴巴集团ix platform的建立旨在帮助员工实现高效协作，激发创新精神，吸引更多优秀人才加入，共同推动企业不断向前发展。\n</answer>"
    
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        # [{'role': 'user', 'content': '请告诉我在中国古代的“四大发明”是什么？'},
        #  {'role': 'assistant', 'content': '中国古代的“四大发明”是'}]
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})

        # 当add_generation_prompt设置为 True 时，方法会在处理后的文本中添加一个生成提示，用于引导模型生成下一个回复；当设置为 False 时，不会添加生成提示
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    # 仅将<s>assistant</s>中间的内容置1
    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):

            # 如果找到开始bos
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                # 找到eos的end位置
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 为什么从start+1开始？？？
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        prompt = self._create_chat_prompt(sample['conversations'])
        # print(prompt)
        # <s>system
        # 你是 MiniMind，是一个有用的人工智能助手。</s>
        # <s>user
        # 请用一段话描述阿里巴巴集团的企业文化。</s>
        # <s>assistant
        # 阿里巴巴集团的企业文化以“客户第一、员工第二、股东第三”为核心价值观，强调“让天下没有难做的生意”的使命。公司倡导开放、透明、分享、责任的团队合作精神，鼓励员工创新、追求卓越，同时注重员工的个人成长和幸福感。阿里巴巴的企业文化还体现在其独特的“六脉神剑”价值观体系中，包括客户第一、拥抱变化、团队合作、诚信、激情、专业等六个方面，这些价值观不仅指导着公司的日常运营，也深深影响着每一位阿里人的行为准则。</s>
        # 这里的第一句system消息是由设置好的tokenlizer的配置文件中设置好的


        # 截断填充
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成动态损失掩码
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置

        return X, Y, loss_mask

# DPO padding id
# “直接偏好优化”（Direct Preference Optimization，DPO）
class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        chosen = item['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = item['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        # print(rejected_prompt)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

if __name__ == "__main__":
    pass
