import random
from tqdm import tqdm
from transformers import AutoTokenizer
import json
# from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
import os

random.seed(42)


def train_tokenizer():
    # 读取JSONL文件并提取文本数据
    def read_texts_from_jsonl(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                yield data['text']

    data_path = '../dataset/pretrain_hq.jsonl'

    # 初始化tokenizer
    # 初始化一个基于BPE（Byte-Pair Encoding）算法的分词器，并配置其预分词步骤
    # 预分词器（Pre-tokenizer）：在应用BPE前对原始文本进行初步切分。
    # ByteLevel：选择字节级别的预分词器，将文本按UTF-8字节处理，支持所有Unicode字符且无需处理未知字符。
    # add_prefix_space=False：不在文本开头添加空格。若为True，会在文本起始处加空格（类似GPT-2，用于标识单词起始），设为False则避免此行为。
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 定义特殊token
    special_tokens = ["<unk>", "<s>", "</s>"]

# BPE 算法从一个初始的字符集（由 initial_alphabet 确定，这里是 pre_tokenizers.ByteLevel.alphabet() 提供的字节级字母表）开始，这个初始字符集里的每个字符都可以看作是一个初始子词。
# 合并过程：训练过程中，BPE 会不断地找出最频繁共现的字节对，并将它们合并成一个新的子词。每进行一次合并操作，就会生成一个新的子词，这个过程会持续进行，直到词汇表达到指定的 vocab_size。例如，如果 vocab_size 设定为 6400，那么训练过程会持续合并字节对，直到词汇表中包含 6400 个子词（包括特殊标记）。
# special_tokens 参数：special_tokens 中指定的特殊标记会被确保包含在最终的词汇表中。这些特殊标记（如分隔符、未知词标记等）会占用词汇表中的位置。
# 假设 special_tokens 中有 3 个特殊标记，那么在生成子词的过程中，实际可用于普通子词的位置就只有 vocab_size - len(special_tokens) 个，也就是 6397 个

    trainer = trainers.BpeTrainer(
        vocab_size=6400,
        special_tokens=special_tokens,  # 确保这三个token被包含
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 读取文本数据
    texts = read_texts_from_jsonl(data_path)

    # 训练tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 检查特殊token的索引
    assert tokenizer.token_to_id("<unk>") == 0
    assert tokenizer.token_to_id("<s>") == 1
    assert tokenizer.token_to_id("</s>") == 2

    # 保存tokenizer
    tokenizer_dir = "./model/minimind_tokenizer"
    # exist_ok=True 防止目录已存在时报错。
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    # 需要额外保存模型文件（如 vocab.json 和 merges.txt）。
    tokenizer.model.save("./model/minimind_tokenizer")

    # 手动创建配置文件
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "0": {
                "content": "<unk>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<s>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "</s>",
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<unk>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<s>system\\n' + system_message + '</s>\\n' }}{% else %}{{ '<s>system\\n你是 MiniMind，是一个有用的人工智能助手。</s>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
    }

    # 保存配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("Tokenizer training completed and saved.")


def eval_tokenizer():
    from transformers import AutoTokenizer

    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./model/minimind_tokenizer")

    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]




    # 功能：将messages按照模型的聊天模板格式化为字符串。
    # tokenize=False：返回格式化后的文本（若为True则返回分词后的Token IDs）。
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print(new_prompt)

    # 获取实际词汇表长度（包括特殊符号）
    actual_vocab_size = len(tokenizer)
    print('tokenizer实际词表长度：', actual_vocab_size)

    model_inputs = tokenizer(new_prompt)
    print('encoder长度：', len(model_inputs['input_ids']))

    input_ids = model_inputs['input_ids']
    # 若设置skip_special_tokens=False，解码后的文本会保留这些标记，确保对话的结构清晰可见。这对于调试或验证输入/输出格式是否符合预期非常关键。
    response = tokenizer.decode(input_ids, skip_special_tokens=False)
    print('decoder和原始文本是否一致：', response == new_prompt)


def main():
    train_tokenizer()
    eval_tokenizer()


if __name__ == '__main__':
    main()
