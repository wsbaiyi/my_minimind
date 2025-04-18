from .VLMConfig import VLMConfig
from .model import *
from typing import Optional, Tuple, List
from torch import nn
import warnings
from transformers import CLIPProcessor, CLIPModel
import torch

warnings.filterwarnings('ignore')


# 投影，将图片维度768转为token维度512
class VisionProj(nn.Module):
    def __init__(self, ve_dim=768, lm_dim=512):
        """
        将视觉特征投影到语言模型维度
        :param ve_dim: 视觉编码器输出维度（如CLIP-ViT的768）
        :param lm_dim: 语言模型隐层维度（如512）
        """
        super().__init__()
        self.vision_proj = nn.Sequential(
            nn.Linear(ve_dim, lm_dim)  # 线性投影层
        )

    def forward(self, image_encoders):
        """
        输入: image_encoders (形状为 [B, num_patches, ve_dim])
        输出: 投影后的视觉特征 [B, num_patches, lm_dim]
        """
        return self.vision_proj(image_encoders)

# 继承自语言模型
class MiniMindVLM(MiniMindLM):
    config_class = VLMConfig  # 继承自语言模型的配置类

    def __init__(self, params: VLMConfig = None):
        """
        初始化多模态视觉语言模型（VLM）
        :param params: 配置参数（包含图像和文本模型参数）
        """
        super().__init__(params)  # 初始化语言模型
        self.params = params
        # 加载冻结的CLIP视觉编码器和处理器
        self.vision_encoder, self.processor = self.get_vision_model()
        # 视觉特征投影层（将CLIP的768维映射到语言模型维度）
        self.vision_proj = VisionProj(lm_dim=params.dim)

    @staticmethod
    def get_vision_model(model_path="/path/to/clip-vit-base-patch16"):
        """
        加载预训练的CLIP视觉模型并冻结参数
        :param model_path: CLIP模型路径
        :return: (CLIP模型实例, CLIP处理器)
        """
        model = CLIPModel.from_pretrained(model_path)
        processor = CLIPProcessor.from_pretrained(model_path)
        for param in model.parameters():  # 冻结所有参数
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    # eval时使用 
    # image转为tensor
    # 输出是(1, C, H, W)，每次输入单张图片；
    # 输出是(B, C, H, W)，每次输入多张图片；
    def image2tensor(image, processor):
        """
        将图像转换为CLIP输入张量（归一化后的像素值）
        :param image: PIL图像对象
        :param processor: CLIP处理器
        :return: 形状为 [1, C, H, W] 的张量（推理）或 [B, C, H, W]（训练）
        """
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")['pixel_values']
        return inputs

    @staticmethod
    #  image_tensors.shape (bs, c, im_h, im_w)
    #  tensor转为emb
    # 3333
    def get_image_embeddings(image_tensors, vision_model):
        """
        提取图像嵌入（ViT的输出特征）
        :param image_tensors: 输入张量 [B, C, H, W]
        :param vision_model: CLIP的视觉编码器
        :return: 图像嵌入 [B, num_patches, ve_dim]（去除了CLS标记）
        """
        with torch.no_grad():
            
            outputs = vision_model.vision_model(pixel_values=image_tensors)
            # 是ViT模型最后一层输出的所有token的隐藏表示

        # 一个patch看作是一个token
        # hidden_size=768
        # outputs.last_hidden_state.shape=(batch_size, num_patches + 1, hidden_size)
        # 因为第一个位置是特殊的标记（如 CLS 标记）
        # 缩减所有的维度值为1，例如batch_size=1，此时batch就消失了
        img_embedding = outputs.last_hidden_state[:, 1:, :].squeeze()#

        # shape=(batch_size, num_patches, hidden_size)
         # 取所有patch的嵌入（排除CLS标记）
        return img_embedding

    # vision_tensors投影到tokens上
        # 4444
    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
      # 将图片向量嵌入到文本向量序列中
        
        # vision_tensors.size=(bs,num_images,num_patches,hidden_dim)
        # token.size=(bs,seq_len)
        # image_ids [34] * 196,# 34是'@'经过tokenlizer编码解码得到的值

        """
        将图像嵌入插入文本序列中（替换特殊标记@）
        :param tokens: 文本token序列 [B, seq_len]
        :param h: 文本嵌入张量 [B, seq_len, dim]
        :param vision_tensors: 投影后的图像嵌入 [B, num_images, num_patches, dim]
        :param seqlen: 最大序列长度
        :return: 融合后的嵌入张量 [B, seq_len, dim]
        """
        def find_indices(tokens, image_ids):
          # 寻找图片向量应嵌入位置的下标
            """
            查找文本中特殊图像标记的位置（如'@@@@...'对应196个'@'）
            :return: 字典 {batch_idx: [(start_idx, end_idx), ...]}
            """
            image_ids_tensor = torch.tensor(image_ids).to(tokens.device)
            len_image_ids = len(image_ids)
            if len_image_ids > tokens.size(1):
                return None
                
    
            # unfold的用法是创建一个滑动窗口视图
            # tokens_view.size=(batch_size, num_windows, len_image_ids) 
            tokens_view = tokens.unfold(dimension = 1, size = len_image_ids, step = 1)
        
            # matches的形状是(batch_size, num_windows)，每个元素是布尔值，表示该窗口是否完全匹配image_ids。
            # 类似于卷积操作的滑动窗口 image_ids_tensor.shape=(num_patches,)
            # .all()检查张量中指定维度上的所有元素是否都为True+
            # matches[i,j]=True 表示这个batch第i个句子的token序列中从j下标开始的连续num_patches个token需要被替换为一个句子
            matches = (tokens_view == image_ids_tensor).all(dim=2)
            
             # 返回的是一个字典 key 为batch_idx 值为这个batch中所有图片起始坐标与结束坐标的起始位置
            # key : [(image0_start_index,image0_end_index),(image1_start_index,image1_end_index),...,]
            return {
                batch_idx: [(idx.item(), idx.item() + len_image_ids - 1) for idx in
                            # nonzero(as_tuple=True)返回的是tuple，第一个元素是非零的索引
                            matches[batch_idx].nonzero(as_tuple=True)[0]]
                            # nonzero返回的是非零元素的坐标
                            # matches[batch_idx].nonzero(as_tuple=True)[0]返回的是这个行向量中非零元素的索引
                for batch_idx in range(tokens.size(0)) if matches[batch_idx].any() 
                #.any() 方法用于检查该批次的布尔值中是否至少有一个为 True
            } or None

        image_indices = find_indices(tokens, self.params.image_ids)
        if vision_tensors is not None and image_indices:

            # (bs,num_images,num_patches,dim)
            vision_proj = self.vision_proj(vision_tensors)
            if len(vision_proj.shape) == 3:
                vision_proj = vision_proj.unsqueeze(0)
            new_h = []

            # h.size(bs,seq_len,dim)
            for i in range(h.size(0)):

                # 遍历字典
                if i in image_indices:

                    # (seq_len,dim)
                    h_i = h[i]
                    img_idx = 0
                    for start_idx, end_idx in image_indices[i]:
                        if img_idx < vision_proj.size(1):

                            # 拼接后的h_i.shape=(seq_len,hidden_dim)
                            h_i = torch.cat((h_i[:start_idx], vision_proj[i][img_idx], h_i[end_idx + 1:]), dim=0)[
                                  :seqlen]
                            img_idx += 1
                    new_h.append(h_i)
                    
                else:
                    new_h.append(h[i])
            return torch.stack(new_h, dim=0)
        return h
    
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **args):
        """
        前向传播（融合文本和图像）
        :param input_ids: 文本token序列 [B, seq_len]
        :param past_key_values: 缓存的键值对（用于生成）
        :param use_cache: 是否缓存中间结果
        :param args: 其他参数（如start_pos、pixel_tensors）
        :return: 输出字典（logits, aux_loss, past_key_values）
        """
        start_pos = args.get('start_pos', 0)
        # start_pos = args.get('start_pos', default = 0)

        # pixel_tensors.shape(B, 1, C, H, W)
        pixel_tensors = args.get('pixel_tensors', None)

        # 文本嵌入
        h = self.tok_embeddings(input_ids)

        
        if pixel_tensors is not None and start_pos == 0:

            # 为什么会多一个维度？？？？
            # 处理图像嵌入（仅在序列起始位置处理）
            if len(pixel_tensors.shape) == 6:
                pixel_tensors = pixel_tensors.squeeze(2)
                
            bs, num, c, im_h, im_w = pixel_tensors.shape # num张图片是因为一段文字可以对应不只一张图片

            # 在训练时，批次是第一个维度0，而每个样本可能有多个图像嵌入，因此在第二个维度1堆叠可以保持形状为(bs, num, ...)，
            # 而在推理时，批次是1，没有维度，所以堆叠在0维可能形成(num, ...)的形状
            stack_dim = 1 if bs > 1 else 0 # 推理时bs=1 训练时bs>1 

            # (bs,num,num_patches,hidden_dim) 训练 / (num,num_patches,hidden_dim) 推理
            # 提取图像嵌入并投影
            vision_tensors = torch.stack([

                # num个shape=(bs,num_patches,hidden_dim)的编码向量stack起来
                # get_image_embeddings会缩减所有的维度值为1，例如batch_size=1，此时batch就消失了
                MiniMindVLM.get_image_embeddings(pixel_tensors[:, i, :, :, :], self.vision_encoder)
                for i in range(num)
            ], dim=stack_dim)

            # vision_tensors 投影到 h的维度
            # 将图像嵌入插入文本序列
            h = self.count_vision_proj(tokens=input_ids, h=h, vision_tensors=vision_tensors, seqlen=input_ids.shape[1])

        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.shape[1]]
        past_kvs = []
        for l, layer in enumerate(self.layers):
            h, past_kv = layer(
                h, pos_cis,
                past_key_value=past_key_values[l] if past_key_values else None,
                use_cache=use_cache
            )
            past_kvs.append(past_kv)

        logits = self.output(self.norm(h))
        aux_loss = sum(l.feed_forward.aux_loss for l in self.layers if isinstance(l.feed_forward, MOEFeedForward))

        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT
