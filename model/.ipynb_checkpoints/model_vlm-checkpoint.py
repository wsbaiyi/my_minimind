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
        super().__init__()
        self.ve_dim = ve_dim
        self.lm_dim = lm_dim
        self.vision_proj = nn.Sequential(
            nn.Linear(self.ve_dim, self.lm_dim)
        )

    def forward(self, image_encoders):
        vision_proj = self.vision_proj(image_encoders)
        return vision_proj


# 继承自语言模型
class MiniMindVLM(MiniMindLM):
    config_class = VLMConfig

    def __init__(self, params: VLMConfig = None):
        super().__init__(params)
        if not params: params = VLMConfig()
        self.params = params
        self.vision_encoder, self.processor = self.__class__.get_vision_model()
        self.vision_proj = VisionProj(lm_dim=params.dim)

    @staticmethod
    def get_vision_model(model_path="/root/llm_learn/model/vision_model/clip-vit-base-patch16"):
        model = CLIPModel.from_pretrained(model_path)
        processor = CLIPProcessor.from_pretrained(model_path)
        # 冻结 vision_encoder 的所有参数
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    # image转为tensor
    def image2tensor(image, processor):
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")['pixel_values']
        return inputs

    @staticmethod
    #  image_tensors.shape (bs, num, c, im_h, im_w)
    #  tensor转为emb
    def get_image_embeddings(image_tensors, vision_model):
        with torch.no_grad():
            
            outputs = vision_model.vision_model(pixel_values=image_tensors)
            # 是ViT模型最后一层输出的所有token的隐藏表示

        # outputs.last_hidden_state.shape=(batch_size, num_patches + 1, hidden_size)
        img_embedding = outputs.last_hidden_state[:, 1:, :].squeeze()#

        # shape=(batch_size, num_patches, hidden_size)
        # num_patches=16*16=196   hidden_size=768
        return img_embedding

    # vision_tensors投影到tokens上
    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
      # 将图片向量嵌入到文本向量序列中
        
        # vision_tensors.size=(bs,num_images,num_patches,hidden_dim)
        # token.size=(bs,seq_len)
        # image_ids是特殊的@
        def find_indices(tokens, image_ids):
          # 寻找图片向量应嵌入位置的下标
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
            vision_proj = self.vision_proj(vision_tensors)
            if len(vision_proj.shape) == 3:
                vision_proj = vision_proj.unsqueeze(0)
            new_h = []

            # h.size(bs,seq_len,dim)
            for i in range(h.size(0)):
                if i in image_indices:
                    h_i = h[i]
                    img_idx = 0
                    for start_idx, end_idx in image_indices[i]:
                        if img_idx < vision_proj.size(1):
                            h_i = torch.cat((h_i[:start_idx], vision_proj[i][img_idx], h_i[end_idx + 1:]), dim=0)[
                                  :seqlen]
                            img_idx += 1
                    new_h.append(h_i)
                    # 拼接后的h_i.shape=(seq_len,hidden_dim)
                else:
                    new_h.append(h[i])
            return torch.stack(new_h, dim=0)
        return h

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **args):
        start_pos = args.get('start_pos', 0)
        # start_pos = args.get('start_pos', default = 0)
        pixel_tensors = args.get('pixel_tensors', None)
        h = self.tok_embeddings(input_ids)

        if pixel_tensors is not None and start_pos == 0:
            if len(pixel_tensors.shape) == 6:
                pixel_tensors = pixel_tensors.squeeze(2)
            bs, num, c, im_h, im_w = pixel_tensors.shape # num张图片是因为一段文字可以对应不只一张图片

            # 在训练时，批次是第一个维度0，而每个样本可能有多个图像嵌入，因此在第二个维度1堆叠可以保持形状为(bs, num, ...)，
            # 而在推理时，批次是1，没有维度，所以堆叠在0维可能形成(num, ...)的形状
            stack_dim = 1 if bs > 1 else 0 # 推理时bs=1 训练时bs>1 

            # (bs,num,num_patches,hidden_dim) 训练 / (num,num_patches,hidden_dim) 推理
            vision_tensors = torch.stack([

                # num个shape=(bs,num_patches,hidden_dim)的编码向量stack起来
                MiniMindVLM.get_image_embeddings(pixel_tensors[:, i, :, :, :], self.vision_encoder)
                for i in range(num)
            ], dim=stack_dim)

            # vision_tensors 投影到 input_ids的维度
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
