import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainingArguments
from trainer.unlearn.base import UnlearnTrainer
from typing import Dict, List, Optional, Any
import numpy as np
from torch.utils.data import DataLoader
import logging
import os
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class DistributionFilter(nn.Module):
    """分布过滤器模块"""
    def __init__(self, hidden_dim, filter_rank=64, filter_strength=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.filter_rank = filter_rank
        self.filter_strength = filter_strength

        # 缩放参数
        self.scale = nn.Parameter(torch.ones(1))

        # 存储过滤方向
        self.register_buffer('filter_directions', torch.eye(hidden_dim))
        self.register_buffer('is_initialized', torch.tensor(False))

    def update_filter(self, forget_embeds, retain_embeds, method='fisher'):
        """根据forget/retain集合更新过滤器"""
        logger.info(f"Updating filter with {len(forget_embeds)} forget and {len(retain_embeds)} retain samples")

        directions = self._compute_discriminant_directions(forget_embeds, retain_embeds, method)

        if directions.dim() == 1:
            directions = directions.unsqueeze(0)

        # 施密特正交化
        if directions.shape[0] > 1:
            directions = torch.qr(directions.T)[0].T

        # 构造抑制矩阵：I - α * D^T * D
        suppress_matrix = torch.eye(self.hidden_dim, device=directions.device)
        for direction in directions:
            suppress_matrix -= self.filter_strength * torch.outer(direction, direction)

        self.filter_directions = suppress_matrix
        self.is_initialized = torch.tensor(True)

        logger.info("Filter updated successfully")

    def _compute_discriminant_directions(self, forget_embeds, retain_embeds, method='fisher'):
        """计算判别方向"""
        if method == 'mean_diff':
            forget_mean = forget_embeds.mean(dim=0)
            retain_mean = retain_embeds.mean(dim=0)
            direction = forget_mean - retain_mean
            direction = direction / torch.norm(direction)
            return direction

        elif method == 'fisher':
            forget_mean = forget_embeds.mean(dim=0)
            retain_mean = retain_embeds.mean(dim=0)

            # 计算类内协方差
            forget_centered = forget_embeds - forget_mean
            retain_centered = retain_embeds - retain_mean

            S_w = (forget_centered.T @ forget_centered +
                   retain_centered.T @ retain_centered) / (len(forget_embeds) + len(retain_embeds) - 2)

            # Fisher方向
            try:
                direction = torch.linalg.solve(S_w + 1e-6 * torch.eye(S_w.shape[0], device=S_w.device),
                                               (forget_mean - retain_mean).unsqueeze(1)).squeeze()
            except:
                # 如果求解失败，回退到均值差异方法
                direction = forget_mean - retain_mean

            direction = direction / torch.norm(direction)
            return direction

        elif method == 'pca':
            forget_centered = forget_embeds - forget_embeds.mean(dim=0)
            try:
                U, S, V = torch.svd(forget_centered.T)
                k = min(self.filter_rank, U.shape[1])
                return V[:, :k].T
            except:
                # SVD失败时回退到均值差异
                return self._compute_discriminant_directions(forget_embeds, retain_embeds, 'mean_diff')

        else:
            raise ValueError(f"Unknown method: {method}")

    def forward(self, hidden_states):
        """应用分布过滤"""
        if not self.is_initialized:
            return hidden_states

        original_shape = hidden_states.shape
        hidden_flat = hidden_states.view(-1, self.hidden_dim)
        filtered = hidden_flat @ self.filter_directions.T
        return filtered.view(original_shape) * self.scale


class FlashModel(nn.Module):
    def __init__(self, base_model, filter_rank, filter_strength, discriminant_method):
        super().__init__()
        # super().__init__(base_model.config)


        self.base_model = base_model
        self.config = base_model.config

        # 创建分布过滤器
        self.filter = DistributionFilter(
            hidden_dim=self.config.hidden_size,
            filter_rank=filter_rank,
            filter_strength=filter_strength
        )

        self.filter_enabled = True

        self.device = next(self.base_model.parameters()).device


    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # 获取base model输出
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )

        if self.filter_enabled and self.filter.is_initialized:
            # 获取最后一层hidden states
            last_hidden_states = outputs.hidden_states[-1]

            # 应用过滤器
            filtered_states = self.filter(last_hidden_states)

            # 重新计算logits
            if hasattr(self.base_model, 'lm_head'):
                logits = self.base_model.lm_head(filtered_states)
            elif hasattr(self.base_model, 'embed_out'):
                logits = self.base_model.embed_out(filtered_states)
            else:
                # 尝试找到输出层
                for name, module in self.base_model.named_modules():
                    if 'lm_head' in name.lower() or 'output' in name.lower():
                        logits = module(filtered_states)
                        break
                else:
                    raise ValueError("Cannot find output layer in the model")

            # 重新计算loss（如果有labels）
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            logits = outputs.logits
            loss = outputs.loss

        return type(outputs)(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=getattr(outputs, 'attentions', None)
        )

    def generate(self, *args, **kwargs):
        """生成文本，应用过滤器"""
        # 确保模型处于评估模式
        self.eval()

        # 保存原始 forward 方法
        original_forward = self.base_model.forward

        def filtered_forward(input_ids, attention_mask=None, **forward_kwargs):
            """包装的 forward 方法，调用 FlashModel 的 forward"""
            # 确保获取隐藏状态
            forward_kwargs['output_hidden_states'] = True

            # 调用 FlashModel 的 forward 方法
            return self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **forward_kwargs
            )

        # 临时替换 base_model 的 forward 方法
        self.base_model.forward = filtered_forward

        try:
            # 调用 base_model 的 generate 方法
            result = self.base_model.generate(*args, **kwargs)
        finally:
            # 恢复原始 forward 方法
            self.base_model.forward = original_forward

        return result



class FlashUnlearn(UnlearnTrainer):
    def __init__(self, filter_rank=64, filter_strength=1.2, discriminant_method="fisher", *args, **kwargs):
        # 首先调用父类的__init__来设置self.model
        super().__init__(*args, **kwargs)

        self.wrapped_model = FlashModel(self.model, filter_rank, filter_strength, discriminant_method)
        self.discriminant_method = discriminant_method
        self.filter_computed = False


    def _extract_features(self, dataset):
        """从数据集中提取分布特征"""
        # 创建dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=min(4, len(dataset)),  # 小batch避免内存问题
            collate_fn=self.data_collator,
            shuffle=False
        )
        forget_features = []
        retain_features = []
        self.model.eval()

        with torch.no_grad():
            for batch in dataloader:

                forget_batch = {
                    'input_ids': batch['forget']['input_ids'].to(self.args.device),
                    'attention_mask': batch['forget']['attention_mask'].to(self.args.device)
                }

                # 获取forget hidden states
                forget_outputs = self.model(
                    input_ids=forget_batch['input_ids'],
                    attention_mask=forget_batch['attention_mask'],
                    output_hidden_states=True
                )

                # 取最后一层的平均池化
                forget_last_hidden = forget_outputs.hidden_states[-1]  # [batch, seq, hidden]
                forget_mask = forget_batch['attention_mask'].unsqueeze(-1).float()
                forget_pooled = (forget_last_hidden * forget_mask).sum(dim=1) / forget_mask.sum(dim=1)
                forget_features.append(forget_pooled)

                retain_batch = {
                    'input_ids': batch['retain']['input_ids'].to(self.args.device),
                    'attention_mask': batch['retain']['attention_mask'].to(self.args.device)
                }

                retain_outputs = self.model(
                    input_ids=retain_batch['input_ids'],
                    attention_mask=retain_batch['attention_mask'],
                    output_hidden_states=True
                )

                # 取最后一层的平均池化
                retain_last_hidden = retain_outputs.hidden_states[-1]
                retain_mask = retain_batch['attention_mask'].unsqueeze(-1).float()
                retain_pooled = (retain_last_hidden * retain_mask).sum(dim=1) / retain_mask.sum(dim=1)
                retain_features.append(retain_pooled)

        forget_features = torch.cat(forget_features, dim=0)
        retain_features = torch.cat(retain_features, dim=0)

        return forget_features, retain_features

    def compute_loss(self, model, inputs, return_outputs=False):
        """在第一次调用时计算filter"""
        if not self.filter_computed:
            logger.info("Computing filter for the first time...")
            self._compute_and_apply_filter(self.wrapped_model)
            self.filter_computed = True
            logger.info("Filter computation completed!")

        # 返回dummy loss，确保trainer正常运行
        dummy_loss = torch.tensor(0.0, requires_grad=True, device=self.args.device)

        if return_outputs:
            vocab_size = getattr(self.model.config, 'vocab_size', 50257)
            dummy_outputs = type('DummyOutputs', (), {
                'loss': dummy_loss,
                'logits': torch.zeros(1, 1, vocab_size, device=self.args.device)
            })()
            return dummy_loss, dummy_outputs

        # logger.info(f'loss: {dummy_loss:.6f}')
        return dummy_loss

    def _compute_and_apply_filter(self, model):
        logger.info("Computing distribution filter...")

        # 提取forget和retain特征
        forget_features, retain_features = self._extract_features(self.train_dataset)

        # 更新过滤器
        method = self.discriminant_method
        model.filter.update_filter(forget_features, retain_features, method)

        logger.info("Distribution filter applied successfully")


    def save_flash_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """保存wrapped_model，包括base_model和filter状态"""
        if output_dir is None:
            output_dir = self.args.output_dir
        logger.info(f'output_dir: {output_dir}')
        # 保存原始的base model
        # self.wrapped_model.base_model.save_pretrained(output_dir)
        # self.model.save_pretrained(output_dir)

        # 保存filter的完整状态
        filter_state = {
            'filter_directions': self.wrapped_model.filter.filter_directions,
            'scale': self.wrapped_model.filter.scale,
            'is_initialized': self.wrapped_model.filter.is_initialized,
            'filter_rank': self.wrapped_model.filter.filter_rank,
            'filter_strength': self.wrapped_model.filter.filter_strength,
            'hidden_dim': self.wrapped_model.filter.hidden_dim,
        }

        # 保存filter状态
        torch.save(filter_state, f"{output_dir}/flash_filter.pt")

        # 保存完整的wrapped_model（可选，用于直接加载）
        # torch.save(self.wrapped_model.state_dict(), f"{output_dir}/wrapped_model.pt")

        logger.info(f"FlashUnlearn wrapped_model saved to {output_dir}")


    def training_step(self, model, inputs):
        model.train()
        loss = self.compute_loss(model, inputs)
        return loss.detach()

    @classmethod
    def load_flash_model(cls, model_path, base_model_class=None, **model_kwargs):
        """加载FlashModel用于推理"""

        # 加载base model
        if base_model_class is not None:
            base_model = base_model_class.from_pretrained(model_path, **model_kwargs)
        else:
            # 自动检测模型类型
            from transformers import AutoModelForCausalLM
            base_model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        logger.info(f'base model has been loaded')

        # 加载filter状态
        filter_path = f"{model_path}/flash_filter.pt"
        if not os.path.exists(filter_path):
            raise FileNotFoundError(f"Filter state not found at {filter_path}")

        filter_state = torch.load(filter_path, map_location='cpu')

        # 创建FlashModel
        flash_model = FlashModel(
            base_model=base_model,
            filter_rank=filter_state['filter_rank'],
            filter_strength=filter_state['filter_strength'],
            discriminant_method='fisher'  # 这个在推理时不重要
        )

        # 恢复filter状态
        flash_model.filter.filter_directions = filter_state['filter_directions']
        flash_model.filter.scale = filter_state['scale']
        flash_model.filter.is_initialized = filter_state['is_initialized']

        return flash_model