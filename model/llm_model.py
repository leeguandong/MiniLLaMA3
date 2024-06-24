# coding=utf-8
import sys

sys.path.append("/home/image_team/image_team_docker_home/lgd/e_commerce_llm/minillama3/")

import torch
from transformers import PreTrainedTokenizerFast, AutoTokenizer, \
    PhiForCausalLM, LlamaForCausalLM, AutoModelForCausalLM, \
    PhiConfig, LlamaConfig, AutoConfig
from model.qwen.tokenization_qwen import QWenTokenizer
from model.qwen.configuration_qwen import QWenConfig
from model.qwen.modeling_qwen import QWenLMHeadModel
from config import PhiModelConfig, MiniLLaMA3ModelConfig
from utils.utils import kaiming_initialization


class ModelHandler:
    def __init__(self, config):
        self.config = config

    def load_tokenizer(self):
        raise NotImplementedError

    def get_model(self, tokenizer):
        raise NotImplementedError

    def get_sft_model(self):
        raise NotImplementedError

    def get_dpo_model(self):
        raise NotImplementedError

    @property
    def vocab_size(self):
        return len(self.load_tokenizer())


class PhiHandler(ModelHandler):
    def load_tokenizer(self):
        # 自己训练的tokenizer
        return PreTrainedTokenizerFast.from_pretrained(self.config.tokenizer_dir)

    def get_model(self, tokenizer):
        config = PhiConfig()

        vocab_size = len(tokenizer)
        if vocab_size % 64 != 0:
            vocab_size = (vocab_size // 64 + 1) * 64
        print(f"source vocab size: {len(tokenizer)}, final vocab sieze: {vocab_size}")

        config.vocab_size = vocab_size
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id

        config.hidden_size = PhiModelConfig().hidden_size
        config.num_attention_heads = PhiModelConfig().num_attention_heads
        config.num_hidden_layers = PhiModelConfig().num_hidden_layers
        config.max_position_embeddings = PhiModelConfig().max_position_embeddings
        config.intermediate_size = PhiModelConfig().intermediate_size
        config.num_key_value_heads = PhiModelConfig().num_key_value_heads

        return PhiForCausalLM(config)

    def get_sft_model(self):
        return PhiForCausalLM.from_pretrained(self.config.model_file)

    def get_dpo_model(self):
        model_train = PhiForCausalLM.from_pretrained(self.config.model_file)
        model_ref = PhiForCausalLM.from_pretrained(self.config.model_file)
        return model_train, model_ref


class Minillama3Handler(ModelHandler):
    def load_tokenizer(self):
        # 用的llama2的tokenzier
        return AutoTokenizer.from_pretrained(self.config.tokenizer_dir)

    def get_model(self, tokenizer):
        config = AutoConfig.for_model(model_type="llama")

        config.hidden_size = MiniLLaMA3ModelConfig().hidden_size
        config.num_attention_heads = MiniLLaMA3ModelConfig().num_attention_heads
        config.num_hidden_layers = MiniLLaMA3ModelConfig().num_hidden_layers
        config.max_position_embeddings = MiniLLaMA3ModelConfig().max_position_embeddings
        config.intermediate_size = MiniLLaMA3ModelConfig().intermediate_size
        config.num_key_value_heads = MiniLLaMA3ModelConfig().num_key_value_heads

        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
        kaiming_initialization(model)
        return model

    def get_sft_model(self):
        return AutoModelForCausalLM.from_pretrained(self.config.model_file)

    def get_dpo_model(self):
        model_train = AutoModelForCausalLM.from_pretrained(self.config.model_file)
        model_ref = AutoModelForCausalLM.from_pretrained(self.config.model_file)
        return model_train, model_ref


class QwenHandler(ModelHandler):
    def load_tokenizer(self):
        tokenizer = QWenTokenizer.from_pretrained(self.config.tokenizer_dir)
        tokenizer.pad_token_id = tokenizer.im_end_id
        return tokenizer

    def get_model(self, tokenizer):
        config = QWenConfig.from_pretrained(self.config.tokenizer_dir)
        return QWenLMHeadModel(config)

    def get_sft_model(self):
        return QWenLMHeadModel.from_pretrained(self.config.model_file)

    def get_dpo_model(self):
        model_train = QWenLMHeadModel.from_pretrained(self.config.model_file)
        model_ref = QWenLMHeadModel.from_pretrained(self.config.model_file)
        return model_train, model_ref
