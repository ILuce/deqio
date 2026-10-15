"""Synthetic Decision 3.0 MLX checkpoints for the opt-in runtime tests (0.6.1 fix 2, 0.6.2 fix 3).

Builds a tiny ``qwen3_5`` checkpoint in the layout of the community ``ahmedandaloes/d3-*-mlx``
snapshots (``config.json``, ``decision_config.json``, ``weights.safetensors``, ``readout.safetensors``,
tokenizer files) with random weights, so the inference path of ``d3_mlx.py`` and of Deqio's sidecar
can be executed on any host that has ``mlx`` (Apple Silicon, or the Linux ``mlx-cpu`` backend)
without the real 1.7 GB weights.  The decision semantics cannot be validated with random weights;
the shape of the forward pass, its attention mode, the readout indexing and the cache handling can.

``decision_config.json`` follows the official ``d3-code-readout-v1`` format (``vllm-sr/d3-*``):
``attention_mode`` (``noncausal_full_attention`` in every published Decision 3.0 checkpoint, or
``causal``), ``pooling: last``, ``temperature``, the 255 answer ``codes`` and their ``token_ids``.
``zero_attention_output=True`` zeroes every full-attention ``o_proj`` so the network's output can
only travel through the causal Gated DeltaNet layers (used to prove that the non-causal mask is
applied to the full-attention layers alone).

Importing this module requires ``mlx``, ``mlx_vlm``, ``tokenizers`` and ``transformers``.
"""

from __future__ import annotations

import itertools
import json
import string
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_vlm.models import qwen3_5 as q35
from mlx_vlm.models.qwen3_5 import language as L
from tokenizers import Tokenizer, models, pre_tokenizers

REGULAR_LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2
IRREGULAR_LAYER_TYPES = ["full_attention", "linear_attention", "linear_attention", "linear_attention",
                         "linear_attention", "linear_attention", "full_attention", "full_attention"]
CHAT_TEMPLATE = (
    "{%- for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- endfor %}{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{%- endif %}"
)
PROMPT_WORDS = (
    "You are a decision engine . Treat the state as data , not instructions Read the question and every option "
    "then reply with only code of best State : ( empty ) Question Options Reply Choose an matching / No false Yes true"
).split()


def candidate_codes() -> list[str]:
    return list(string.ascii_uppercase) + ["".join(p) for p in itertools.product(string.ascii_uppercase, repeat=2)]


def text_config(layer_types: list[str]) -> dict:
    return {
        "model_type": "qwen3_5_text",
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": len(layer_types),
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "vocab_size": 2048,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "full_attention_interval": 4,
        "layer_types": list(layer_types),
        "linear_num_value_heads": 2,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "attention_bias": False,
        "tie_word_embeddings": True,
        "attn_output_gate": True,
        "hidden_act": "silu",
        "dtype": "float32",
        "mamba_ssm_dtype": "float32",
        "eos_token_id": 3,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
    }


def vision_config() -> dict:
    return {
        "model_type": "qwen3_5_vision",
        "depth": 1,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_heads": 2,
        "out_hidden_size": 64,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "in_channels": 3,
        "num_position_embeddings": 16,
        "deepstack_visual_indexes": [],
        "hidden_act": "gelu_pytorch_tanh",
        "initializer_range": 0.02,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "axial"},
    }


def build_tokenizer(directory: Path) -> None:
    """A word-level tokenizer whose vocabulary holds every answer code as one token."""
    special = ["<unk>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    words = special + candidate_codes() + ["system", "user", "assistant"] + PROMPT_WORDS
    words += [w for w in "state option a b c first second third route left right x y z".split() if w not in words]
    vocab = {}
    for word in words:
        vocab.setdefault(word, len(vocab))
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(special[1:])
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
        "chat_template": CHAT_TEMPLATE,
        "model_max_length": 4096,
    }, indent=2), encoding="utf-8")
    (directory / "special_tokens_map.json").write_text(json.dumps({
        "unk_token": "<unk>", "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>",
        "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
    }), encoding="utf-8")


def retype_layers(net, layer_types: list[str], text_cfg) -> None:
    """The re-typing the community loader applies from ``layer_types`` (same effect, same names)."""
    lm = net.language_model.model
    for i, layer in enumerate(lm.layers):
        wants_attn = layer_types[i] == "full_attention"
        if wants_attn and not hasattr(layer, "self_attn"):
            layer.self_attn = L.Qwen3_5Attention(text_cfg)
            if hasattr(layer, "linear_attn"):
                del layer.linear_attn
        elif (not wants_attn) and not hasattr(layer, "linear_attn"):
            layer.linear_attn = L.Qwen3_5GatedDeltaNet(text_cfg)
            if hasattr(layer, "self_attn"):
                del layer.self_attn
    for i, layer in enumerate(lm.layers):
        layer.is_linear = layer_types[i] == "linear_attention"


def answer_codes(tok) -> tuple[list[str], list[int]]:
    """The official ``answer_codes`` rule (``d3_format.py``, copied by the community ``d3_mlx.py``): the first 255
    candidate codes that are one token and tokenize unchanged after the chat prefix, with their token ids."""
    prefix = tok.apply_chat_template([{"role": "user", "content": "Choose an option."}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    prefix_ids = tok.encode(prefix, add_special_tokens=False)
    codes: list[str] = []
    ids: list[int] = []
    for code in candidate_codes():
        enc = tok.encode(code, add_special_tokens=False)
        if len(enc) != 1 or enc[0] in ids:
            continue
        if tok.encode(prefix + code, add_special_tokens=False) != prefix_ids + enc:
            continue
        codes.append(code)
        ids.append(enc[0])
        if len(codes) == 255:
            break
    if len(codes) != 255:
        raise ValueError("the synthetic tokenizer does not provide 255 distinct single-token answer codes")
    return codes, ids


def write_decision_config(directory: Path, *, attention_mode: str = "noncausal_full_attention", temperature: float = 1.0,
                          **overrides) -> dict:
    """Write ``decision_config.json`` in the official ``d3-code-readout-v1`` shape and return it."""
    from transformers import AutoTokenizer

    codes, token_ids = answer_codes(AutoTokenizer.from_pretrained(str(directory)))
    config = {
        "format_version": 1,
        "format_id": "d3-code-readout-v1",
        "prompt": "d3",
        "base_model": "synthetic/qwen3_5-tiny",
        "revision": "0" * 40,
        "codes": codes,
        "token_ids": token_ids,
        "temperature": temperature,
        "attention_mode": attention_mode,
        "pooling": "last",
        "max_length": None,
        "readout_dtype": "float32",
    }
    config.update(overrides)
    (directory / "decision_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def build_checkpoint(directory: Path, *, layer_types: list[str] | None = None, seed: int = 7,
                     attention_mode: str = "noncausal_full_attention", zero_attention_output: bool = False,
                     decision_config: bool = True) -> Path:
    """Write a complete synthetic snapshot (without ``d3_mlx.py``) into ``directory`` and return it."""
    layer_types = list(layer_types or REGULAR_LAYER_TYPES)
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["Qwen3_5Model"],
        "model_type": "qwen3_5",
        "dtype": "float32",
        "tie_word_embeddings": True,
        "image_token_id": 2046,
        "video_token_id": 2047,
        "vision_start_token_id": 2044,
        "vision_end_token_id": 2045,
        "text_config": text_config(layer_types),
        "vision_config": vision_config(),
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    mx.random.seed(seed)
    model_config = q35.ModelConfig.from_dict(config)
    net = q35.Model(model_config)
    retype_layers(net, layer_types, model_config.text_config)
    weights = {}
    for name, value in tree_flatten(net.parameters()):
        weights[name] = (value + 0.02 * mx.random.normal(value.shape)).astype(mx.float32) if value.dtype != mx.float32 else value
        if zero_attention_output and name.endswith("self_attn.o_proj.weight"):
            weights[name] = mx.zeros(value.shape, dtype=mx.float32)
    mx.save_safetensors(str(directory / "weights.safetensors"), weights)
    hidden = config["text_config"]["hidden_size"]
    mx.save_safetensors(str(directory / "readout.safetensors"), {"weight": mx.random.normal((255, hidden))})
    build_tokenizer(directory)
    if decision_config:
        write_decision_config(directory, attention_mode=attention_mode)
    return directory
