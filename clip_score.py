"""CLIP ViT patch–text margin maps + residual r = margin(render) − margin(GT) for JVP pruning.

Uses **OpenAI CLIP** (``pip install git+https://github.com/openai/CLIP.git``): ViT vision encoder yields a grid of
patch embeddings in the **same joint space** as ``encode_text``, so per-patch cosine similarity to pooled text
prototypes is well-defined (contrast ResNet CLIP, which has no single per-pixel joint embedding without extra hooks).

Default model **ViT-B/32** is the smallest ViT variant in the official CLIP release (fewer patch tokens than ViT-B/16).
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence

import clip
import clip.model as clip_model
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from torchvision.transforms import InterpolationMode

SCENE_PRESETS = {'train': {
    'main_prompts': ('train', 'locomotive', 'railway train'),
    'background_prompts': ('railroad track', 'railway tracks', 'cobblestone',
        'cobblestones', 'dirt ground', 'dirt road', 'tree', 'trees', 'sky',
        'mountain', 'mountains', 'shipping container', 'container', 'car',
        'cars', 'grass', 'grass field', 'meadow', 'field'),
}}

_CLIP_PROMPTS_SHORT = (
    "a photo of a {0}.",
    "a photo of the {0}.",
    "a rendering of a {0}.",
    "a cropped photo of the {0}.",
    "a dark photo of the {0}.",
    "a bright photo of a {0}.",
    "a black and white photo of the {0}.",
    "a jpeg corrupted photo of a {0}.",
)

CLIP_MARGIN_DEFAULT_MODEL = "ViT-B/32"

_ACTIVE_CLIP_CTX: CLIPMarginContext | None = None


@dataclass
class CLIPMarginContext:
    model: torch.nn.Module
    input_resolution: int
    main_emb: torch.Tensor
    bg_embs: torch.Tensor
    use_amp: bool = False


def get_active_clip_margin_context() -> CLIPMarginContext:
    if _ACTIVE_CLIP_CTX is None:
        raise RuntimeError(
            "clip_margin phi requires an active CLIPMarginContext; "
            "use build_clip_context_for_scene() and push_clip_margin_context() "
            "before calling JVP with phi_mode='clip_margin'."
        )
    return _ACTIVE_CLIP_CTX


@contextmanager
def push_clip_margin_context(ctx: CLIPMarginContext):
    global _ACTIVE_CLIP_CTX
    prev = _ACTIVE_CLIP_CTX
    _ACTIVE_CLIP_CTX = ctx
    try:
        yield
    finally:
        _ACTIVE_CLIP_CTX = prev


def load_clip_model(
    device: torch.device | str,
    model_name: str = CLIP_MARGIN_DEFAULT_MODEL,
) -> torch.nn.Module:
    name = str(model_name).strip()
    model, _pre = clip.load(name, device=str(device), jit=False)
    if not isinstance(model.visual, clip_model.VisionTransformer):
        raise RuntimeError(
            "clip_margin requires a ViT CLIP model (e.g. ViT-B/32, ViT-B/16); "
            "{!r} uses {!r}. ResNet CLIP has no supported per-patch joint embedding here.".format(
                name,
                type(model.visual).__name__,
            )
        )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def encode_class_embedding_clip(
    model: torch.nn.Module,
    class_name: str,
    templates: Sequence[str],
    device: torch.device,
    *,
    use_amp: bool = False,
) -> torch.Tensor:
    texts = [t.format(class_name) for t in templates]
    tokens = clip.tokenize(texts).to(device)
    amp_on = bool(use_amp) and device.type == "cuda"
    cm = torch.autocast(device.type, dtype=torch.float16, enabled=amp_on)
    with cm:
        feats = model.encode_text(tokens)
    feats = F.normalize(feats.float(), p=2, dim=-1)
    emb = feats.mean(dim=0)
    return F.normalize(emb, p=2, dim=-1)


def build_clip_context_for_scene(
    scene: str,
    *,
    device: torch.device | str = "cuda",
    model_name: str = CLIP_MARGIN_DEFAULT_MODEL,
    templates: Sequence[str] | None = None,
    use_amp: bool = False,
) -> CLIPMarginContext:
    if scene not in SCENE_PRESETS:
        raise ValueError(f"Unknown scene {scene!r}; use one of {list(SCENE_PRESETS.keys())}")
    preset = SCENE_PRESETS[scene]
    tpl = tuple(templates) if templates is not None else _CLIP_PROMPTS_SHORT
    dev = torch.device(device)
    model = load_clip_model(dev, model_name=model_name)
    res = int(model.visual.input_resolution)
    amp_e = bool(use_amp) and dev.type == "cuda"
    mains = torch.stack(
        [encode_class_embedding_clip(model, m, tpl, dev, use_amp=amp_e) for m in preset["main_prompts"]]
    )
    main_emb = F.normalize(mains.mean(dim=0), p=2, dim=-1)
    bg = torch.stack(
        [encode_class_embedding_clip(model, b, tpl, dev, use_amp=amp_e) for b in preset["background_prompts"]]
    )
    return CLIPMarginContext(
        model=model,
        input_resolution=res,
        main_emb=main_emb,
        bg_embs=bg,
        use_amp=amp_e,
    )


def _clip_preprocess_bchw(image_chw: torch.Tensor, n_px: int) -> torch.Tensor:
    """``image_chw`` in ``[0,1]`` → CLIP-normalized ``[1,3,n_px,n_px]``."""
    x = image_chw.unsqueeze(0).clamp(0.0, 1.0)
    x = TVF.resize(x, [n_px, n_px], interpolation=InterpolationMode.BICUBIC, antialias=True)
    dtype = x.dtype
    device = x.device
    mean = torch.tensor([0.48145466, 0.45821073, 0.40821073], device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device, dtype=dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def _vit_patch_grid_embeddings(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Joint embedding map ``[B, D, Hp, Wp]`` (ViT patches, excludes CLS)."""
    vt = model.visual
    assert isinstance(vt, clip_model.VisionTransformer)
    x = vt.conv1(x.type(vt.conv1.weight.dtype))
    b, w, hx, wx = x.shape
    x = x.reshape(b, w, -1).permute(0, 2, 1)
    cls = vt.class_embedding.to(x.dtype) + torch.zeros(
        b, 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls, x], dim=1)
    x = x + vt.positional_embedding.to(x.dtype)
    x = vt.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = vt.transformer(x)
    x = x.permute(1, 0, 2)
    x = vt.ln_post(x)
    x = x @ vt.proj
    patches = x[:, 1:, :]
    gs = int(round(math.sqrt(float(patches.shape[1]))))
    if gs * gs != patches.shape[1]:
        raise RuntimeError(
            "CLIP ViT patch grid not square: N={} (expected square grid)".format(patches.shape[1])
        )
    x = patches.reshape(b, gs, gs, -1).permute(0, 3, 1, 2).contiguous()
    return F.normalize(x.float(), p=2, dim=1, eps=1e-8)


def clip_margin_map(
    image_chw: torch.Tensor,
    ctx: CLIPMarginContext | None = None,
) -> torch.Tensor:
    """Per-patch margin: cos(P, t_main) − max_k cos(P, t_bg). Shape ``[1, Hp, Wp]``."""
    c = ctx or default_train_context(str(image_chw.device))
    n_px = int(c.input_resolution)
    x = _clip_preprocess_bchw(image_chw, n_px)
    amp_on = bool(c.use_amp) and x.device.type == "cuda"
    cm = torch.autocast(x.device.type, dtype=torch.float16, enabled=amp_on)
    with cm:
        feats = _vit_patch_grid_embeddings(c.model, x)
    _, _, hp, wp = feats.shape
    pf = feats.flatten(2).transpose(1, 2).squeeze(0)
    s_main = (pf * c.main_emb.unsqueeze(0)).sum(dim=-1)
    s_bg = torch.matmul(pf, c.bg_embs.T)
    max_bg = s_bg.max(dim=1).values
    margin = (s_main - max_bg).reshape(1, hp, wp)
    return margin


def clip_margin_residual_maps(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """r(θ) at patch resolution: margin(render) − margin(GT). ``image`` / ``target`` are ``[3,H,W]``."""
    ctx = get_active_clip_margin_context()
    m_r = clip_margin_map(image, ctx)
    with torch.no_grad():
        m_g = clip_margin_map(target, ctx)
    return m_r - m_g


from functools import lru_cache

@lru_cache(maxsize=2)
def default_train_context(device):
    return build_clip_context_for_scene('train', device=device)
