"""
chunked_vision_encoder.py — Plan 4: Chunked-Vision-Encoding
=============================================================

Löst Multimodal+OOM durch Vision-Encoding in Chunks (1 Bild pro forward-pass).

Architektur:
  Statt processor(text=..., images=[img1, img2, img3]) der alle 3 Bilder in
  einem vision_tower forward encodiert (peak 11.05GB → OOM bei 12GB GPU),
  iterieren wir über Bilder einzeln:
    for img in images:
        projected = multi_modal_projector(vision_tower(pixel_values=[1,3,896,896]))
        all_projected.append(projected)
        del pixel_values, vision_out
        torch.cuda.empty_cache()
        gc.collect()
    full_projected = torch.cat(all_projected, dim=0)  # [N*256, hidden]

  Dann: tokenize den text separat (tokenizer-only path), merge via
  masked_scatter wo die <image_soft_token>-Positionen im input_ids sind.

Trade-off:
  - 3× langsameres Vision-Encoding (3× 2.07s = 6.2s statt 1× 6.2s)
  - ABER: peak VRAM drastisch reduziert (7.03GB statt 11.05GB)
  - → Pre-fill kann in headroom laufen (use_cache=True möglich)

Public API:
    chunked_encode_vision(model, images) -> Tensor [N*256, hidden_size]
    chunked_process_multimodal(model, processor, tokenizer, messages) ->
        dict mit input_ids, inputs_embeds, attention_mask, etc.
"""
from __future__ import annotations

import gc
import os
from typing import List

import torch


def chunked_encode_vision(
    model,
    images: List,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Encode N images one-at-a-time to avoid OOM in vision_tower.

    Args:
        model: Gemma3ForConditionalGeneration (after int8 quantize + PX patch)
        images: list of PIL.Image (RGB)
        device: cuda device
        dtype: target dtype (bf16 — same as model)

    Returns:
        projected: Tensor [N*256, hidden_size] — all images' projected features
                   concatenated along the token dim.
                   Bei 3 Bildern: shape=[768, 2560], dtype=bf16, ~3.9 MB.
    """
    vision_tower = model.model.vision_tower
    mm_projector = model.model.multi_modal_projector
    image_processor = model.model.vision_tower.image_processor if hasattr(model.model.vision_tower, 'image_processor') else None
    # NOTE: wir nutzen den AutoProcessor für die Bild-Vorverarbeitung (resize,
    # normalize etc.) und nur den vision_tower für das Encoding. Das ist
    # sauberer als processor.image_processor + vision_tower getrennt.
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
    image_processor = processor.image_processor

    all_projected = []
    for i, img in enumerate(images):
        # 1. PIL → pixel_values [1, 3, 896, 896] float32
        pixel_values = image_processor([img], return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(device, dtype=dtype)

        # 2. vision_tower forward
        with torch.inference_mode():
            vision_out = vision_tower(pixel_values=pixel_values, return_dict=True)

        # 3. multi_modal_projector: [1, 4096, 1152] → [1, 256, 2560]
        with torch.inference_mode():
            projected = mm_projector(vision_out.last_hidden_state)
        # squeeze batch dim → [256, 2560]
        projected = projected.squeeze(0)

        all_projected.append(projected.cpu())  # auf CPU zwischen Bildern!

        # 4. Cleanup GPU memory
        del pixel_values, vision_out, projected
        torch.cuda.empty_cache()
        gc.collect()

    # 5. Cat all + move to GPU once
    full_projected = torch.cat(all_projected, dim=0).to(device, dtype=dtype)
    return full_projected


def merge_inputs_embeds(
    model,
    input_ids: torch.Tensor,
    projected_features: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Merge text-embeddings with image-features at the image_soft_token positions.

    Gemma3 chat template inserts <image_soft_token> placeholders for each image.
    Each image has 256 such placeholders. We replace these placeholders' embeddings
    with the projected vision features.

    Args:
        model: HF model
        input_ids: [B, T] token IDs
        projected_features: [N*256, hidden] vision features
        image_token_id: token ID for <image_soft_token>

    Returns:
        inputs_embeds: [B, T, hidden] — text embeddings with image positions replaced
    """
    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids).clone()  # [B, T, hidden] — clone damit in-place safe

    B, T, hidden = inputs_embeds.shape
    assert B == 1, "Only batch=1 supported for chunked vision encoder"

    # Find image_token positions in the input_ids sequence
    image_mask = (input_ids[0] == image_token_id)  # [T] bool
    n_image_tokens = image_mask.sum().item()
    assert n_image_tokens == projected_features.shape[0], \
        f"Image token count {n_image_tokens} != projected features {projected_features.shape[0]}"

    # Get indices of image positions, in order
    image_indices = image_mask.nonzero(as_tuple=True)[0]  # [n_image_tokens]

    # Assign projected_features to those positions (in-place)
    inputs_embeds[0, image_indices] = projected_features.to(inputs_embeds.dtype)

    return inputs_embeds


def chunked_process_multimodal(
    model,
    processor,
    tokenizer,
    messages: list,
    images: list,
    device: str = "cuda",
):
    """Process multimodal input with chunked vision encoding.

    Args:
        model: Gemma3ForConditionalGeneration
        processor: AutoProcessor (für chat-template)
        tokenizer: AutoTokenizer (für tokenize)
        messages: list of {role, content} — content may be multipart mit image-blocks
        images: list of PIL.Image, in order
        device: cuda device

    Returns:
        dict with 'input_ids', 'inputs_embeds', 'attention_mask' — ready for model.forward()
    """
    # NOTE: Wir brauchen die image-blocks IM Content (damit das chat template
    # pro Bild einen <start_of_image> einfügt), aber OHNE data-URL.
    # Wir nehmen die messages 1:1 (Aufrufer muss die image-blocks passend
    # gebaut haben — entweder mit PIL-data-URL via _extract_images, oder
    # einfach {"type":"image"} stubs für manuelles Bauen).

    # 1. Apply chat template (text mit <start_of_image> Stubs)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 2. Expand boi_token zu full_image_sequence (manuell — der processor macht
    #    das intern, aber wir wollen NICHT den ganzen processor laufen lassen
    #    weil der alle Bilder auf einmal encodet → OOM).
    n_images = len(images)
    boi_token = tokenizer.boi_token  # '<start_of_image>'
    image_token = tokenizer.image_token  # '<image_soft_token>'
    eoi_token = tokenizer.eoi_token  # '<end_of_image>'
    image_seq_length = getattr(processor, "image_seq_length", 256)
    full_image_sequence = f"\n\n{boi_token}{image_token * image_seq_length}{eoi_token}\n\n"
    # replace ALL boi-tokens (eines pro Bild) — str.replace macht das automatisch
    expanded_text = prompt_text.replace(boi_token, full_image_sequence)
    n_boi_in_prompt = prompt_text.count(boi_token)
    n_image_in_expanded = expanded_text.count(image_token)
    assert n_image_in_expanded == n_boi_in_prompt * image_seq_length, \
        f"expansion mismatch: prompt had {n_boi_in_prompt} boi, expanded has {n_image_in_expanded} image_soft (expected {n_boi_in_prompt * image_seq_length})"
    assert n_boi_in_prompt == n_images, \
        f"image count mismatch: prompt had {n_boi_in_prompt} boi, len(images)={n_images}"

    # 3. Encode all images (chunked, 1 per forward)
    projected = chunked_encode_vision(model, images, device=device)
    # projected shape: [n_images * 256, hidden_size]

    # 4. Tokenize the expanded text (jetzt mit 256*N image_soft_token Stellen)
    inputs = tokenizer(expanded_text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)

    # 5. Find image_soft_token_id
    image_token_id = tokenizer.image_token_id  # 262144

    # 6. Merge text-embeddings with image-features
    inputs_embeds = merge_inputs_embeds(model, input_ids, projected, image_token_id)

    # 7. Build attention_mask (all 1s for prompt)
    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
    }
