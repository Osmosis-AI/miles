"""Compatibility patches for NVIDIA Megatron Bridge.

Keep this module small: it is imported for side effects before Bridge weight
conversion paths are used.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _patch_qwen35_mtp_names() -> None:
    """Map Megatron Core's Qwen3.5 MTP names to Bridge's expected names.

    The Bridge Qwen3.5 registry currently uses
    ``language_model.mtp.layers.*.mtp_model_layer`` while the model built by
    Megatron Core exposes ``language_model.mtp.layers.*.transformer_layer``.
    Normalizing the lookup key lets Bridge reuse its existing MTP mappings for
    HF load/export and LoRA adapter export.
    """

    try:
        from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLBridge, Qwen35VLMoEBridge
    except Exception as exc:  # pragma: no cover - depends on optional Bridge install
        logger.debug("Skipping Qwen3.5 Megatron Bridge patch: %s", exc)
        return

    def _normalize_qwen35_name(self, megatron_param: str) -> str:
        del self
        name = megatron_param.replace(".to_wrap.", ".")
        if ".mtp.layers." in name:
            name = name.replace(".transformer_layer.", ".mtp_model_layer.")
        return name

    for bridge_cls in (Qwen35VLBridge, Qwen35VLMoEBridge):
        if getattr(bridge_cls, "_miles_qwen35_mtp_patch", False):
            continue
        bridge_cls._get_lora_unwrapped_name = _normalize_qwen35_name
        bridge_cls._miles_qwen35_mtp_patch = True


def _patch_peft_adapter_mtp_names() -> None:
    """Normalize Qwen3.5 MTP names when Bridge exports LoRA adapters."""

    try:
        from megatron.bridge.models.conversion.peft_bridge import MegatronPeftBridge
    except Exception as exc:  # pragma: no cover - depends on optional Bridge install
        logger.debug("Skipping PEFT adapter Megatron Bridge patch: %s", exc)
        return

    if getattr(MegatronPeftBridge, "_miles_qwen35_mtp_patch", False):
        return

    original_resolve = MegatronPeftBridge._resolve_hf_adapter_param_name
    original_base_names = MegatronPeftBridge._get_base_hf_param_names_for_adapter

    def _resolve_hf_adapter_param_name(
        self,
        mapping_registry,
        global_base_prefix: str,
        megatron_adapter_suffix: str,
        base_suffix: str,
        adapter_key,
    ):
        try:
            return original_resolve(
                self, mapping_registry, global_base_prefix, megatron_adapter_suffix, base_suffix, adapter_key
            )
        except AssertionError:
            base_name = f"{global_base_prefix}{base_suffix}"
            if ".mtp.layers." not in base_name or ".transformer_layer." not in base_name:
                raise
            return original_resolve(
                self,
                mapping_registry,
                global_base_prefix.replace(".transformer_layer.", ".mtp_model_layer."),
                megatron_adapter_suffix,
                base_suffix.replace(".transformer_layer.", ".mtp_model_layer."),
                adapter_key,
            )

    MegatronPeftBridge._resolve_hf_adapter_param_name = _resolve_hf_adapter_param_name

    def _get_base_hf_param_names_for_adapter(
        self,
        mapping_registry,
        global_base_prefix: str,
        adapter_key,
        base_suffix: str,
    ):
        result = original_base_names(self, mapping_registry, global_base_prefix, adapter_key, base_suffix)
        if result or ".mtp.layers." not in f"{global_base_prefix}{base_suffix}":
            return result
        return original_base_names(
            self,
            mapping_registry,
            global_base_prefix.replace(".transformer_layer.", ".mtp_model_layer."),
            adapter_key,
            base_suffix.replace(".transformer_layer.", ".mtp_model_layer."),
        )

    MegatronPeftBridge._get_base_hf_param_names_for_adapter = _get_base_hf_param_names_for_adapter
    MegatronPeftBridge._miles_qwen35_mtp_patch = True


_patch_qwen35_mtp_names()
_patch_peft_adapter_mtp_names()
