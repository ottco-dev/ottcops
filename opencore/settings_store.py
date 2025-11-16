"""Persistence and normalization for application-level settings."""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from .config import (
    DEFAULT_LLM_CONFIG,
    LLM_ALLOWED_PROVIDERS,
    MQTT_DEFAULT_CONFIG,
    MQTT_SENSOR_KINDS,
    SETTINGS_PATH,
)

logger = logging.getLogger("ottcouture.app")

_app_settings: Dict[str, Any] = {}


def normalize_llm_config(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Sanitize provider settings and merge them with defaults."""

    config = DEFAULT_LLM_CONFIG.copy()
    if isinstance(payload, dict):
        for key in config:
            if key not in payload:
                continue
            value = payload[key]
            if value is None:
                continue
            if isinstance(value, str):
                value = value.strip()
            config[key] = value

    provider = str(config.get("provider") or DEFAULT_LLM_CONFIG["provider"]).lower()
    if provider not in LLM_ALLOWED_PROVIDERS:
        provider = DEFAULT_LLM_CONFIG["provider"]
    config["provider"] = provider

    vision_raw = str(config.get("vision") or DEFAULT_LLM_CONFIG["vision"]).lower()
    config["vision"] = "manual" if vision_raw == "manual" else DEFAULT_LLM_CONFIG["vision"]

    for text_key in ("apiBase", "model", "apiKey", "systemPrompt"):
        config[text_key] = config.get(text_key) or ""

    return config


def normalize_llm_profile(entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Ensure a profile dict always contains id, name and normalized config."""

    if not isinstance(entry, dict):
        return None
    config = normalize_llm_config(entry.get("config"))
    profile_id = str(entry.get("id") or entry.get("profile_id") or uuid.uuid4().hex)
    name = (entry.get("name") or f"Profile {profile_id[:6]}").strip()
    return {"id": profile_id, "name": name or profile_id, "config": config}


def normalize_llm_profiles(raw: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    profiles: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for entry in raw:
            normalized = normalize_llm_profile(entry)
            if normalized:
                profiles.append(normalized)
    return profiles


def _normalize_sensor_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in (value or "").lower())
    return cleaned.strip("-") or f"sensor-{uuid.uuid4().hex[:6]}"


def normalize_mqtt_sensor(entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict):
        return None
    label = (entry.get("label") or entry.get("name") or "MQTT Sensor").strip()
    topic = (entry.get("topic") or "").strip()
    if not topic:
        return None
    kind_raw = str(entry.get("kind") or entry.get("type") or "custom").lower()
    kind = kind_raw if kind_raw in MQTT_SENSOR_KINDS else "custom"
    sensor_id = entry.get("id") or _normalize_sensor_id(label or topic)
    return {
        "id": sensor_id,
        "label": label or sensor_id,
        "topic": topic,
        "kind": kind,
        "unit": (entry.get("unit") or "").strip(),
    }


def normalize_mqtt_config(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    config = dict(MQTT_DEFAULT_CONFIG)
    if isinstance(payload, dict):
        broker = (payload.get("broker") or "").strip()
        if broker:
            config["broker"] = broker
        try:
            port = int(payload.get("port") or config["port"])
            config["port"] = port
        except (TypeError, ValueError):
            pass
        for key in ("username", "password"):
            if payload.get(key) is not None:
                config[key] = str(payload.get(key))
        if payload.get("use_tls") is not None:
            config["use_tls"] = bool(payload.get("use_tls"))
        sensors: List[Dict[str, Any]] = []
        if isinstance(payload.get("sensors"), list):
            for entry in payload["sensors"]:
                normalized = normalize_mqtt_sensor(entry)
                if normalized:
                    sensors.append(normalized)
        config["sensors"] = sensors
    return config


def load_app_settings() -> Dict[str, Any]:
    defaults = {
        "default_model_id": None,
        "llm_config": DEFAULT_LLM_CONFIG.copy(),
        "llm_profiles": [],
        "active_llm_profile": None,
        "mqtt": MQTT_DEFAULT_CONFIG.copy(),
    }
    if SETTINGS_PATH.is_file():
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            defaults.update({k: v for k, v in data.items() if k != "llm_config"})
            defaults["llm_config"] = normalize_llm_config(data.get("llm_config"))
            defaults["llm_profiles"] = normalize_llm_profiles(data.get("llm_profiles"))
            defaults["active_llm_profile"] = data.get("active_llm_profile")
            defaults["mqtt"] = normalize_mqtt_config(data.get("mqtt"))
            if defaults["active_llm_profile"] and not any(
                p.get("id") == defaults["active_llm_profile"] for p in defaults["llm_profiles"]
            ):
                defaults["active_llm_profile"] = None
        except json.JSONDecodeError:
            logger.warning("app-settings.json could not be parsed, falling back to defaults.")
    return defaults


def save_app_settings(data: Dict[str, Any]) -> None:
    SETTINGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def bootstrap_app_settings() -> Dict[str, Any]:
    global _app_settings
    _app_settings = load_app_settings()
    return _app_settings


def get_app_settings() -> Dict[str, Any]:
    return _app_settings


def get_llm_profiles() -> List[Dict[str, Any]]:
    profiles = normalize_llm_profiles(_app_settings.get("llm_profiles") or [])
    _app_settings["llm_profiles"] = profiles
    return profiles


def get_active_llm_profile_id() -> Optional[str]:
    return _app_settings.get("active_llm_profile")


def get_llm_config(profile_id: Optional[str] = None) -> Dict[str, Any]:
    profiles = get_llm_profiles()
    target_id = profile_id or get_active_llm_profile_id()
    if target_id:
        for profile in profiles:
            if profile.get("id") == target_id:
                return normalize_llm_config(profile.get("config"))
    return normalize_llm_config(_app_settings.get("llm_config"))


def set_active_llm_profile(profile_id: Optional[str]) -> Optional[str]:
    profiles = get_llm_profiles()
    if profile_id is None:
        _app_settings["active_llm_profile"] = None
    elif any(p.get("id") == profile_id for p in profiles):
        _app_settings["active_llm_profile"] = profile_id
    else:
        raise HTTPException(status_code=404, detail="Profile not found.")
    save_app_settings(_app_settings)
    return _app_settings.get("active_llm_profile")


def upsert_llm_profile(
    config: Dict[str, Any], name: Optional[str] = None, profile_id: Optional[str] = None, activate: bool = True
) -> Dict[str, Any]:
    profiles = get_llm_profiles()
    normalized = normalize_llm_profile({"config": config, "id": profile_id, "name": name})
    if normalized is None:
        raise HTTPException(status_code=400, detail="Profile could not be normalized.")
    updated = False
    for idx, profile in enumerate(profiles):
        if profile.get("id") == normalized["id"]:
            profiles[idx] = normalized
            updated = True
            break
    if not updated:
        profiles.append(normalized)
    _app_settings["llm_profiles"] = profiles
    if activate:
        _app_settings["active_llm_profile"] = normalized["id"]
    save_app_settings(_app_settings)
    return normalized


def delete_llm_profile(profile_id: str) -> List[Dict[str, Any]]:
    profiles = get_llm_profiles()
    filtered = [p for p in profiles if p.get("id") != profile_id]
    if len(filtered) == len(profiles):
        raise HTTPException(status_code=404, detail="Profile not found.")
    _app_settings["llm_profiles"] = filtered
    if _app_settings.get("active_llm_profile") == profile_id:
        _app_settings["active_llm_profile"] = None
    save_app_settings(_app_settings)
    return filtered


def persist_llm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_llm_config(config)
    _app_settings["llm_config"] = normalized
    save_app_settings(_app_settings)
    return normalized


def reset_llm_config() -> Dict[str, Any]:
    return persist_llm_config(DEFAULT_LLM_CONFIG.copy())


def build_llm_settings_payload(message: Optional[str] = None) -> Dict[str, Any]:
    payload = {
        "config": get_llm_config(),
        "profiles": get_llm_profiles(),
        "active_profile_id": get_active_llm_profile_id(),
    }
    if message:
        payload["message"] = message
    return payload


def get_default_model_id() -> Optional[str]:
    return _app_settings.get("default_model_id")


def set_default_model_id(model_id: Optional[str]) -> Optional[str]:
    _app_settings["default_model_id"] = model_id
    save_app_settings(_app_settings)
    return model_id


def get_mqtt_config() -> Dict[str, Any]:
    config = normalize_mqtt_config(_app_settings.get("mqtt"))
    _app_settings["mqtt"] = config
    return config


def persist_mqtt_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_mqtt_config(payload)
    _app_settings["mqtt"] = normalized
    save_app_settings(_app_settings)
    return normalized
