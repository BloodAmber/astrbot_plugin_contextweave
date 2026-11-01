"""Adapt both native and proactive requests using this plugin's local log."""
from __future__ import annotations

import time

from astrbot.core.agent.message import TextPart

from .context_weaver import CURRENT_TAG, DELTA_TAG, POLICY, iso_time, pack, select_delta, strip_history_images
from .history_storage import HistoryStorage
from .message_utils import MessageUtils
from .image_context import attach_referenced_images
from .recent_images import attach_recent_images, message_parts_with_visuals
from .animation_context import remember_image_origins
from .engagement import interaction_metadata
from .image_focus import attach_focus_images
from .text_filter import TextFilter


def settings(config):
    cfg = config.get("context_weave", {})
    def number(name, default, lower, upper):
        try:
            return min(upper, max(lower, int(cfg.get(name, default))))
        except (TypeError, ValueError):
            return default
    return {"max_age": number("max_age_seconds", 900, 0, 86400),
            "max_chars": number("max_delta_chars", 16000, 2000, 64000),
            "max_message_chars": number("max_message_chars", 1200, 200, 8000)}


def reply_metadata(message):
    result = []
    for part in getattr(message, "message", []) or []:
        if part.__class__.__name__ == "Reply":
            result.append({"message_id": str(getattr(part, "id", "") or ""),
                           "sender_id": str(getattr(part, "sender_id", "") or ""),
                           "sender_name": str(getattr(part, "sender_nickname", "") or ""),
                           "sent_at": iso_time(getattr(part, "time", None))})
    return result


async def adapt_request(event, req, config):
    if event.get_extra("_contextweave_adapted", False) or not req.conversation:
        return
    if not HistoryStorage.is_chat_enabled(event):
        return
    # Another plugin may own a custom request. Restrict adaptation to requests
    # built here or by the native @/private path.
    if not event.get_extra("_contextweave_proactive", False) and event.get_extra("provider_request") is not req:
        # AstrBot shallow-copies plugin requests; native ones share conversation
        # but arrive as provider_request as well. Explicit wake is the safe fallback.
        if not getattr(event, "is_at_or_wake_command", False):
            return
    cfg = settings(config)
    current = event.message_obj
    current_id = str(getattr(current, "message_id", "") or "")
    timestamp = float(getattr(current, "timestamp", 0) or time.time())
    req.contexts, removed = strip_history_images(req.contexts)
    TextFilter.clean_assistant_history(req.contexts,config)
    # Flow has already run (higher priority). Replace ONLY its new trailing block,
    # retaining every previously saved historical block and its conversation.
    pending = event.get_extra("_group_context_flow_pending_cursor", {})
    if (event.get_extra("_group_context_flow_injected", False) and
            isinstance(pending, dict) and pending.get("injected_count", 0) > 0 and req.contexts):
        last = req.contexts[-1]
        value = last.get("content", "")
        if isinstance(value, str) and value.startswith("<group_messages_delta>\n") and value.endswith("\n</group_messages_delta>"):
            req.contexts.pop()
    private = event.is_private_chat()
    chat_id = event.get_sender_id() if private else event.get_group_id()
    history = HistoryStorage.get_history(event.get_platform_name(), private, chat_id)
    image_sources = await attach_referenced_images(event, req, history)
    image_focus, focus_reason = await attach_focus_images(event, req, history, image_sources)
    image_sources = list(dict.fromkeys(image_sources + [s['message_id'] for s in image_focus]))
    recent_images = attach_recent_images(event, req, history, config, image_sources)
    remember_image_origins(event, history, image_sources)
    records = []
    for message in history:
        mid = str(getattr(message, "message_id", "") or "")
        if mid == current_id:
            break  # Do not leak newer arrivals into an older in-flight request.
        sender = getattr(message, "sender", None)
        # The native conversation already stores our actual assistant replies.
        if str(getattr(sender, "user_id", "")) == str(event.get_self_id()):
            continue
        # No image-caption requests in this path: retain only cached captions.
        parts, visual_indexes = message_parts_with_visuals(event, message)
        text = await MessageUtils.outline_message_list(parts,
                                                      umo=event.unified_msg_origin, cached_images_only=True)
        if not text.strip() and not visual_indexes:
            continue
        records.append({"message_id": mid, "sender_id": str(getattr(sender, "user_id", "")),
                        "sender_name": str(getattr(sender, "nickname", "")),
                        "timestamp": float(getattr(message, "timestamp", 0) or 0),
                        "sent_at": iso_time(getattr(message, "timestamp", None)),
                        "reply_to": reply_metadata(message), "text": text})
        if visual_indexes:
            records[-1]['visual_component_indexes'] = visual_indexes
    try:
        cold_limit = min(200, max(1, int(config.get("group_msg_history", 20))))
    except (TypeError, ValueError):
        cold_limit = 20
    delta = select_delta(records, req.contexts, current_id, timestamp, cold_limit=cold_limit, **cfg)
    event.set_extra('_contextweave_timeline_records', records)
    if delta["messages"] or any(delta["omitted"].values()) or delta["coverage"] != "incremental":
        req.contexts.append({"role": "user", "content": pack(DELTA_TAG, delta)})
    metadata = {"message_id": current_id, "sender_id": str(event.get_sender_id()),
                "sender_name": event.get_sender_name(), "sent_at": iso_time(timestamp),
                "reply_to": reply_metadata(current),
                "mode": "proactive" if event.get_extra("_contextweave_proactive", False) else "direct"}
    metadata['image_sources'] = image_sources
    metadata['image_references'] = image_focus
    metadata['image_reference_reason'] = focus_reason
    metadata['image_reference_version'] = 2
    metadata['historical_image_references'] = [{k:v for k,v in r.items() if k!='path'}
        for r in event.get_extra('_contextweave_historical_images', [])]
    metadata['interaction'] = interaction_metadata(event)
    metadata['recent_images'] = recent_images
    metadata["observation"] = {
        "latest_previous_record_at": records[-1]["sent_at"] if records else None,
        "latest_previous_record_id": records[-1]["message_id"] if records else None,
        "newly_included_records": len(delta["messages"]),
        "older_background_records": delta["older_background_count"],
        "meaning": "仅描述当前会话可见的记录，不表示群里没有其他消息；零新增不等于无人发言",
    }
    req.extra_user_content_parts.append(TextPart(text=pack(CURRENT_TAG, metadata)))
    rules = config.get('context_weave', {}).get('system_rules', POLICY)
    if not isinstance(rules, str):
        rules = POLICY
    if rules.strip():
        req.system_prompt = (req.system_prompt or "") + "\n\n" + rules.strip()
    event.set_extra("_contextweave_adapted", True)
    event.set_extra("_contextweave_stats", {"messages": len(delta["messages"]), "omitted": delta["omitted"],
                                            "removed_history_images": removed, "mode": metadata["mode"]})
