"""Bounded recent visual background, always scoped to the current chat."""
from __future__ import annotations

from astrbot.api.all import Image
from astrbot.core.agent.message import ImageURLPart, TextPart

from .context_weaver import iso_time, pack
from .image_context import local_path


def recent_image_count(config):
    try:
        return min(20, max(0, int(config.get('image_processing', {}).get('recent_image_count', 2))))
    except (TypeError, ValueError, OverflowError):
        return 2


def select_recent_images(event, history, count):
    if count <= 0:
        return []
    current_id = str(event.message_obj.message_id)
    now = float(event.message_obj.timestamp)
    eligible = []
    for message in history:
        if float(getattr(message, 'timestamp', 0) or 0) <= now:
            eligible.append(message)
        if str(message.message_id) == current_id:
            break
    # Direct events may not have reached the local log (e.g. persistence failure).
    if not any(str(m.message_id) == current_id for m in eligible):
        eligible.append(event.message_obj)
    selected, seen = [], set()
    for message in reversed(eligible):
        sender = getattr(message, 'sender', None)
        if str(getattr(sender, 'user_id', '')) == str(event.get_self_id()):
            continue
        for index in range(len(message.message or []) - 1, -1, -1):
            part = message.message[index]
            if not isinstance(part, Image):
                continue
            # Background images must already be local: never perform a series
            # of new downloads or caption calls just to fill the recent window.
            path = local_path(part.file) or local_path(part.url)
            if not path or path in seen:
                continue
            seen.add(path)
            selected.append({'message_id': str(message.message_id),
                             'sender_id': str(getattr(sender, 'user_id', '')),
                             'sender_name': str(getattr(sender, 'nickname', '')),
                             'sent_at': iso_time(getattr(message, 'timestamp', None)),
                             'component_index': index, 'path': path})
            if len(selected) >= count:
                return list(reversed(selected))
    return list(reversed(selected))


def attach_recent_images(event, req, history, config, explicit_sources):
    selected = select_recent_images(event, history, recent_image_count(config))
    existing = {local_path(ref) or ref for ref in req.image_urls}
    existing.update(local_path(p.image_url.url) or p.image_url.url for p in req.extra_user_content_parts if isinstance(p,ImageURLPart))
    attached, metadata = [], []
    for image in selected:
        record = {key: value for key, value in image.items() if key != 'path'}
        # Current and explicit quote images have already been prepared by the
        # native pipeline, so their resulting file names may differ from storage.
        explicit = image['message_id'] in explicit_sources and bool(req.image_urls)
        if explicit or image['path'] in existing:
            record['input'] = 'current_or_explicit_attachment'
        else:
            record['input'] = 'recent_visual_background'
            label = {**record, 'meaning': '以下图片属于此条历史消息，仅作背景，不代表有人刚发图或在向你提问'}
            req.extra_user_content_parts.extend([
                TextPart(text=pack('contextweave_recent_image', label)).mark_as_temp(),
                ImageURLPart(image_url=ImageURLPart.ImageURL(url=image['path'])).mark_as_temp(),
            ])
            attached.append(image['path'])
            existing.add(image['path'])
        metadata.append(record)
    event.set_extra('_contextweave_recent_images', selected)
    event.set_extra('_contextweave_recent_image_refs', attached)
    event.set_extra('_contextweave_recent_image_stats', {'selected': len(selected), 'added': len(attached)})
    return metadata


def message_parts_with_visuals(event, message):
    """Selected images have actual visual parts, so don't describe them as absent."""
    selected = {r['component_index'] for r in event.get_extra('_contextweave_recent_images', [])
                if r['message_id'] == str(message.message_id)}
    parts = [part for index, part in enumerate(message.message or []) if index not in selected]
    return parts, sorted(selected)
