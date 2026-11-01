"""Resolve only current, explicitly quoted or narrowly linked retry images."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote, urlparse

from astrbot import logger
from astrbot.api.all import Image, Plain, Reply
from astrbot.core.utils.quoted_message_parser import extract_quoted_message_images

from .context_weaver import CURRENT_TAG, texts


def local_path(ref):
    if not isinstance(ref, str) or not ref:
        return None
    if ref.startswith('file:'):
        parsed = urlparse(ref)
        ref = unquote(parsed.path)
        if parsed.netloc:
            ref = '//' + parsed.netloc + ref
        elif os.name == 'nt' and re.match(r'^/[A-Za-z]:', ref):
            ref = ref[1:]
    try:
        return str(Path(ref).resolve()) if Path(ref).is_file() else None
    except (OSError, ValueError):
        return None


async def component_images(parts):
    refs = []
    for part in parts or []:
        if not isinstance(part, Image):
            continue
        # The persisted file can outlive the component's preferred temp URL.
        ref = local_path(part.file) or local_path(part.url)
        if not ref:
            try:
                ref = await asyncio.wait_for(part.convert_to_file_path(), timeout=8)
            except Exception as exc:
                logger.warning('ContextWeave image unavailable: %s', type(exc).__name__)
                continue
            ref = local_path(ref)
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _metadata(message):
    for text in texts(message):
        match = re.search(r'<' + CURRENT_TAG + r'>\n(.*?)\n</' + CURRENT_TAG + '>', text, re.S)
        if match:
            try:
                value = json.loads(match.group(1))
                if isinstance(value, dict):
                    return value
            except ValueError:
                pass
    return None


async def attach_referenced_images(event, req, history):
    if event.get_extra('_contextweave_observation'):
        return []
    # History is already scoped by platform/group/private peer. Don't inspect
    # later arrivals while processing an older request.
    current_id = str(event.message_obj.message_id)
    now = float(event.message_obj.timestamp)
    by_id = {}
    for msg in history:
        if str(msg.message_id) == current_id:
            break
        if float(getattr(msg, 'timestamp', 0) or 0) <= now:
            by_id[str(msg.message_id)] = msg
    sources = []
    refs = []
    proactive = event.get_extra('_contextweave_proactive')
    native_collected = not proactive and (req.image_urls or any(
        '<image_caption>' in getattr(p, 'text', '') for p in req.extra_user_content_parts))
    direct = await component_images(event.message_obj.message) if proactive and not req.image_urls else []
    if direct or any(isinstance(p, Image) for p in event.message_obj.message):
        sources.append(current_id)
        refs.extend(direct)
    for reply in [p for p in event.message_obj.message if isinstance(p, Reply)]:
        mid = str(reply.id)
        cached = by_id.get(mid)
        if native_collected:
            if any(isinstance(p, Image) for p in (reply.chain or [])) or (
                    cached and any(isinstance(p, Image) for p in cached.message)):
                sources.append(mid)
            continue
        # Prefer our durable original rather than an expired quote temp file.
        images = await component_images(cached.message) if cached else []
        if not images:
            images = await component_images(reply.chain)
        # Native collection already does its own remote fallback. The plugin
        # request branch skips that collection, so fill only that gap here.
        if not images and proactive and cached is None:
            try:
                images = await asyncio.wait_for(extract_quoted_message_images(event, reply), timeout=8)
            except Exception as exc:
                logger.warning('ContextWeave quoted image lookup failed: %s', type(exc).__name__)
        if images:
            sources.append(mid)
            refs.extend(images)
    # Existing native attachments survive and aren't repeated under file URIs.
    seen = {local_path(ref) or ref for ref in req.image_urls}
    added = 0
    for ref in refs:
        key = local_path(ref) or ref
        if key not in seen:
            req.image_urls.append(ref)
            seen.add(key)
            added += 1
    event.set_extra('_contextweave_image_stats', {'added': added, 'source_count': len(set(sources))})
    return list(dict.fromkeys(sources))
