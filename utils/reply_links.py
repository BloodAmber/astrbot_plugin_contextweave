"""Record successful OneBot reply receipts, without replacing native sending."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import uuid

from astrbot import logger
from .history_storage import HistoryStorage


class ReplyLinks:
    _locks = {}

    @staticmethod
    def _path(event):
        scope = f'{event.get_platform_id()}|{event.unified_msg_origin}|{event.get_self_id()}'
        key = hashlib.sha256(scope.encode()).hexdigest()
        base = HistoryStorage.base_storage_path
        if not base:
            return None
        return Path(base)/'reply_links'/(key+'.json')

    @classmethod
    def read(cls, event, conversation_id):
        path = cls._path(event)
        if path is None or not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            return {str(r['reply_id']):r for r in data if r.get('conversation_id') == str(conversation_id)}
        except (OSError, ValueError, TypeError, KeyError):
            return {}

    @classmethod
    async def record(cls, event, conversation_id, reply_id):
        path = cls._path(event)
        if path is None:
            return
        lock = cls._locks.setdefault(str(path), asyncio.Lock())
        async with lock:
            try:
                data = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else []
                if not isinstance(data, list):
                    data = []
            except (OSError, ValueError):
                data = []
            data = [r for r in data if isinstance(r,dict) and str(r.get('reply_id')) != str(reply_id)]
            data.append({'reply_id':str(reply_id), 'source_message_id':str(event.message_obj.message_id),
                         'conversation_id':str(conversation_id)})
            path.parent.mkdir(parents=True,exist_ok=True)
            temp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
            try:
                temp.write_text(json.dumps(data[-256:],ensure_ascii=False),encoding='utf-8')
                os.replace(temp,path)
            finally:
                if temp.exists():
                    temp.unlink()

    @classmethod
    def clear(cls,event):
        path=cls._path(event)
        if path and path.is_file():
            try:
                path.unlink()
            except OSError as exc:
                logger.warning('ContextWeave reply receipts not cleared: %s',type(exc).__name__)


class _ReceiptBot:
    def __init__(self, original, callback, event):
        self.original, self.callback, self.event = original, callback, event

    def __getattr__(self,name):
        return getattr(self.original,name)

    async def _send(self, method, args, kwargs, destination):
        result = await getattr(self.original,method)(*args,**kwargs)
        expected = self.event.get_group_id() if destination=='group_id' else self.event.get_sender_id()
        if str(kwargs.get(destination,'')) == str(expected):
            if isinstance(result,dict) and (result.get('status') not in {None,'ok'} or result.get('retcode',0)!=0):
                return result
            value = result.get('data',result) if isinstance(result,dict) else None
            mid = value.get('message_id') if isinstance(value,dict) else None
            if mid is not None and str(mid).lstrip('-').isdigit() and str(mid)!='0':
                try:
                    await self.callback(str(mid))
                except Exception as exc:
                    # Successful delivery must stay successful if bookkeeping fails.
                    logger.warning('ContextWeave reply receipt not saved: %s',type(exc).__name__)
        return result

    async def send_group_msg(self,*args,**kwargs):
        return await self._send('send_group_msg',args,kwargs,'group_id')

    async def send_private_msg(self,*args,**kwargs):
        return await self._send('send_private_msg',args,kwargs,'user_id')


def install_receipt_capture(event, req):
    if (event.get_platform_name()!='aiocqhttp' or not req.conversation
            or not getattr(event,'bot',None) or event.get_extra('_contextweave_receipt_original_send')):
        return
    original_send = event.send
    conversation_id = req.conversation.cid
    lock = asyncio.Lock()

    async def send_with_receipt(message):
        async with lock:
            original_bot = event.bot
            async def capture(mid):
                await ReplyLinks.record(event,conversation_id,mid)
            proxy = _ReceiptBot(original_bot,capture,event)
            event.bot = proxy
            try:
                return await original_send(message)
            finally:
                if event.bot is proxy:
                    event.bot = original_bot

    event.set_extra('_contextweave_receipt_original_send',original_send)
    event.send = send_with_receipt


def restore_receipt_capture(event):
    original = event.get_extra('_contextweave_receipt_original_send')
    if original:
        event.send = original
        event.set_extra('_contextweave_receipt_original_send',None)
