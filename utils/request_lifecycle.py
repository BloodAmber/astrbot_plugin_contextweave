"""Coordinate native and proactive requests without owning AstrBot's session lock."""
import asyncio
import copy
import json

from astrbot.api import logger


class ReplyTasks:
    """Track the whole event task, including history save and message delivery.

    Agent-done is too early: AstrBot persists the answer afterwards. Task completion
    also covers exceptions, cancellation, provider failures and silent responses.
    """

    def __init__(self):
        self.tasks = {}

    def busy(self, origin):
        current = asyncio.current_task()
        return any(task is not current and not task.done()
                   for task in self.tasks.get(origin, ()))

    def claim(self, origin):
        task = asyncio.current_task()
        if task is None:
            return
        tasks = self.tasks.setdefault(origin, set())
        if task in tasks:
            return
        tasks.add(task)

        def finished(done):
            tasks.discard(done)
            if not tasks and self.tasks.get(origin) is tasks:
                self.tasks.pop(origin, None)

        task.add_done_callback(finished)


async def refresh_owned_request(event, req, context):
    """Refresh our prebuilt request AFTER the native session lock is acquired.

    Only replace the original history segment. Keep persona begin-dialogs, file
    extraction, attachments, quoted text and current-turn instructions intact.
    This hook must precede other plugins' on_llm_request mutations.
    """
    snapshot = event.get_extra('_contextweave_history_snapshot')
    if not snapshot or event.get_extra('_contextweave_history_refreshed'):
        return
    if not req.conversation or req.conversation.cid != snapshot['cid']:
        return
    manager = context.conversation_manager
    cid = await manager.get_curr_conversation_id(event.unified_msg_origin)
    if cid != snapshot['cid']:
        # A queued request must not resurrect a reset/deleted conversation.
        event.stop_event()
        logger.info('ContextWeave: skipped queued request after conversation switch')
        return
    latest = await manager.get_conversation(event.unified_msg_origin, cid)
    if latest is None:
        event.stop_event()
        return
    old = json.loads(snapshot['history'])
    fresh = json.loads(latest.history)
    if old != fresh:
        contexts = req.contexts
        if not isinstance(contexts, list):
            event.stop_event()
            return
        if old:
            offsets = [i for i in range(len(contexts) - len(old) + 1)
                       if contexts[i:i + len(old)] == old]
            if len(offsets) != 1:
                # Never overwrite a context another request builder has changed.
                event.stop_event()
                logger.warning('ContextWeave: skipped stale request with unrecognized history layout')
                return
            offset = offsets[0]
        else:
            # Native persona examples precede history; extracted file results follow it.
            offset = next((i for i, msg in enumerate(contexts)
                           if msg.get('role') == 'system'
                           and str(msg.get('content', '')).startswith('File Extract Results of user uploaded files:')),
                          len(contexts))
        req.contexts = contexts[:offset] + copy.deepcopy(fresh) + contexts[offset + len(old):]
        logger.info('ContextWeave: refreshed queued conversation history (%s -> %s messages)', len(old), len(fresh))
    req.conversation = latest
    event.set_extra('_contextweave_history_refreshed', True)
