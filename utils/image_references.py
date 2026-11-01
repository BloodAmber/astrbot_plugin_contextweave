"""Follow message IDs only; never infer a reference from words or recency."""
from astrbot.api.all import Image, Reply
from .reply_links import ReplyLinks


def scoped_messages(event,history):
    result={}
    now=float(event.message_obj.timestamp)
    current=str(event.message_obj.message_id)
    for message in history:
        if float(getattr(message,'timestamp',0) or 0)<=now:
            result[str(message.message_id)]=message
        if str(message.message_id)==current:
            break
    result[current]=event.message_obj
    return result


def trace_image_ids(message, messages, links, self_id, visited=None, depth=0):
    if depth>8:
        return []
    visited=set() if visited is None else visited
    mid=str(message.message_id)
    if mid in visited:
        return []
    visited.add(mid)
    if any(isinstance(p,Image) for p in message.message or []):
        return [mid]
    found=[]
    for part in message.message or []:
        if not isinstance(part,Reply):
            continue
        target=str(part.id)
        if str(part.sender_id)==str(self_id):
            # The ledger contains only actual successfully sent QQ message IDs.
            link=links.get(target)
            if not link:
                continue
            target=str(link['source_message_id'])
        source=messages.get(target)
        if source is not None:
            found.extend(trace_image_ids(source,messages,links,self_id,visited,depth+1))
    return list(dict.fromkeys(found))


def linked_reference_ids(event,req,history):
    messages=scoped_messages(event,history)
    links=ReplyLinks.read(event,req.conversation.cid) if req.conversation else {}
    return trace_image_ids(event.message_obj,messages,links,event.get_self_id())

