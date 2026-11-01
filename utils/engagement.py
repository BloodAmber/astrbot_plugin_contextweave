"""Describe why a turn was triggered without prescribing a speaking style."""
from astrbot.api.all import At, Plain, Reply


def is_group_invitation(event):
    if event.is_private_chat() or str(event.get_sender_id()) == str(event.get_self_id()):
        return False
    mentioned_self = False
    for part in event.message_obj.message or []:
        if isinstance(part, At):
            if str(part.qq) != str(event.get_self_id()):
                return False
            mentioned_self = True
        elif isinstance(part, Plain):
            if part.text.strip():
                return False
        elif isinstance(part, Reply):
            continue  # A quote can identify the discussion being invited into.
        else:
            return False
    return mentioned_self


def interaction_metadata(event):
    if event.get_extra('_contextweave_observation'):
        kind = 'observation'
    elif event.get_extra('_contextweave_invitation') or is_group_invitation(event):
        kind = 'group_invitation'
    elif event.is_private_chat() or not event.get_extra('_contextweave_proactive'):
        kind = 'direct'
    else:
        kind = 'group_opportunity'
    addressees = []
    for part in event.message_obj.message or []:
        value = part.qq if isinstance(part, At) else (
            part.sender_id if isinstance(part, Reply) else None)
        if value is not None and str(value) not in {'', '0'} and str(value) not in addressees:
            addressees.append(str(value))
    return {'kind':kind, 'trigger':event.get_extra('_contextweave_trigger', kind),
            'mentioned_or_quoted_sender_ids':addressees}
