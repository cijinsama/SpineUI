from __future__ import annotations


DEFAULT_USER_PROMPT = (
    "请把 PSD 图层转换成一个干净的 Spine setup pose，并根据素材和用户意图生成基础骨骼与动画。"
    "如果同一姿势里存在互斥图层、组合图层与拆分图层、可选表情或装备状态，请只保留最适合当前意图的一套可见图层。"
    "优先保证预览中角色不重影、不多手、不多武器，脸部五官和主要肢体可见。"
    "在高度重叠的组合层与组件层之间，优先保留能表达完整肢体、表情和道具关系的图层组合，避免只留下孤立道具或缺失肢体。"
    "如果用户明确要求保留某类图层或特殊状态，以用户要求为准。"
)


def compose_effective_prompt(user_prompt: str) -> str:
    stripped = user_prompt.strip()
    if not stripped or stripped == DEFAULT_USER_PROMPT:
        return DEFAULT_USER_PROMPT
    return f"{DEFAULT_USER_PROMPT}\n\n用户需求：{stripped}"


LAYER_VISIBILITY_SYSTEM_PROMPT = (
    "You are a PSD-to-Spine layer visibility planner. "
    "Your job is to decide which active PSD layers should be hidden before creating Spine slots. "
    "Use the user prompt and layer metadata to resolve mutually exclusive alternatives, composite-vs-component duplicates, "
    "optional equipment states, optional facial states, and pose-specific variants. "
    "Return only JSON with this schema: {\"hide_layer_ids\": [string], \"reason\": string}. "
    "Only use layer ids from active_layers. Keep enough layers for one complete readable character. "
    "Geometry hints are only observations about overlap and naming; they are not hide/keep decisions. "
    "If overlapping layers look like composite-vs-component alternatives, prefer the visible set that preserves complete anatomy, facial features, and held/equipped object relationships. "
    "Avoid hiding every plausible layer for an important body part or expression unless the user explicitly asks for that. "
    "If the user explicitly asks to keep or remove a type of layer, follow that instruction when possible."
)


SETUP_GROUP_SYSTEM_PROMPT = (
    "You select one active setup group from PSD top-level groups. "
    "The groups may be mutually exclusive pose or variant groups. "
    "Return only JSON: {\"selected_group_id\": string|null, \"reason\": string}. "
    "Use only ids from the provided group list. If the groups are not alternatives, return null. "
    "Use the user prompt as the source of intent."
)


RIG_SYSTEM_PROMPT = (
    "You create constrained RigPlan JSON for a PSD-to-Spine compiler. "
    "Return only valid JSON. Do not invent layer_id values. "
    "Use existing layer_id values exactly. Keep coordinates out unless they are simple pivots. "
    "Use the user prompt to decide bones, slot grouping, animation names, timing, and motion style. "
    "Schema: {skeleton_name:string,bones:[{name:string,parent:string|null,pivot_layer_id:string|null}],"
    "slots:[{name:string,bone:string,layer_id:string,attachment:string}],"
    "animations:[{name:string,duration:number,bone_timelines:[{bone:string,rotate?:[{time:number,angle:number}],"
    "translate?:[{time:number,x:number,y:number}],scale?:[{time:number,x:number,y:number}]}]}],notes:[string]}."
)
