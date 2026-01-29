from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import ollama

from ..fake_data.fake_data import current_phase
from ..models.models import Message, Project, User, UserFocus


def user_interest_vector(
    user: User, 
    messages: List[Message], 
    embeddings: np.ndarray,
    lookback_days: int = 14
) -> np.ndarray:
    """
        Compute user's interest vector from:
            1. Messages they WROTE (weight: 3.0)
            2. Messages they REPLIED to (weight: 2.0)  
            3. Messages they REACTED to (weight: 1.5)
            4. Messages that MENTION them (weight: 1.0)
    """
    base_ts = messages[0].ts  # assume sorted
    cutoff_ts = base_ts + timedelta(days=lookback_days)
    
    engagement_idxs: List[tuple[float, int]] = []
    
    for i, msg in enumerate(messages):
        if msg.ts > cutoff_ts:
            continue  # too old
            
        weight = 0.0
        
        # 1. Messages they authored (strongest signal)
        if msg.author_id == user.id:
            weight += 3.0
            
        # 2. Messages they replied to (we'd need thread_root_id matching)
        elif msg.thread_root_id and any(r.author_id == user.id for r in msg.replies):
            weight += 2.0
            
        # 3. Messages they reacted to (need reactions field to include user IDs)
        elif msg.reacting_users and any(user.id in reactors for reactors in msg.reacting_users.values()):
            weight += 1.5
            
        # 4. Messages mentioning them (need @mentions parsing)
        elif f"@{user.id}" in msg.text or user.name.lower() in msg.text.lower():
            weight += 1.0
            
        # Fresh engagement matters more
        days_old = (cutoff_ts - msg.ts).days
        freshness_weight = max(0.1, 1.0 - (days_old / lookback_days) * 0.9)
        final_weight = weight * freshness_weight
        if final_weight > 0:
            engagement_idxs.append((final_weight, i))
    
    if not engagement_idxs:
        return np.zeros(embeddings.shape[1])
    
    # Weighted average of engaged message embeddings
    weighted_embs = np.zeros(embeddings.shape[1])
    total_weight = 0.0
    
    for weight, idx in engagement_idxs:
        weighted_embs += weight * embeddings[idx]
        total_weight += weight
    
    return weighted_embs / total_weight

def get_user_top_clusters(
    user: User, 
    clusters: Dict[int, List[int]],  # cluster indexes to message indexes
    messages: List[Message], 
    embeddings: np.ndarray
) -> List[int]:
    """Rank clusters by similarity to user's interest vector."""
    user_vec = user_interest_vector(user, messages, embeddings)
    
    cluster_scores = []
    for cid, msg_idxs in clusters.items():
        centroid = embeddings[msg_idxs].mean(axis=0)
        if np.linalg.norm(centroid) == 0 or np.linalg.norm(user_vec) == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(centroid, user_vec))
        cluster_scores.append((similarity, cid))
    
    cluster_scores.sort(reverse=True)
    return [cid for _, cid in cluster_scores[:4]]  # Top 4 clusters




def build_focus_index(focus_list: List[UserFocus]) -> Dict[tuple, UserFocus]:
    return {(f.user_id, f.day): f for f in focus_list}

def role_topic_weight(msg: Message, role: str, phase: str) -> float:
    w = 1.0
    if role in ("ME", "EE"):
        if phase in ("detailed_design", "proto_build"):
            if msg.is_decision or msg.is_blocker:
                w += 2.0
        if phase in ("dvt", "pvt"):
            if msg.is_risk:
                w += 2.0
    if role == "SCM":
        if msg.is_blocker or msg.is_risk:
            w += 2.0
    if role in ("EM", "PM"):
        if msg.is_decision or msg.is_risk or msg.is_blocker:
            w += 2.0
    # slight bump for “important” messages
    w += 0.2 * len(msg.reactions)
    return w

def build_digest_for_user(
    user: User,
    day: int,
    clusters: Dict[int, List[int]],
    projects: List[Project],
    messages: List[Message],
    embeddings,  # ndarray, unused directly here but available if you want similarity
    focus_idx: Dict[tuple, UserFocus],
    max_items: int = 15,
) -> str:
    base = datetime(2025, 1, 1)
    focus = focus_idx.get((user.id, day))
    if not focus:
        return f"No digest for {user.name} on day {day}."

    proj_by_id = {p.id: p for p in projects}

    # get top clusters for this user
    top_clusters = get_user_top_clusters(user, clusters, messages, embeddings)

    # Candidate messages: same day + in focused projects
    candidates: List[Tuple] = []
    for i, m in enumerate(messages):
        m_day = (m.ts - base).days
        if m_day != day:
            continue
        if m.project_id not in focus.project_ids:
            continue
        candidates.append((i,m))
    
 

    # Score each candidate
    scored: List[tuple[float, Message]] = []
    for _,(i,m) in enumerate(candidates):
        phase = current_phase(proj_by_id[m.project_id], day)
        score = role_topic_weight(m, user.role, phase)
        for cid in top_clusters:
            if i in clusters.get(cid, []):
                score += 1.0
                break
        scored.append((score, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_msgs = [m for _, m in scored[:max_items]]

    digest = generate_llm_digest(user, top_msgs, projects, focus.project_ids, day)
    return digest

def generate_llm_digest(
    user: User, 
    messages: List[Message], 
    projects: List[Project], 
    focus_projects: List[str],
    day: int
) -> str:
    """Use Ollama to generate natural, concise digest."""
    
    if not messages:
        return f"**Daily digest for {user.name} ({user.role}) – Day {day}**\n\nNo high-priority updates for your focus projects today."
    
    # Build context for LLM
    proj_by_id = {p.id: p for p in projects}
    
    context_msgs = []
    for m in messages:
        proj_name = proj_by_id[m.project_id].name
        phase = current_phase(proj_by_id[m.project_id], day)
        tags = []
        if m.is_decision: 
            tags.append("DECISION")
        if m.is_risk: 
            tags.append("RISK") 
        if m.is_blocker: 
            tags.append("BLOCKER")
        tag_str = f"[{', '.join(tags)}]" if tags else ""
        
        context_msgs.append(
            f"{proj_name} ({phase}): {tag_str} {m.text} (@ {m.author_id})"
        )
    
    context = "\n".join(context_msgs[:12])  # Top 12 messages max
    
    # LLM Prompt (optimized for brevity + actionability)
    prompt = f"""You are creating a daily digest for a {user.role} engineer. 

FOCUS PROJECTS: {', '.join(focus_projects)}

RELEVANT UPDATES:
{context}

Generate a concise digest (200-300 words max) with:
1. One-line summary of key themes
2. 3-6 bullet points of ACTIONABLE items (decisions, blockers, risks)
3. Project-phase context where relevant

Format with markdown headers. Be direct, skimmable, and action-oriented."""

    try:
        response = ollama.generate(
            model="llama3.2:3b",  # Fast, free, local
            prompt=prompt,
            options={
                "temperature": 0.1,  # Low creativity for consistency
                "num_predict": 400,  # Word limit
            }
        )
        return f"**Daily digest for {user.name} ({user.role}) – Day {day}**\n\n" + response['response']
        
    except Exception as e:
        # Fallback to rule-based if LLM fails
        print(f"LLM failed: {e}, using fallback")
        return build_rule_based_digest(user, messages, projects, day)

def build_rule_based_digest(user, messages, projects, day):
    """Fallback if LLM unavailable."""
    proj_by_id = {p.id: p for p in projects}
    grouped = defaultdict(list)
    for m in messages:
        grouped[m.project_id].append(m)
    
    lines = [f"**Daily digest for {user.name} ({user.role}) – Day {day}**"]
    lines.append("")
    
    for pid in grouped:
        proj = proj_by_id[pid]
        phase = current_phase(proj, day)
        lines.append(f"### {proj.name} – {phase.replace('_', ' ').title()}")
        for m in grouped[pid][:4]:
            tag = ""
            if m.is_decision: 
                tag = "[DECISION] "
            elif m.is_risk: 
                tag = "[RISK] "
            elif m.is_blocker: 
                tag = "[BLOCKER] "
            lines.append(f"- {tag}{m.text}")
        lines.append("")
    
    return "\n".join(lines)
