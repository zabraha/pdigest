# robotics_digest/models.py
from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel

Role = Literal["ME", "EE", "SCM", "EM", "PM"]

class User(BaseModel):
    id: str
    name: str
    role: Role

class ProjectPhase(BaseModel):
    name: Literal["concept", "detailed_design", "proto_build", "dvt", "pvt", "ramp"]
    start_day: int
    end_day: int

class Project(BaseModel):
    id: str
    name: str
    phases: List[ProjectPhase]

class Message(BaseModel):
    id: str
    ts: datetime
    author_id: str
    project_id: str
    channel: str
    text: str
    thread_root_id: Optional[str] = None
    reactions: List[str] = []
    is_decision: bool = False
    is_risk: bool = False
    is_blocker: bool = False
    mentions: List[str] = []  # ["U1", "U3"]
    reacting_users: Dict[str, List[str]] = {}  # {"thumbsup": ["U2", "U4"]}
    reply_count: int = 0
    replies: List['Message'] = []  # direct replies

class UserFocus(BaseModel):
    user_id: str
    day: int          # 0..29
    project_ids: List[str]
