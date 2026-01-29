# robotics_digest/fake_data.py
import random
from datetime import datetime, timedelta
from typing import List, Tuple

from ..models.models import Message, Project, ProjectPhase, User, UserFocus

RNG = random.Random(42)

def generate_users() -> List[User]:
    roles = ["ME", "EE", "SCM", "EM", "PM"]
    names = ["Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy"]
    users = []
    for i, name in enumerate(names):
        role = roles[i % len(roles)]
        users.append(User(id=f"U{i}", name=name, role=role))  # type: ignore[arg-type]
    return users

def generate_projects() -> List[Project]:
    # Two robotics projects with overlapping/transitioning phases over 30 days
    phases_p1 = [
        ProjectPhase(name="concept",         start_day=0,  end_day=4),
        ProjectPhase(name="detailed_design", start_day=5,  end_day=14),
        ProjectPhase(name="proto_build",     start_day=15, end_day=24),
        ProjectPhase(name="dvt",             start_day=25, end_day=29),
    ]
    phases_p2 = [
        ProjectPhase(name="concept",         start_day=10, end_day=14),
        ProjectPhase(name="detailed_design", start_day=15, end_day=19),
        ProjectPhase(name="proto_build",     start_day=20, end_day=29),
    ]
    return [
        Project(id="P1", name="Robot Arm", phases=phases_p1),
        Project(id="P2", name="Mobile Base", phases=phases_p2),
    ]

def current_phase(project: Project, day: int) -> str:
    for ph in project.phases:
        if ph.start_day <= day <= ph.end_day:
            return ph.name
    return project.phases[-1].name

def generate_user_focus(users: List[User], projects: List[Project]) -> List[UserFocus]:
    focus: List[UserFocus] = []
    for day in range(30):
        for u in users:
            # Simple rule: first half mostly P1, second half mostly P2, with some overlap
            if day < 10:
                proj_ids = ["P1"]
            elif day < 20:
                proj_ids = ["P1", "P2"]
            else:
                proj_ids = ["P2"]
            focus.append(UserFocus(user_id=u.id, day=day, project_ids=proj_ids))
    return focus

def sample_message_text(role: str, phase: str) -> Tuple[str, bool, bool, bool]:
    # Very rough templates to create realistic content
    decisions = [
        "DECISION: switch bearing supplier due to lead time risk.",
        "DECISION: increase motor torque margin by 10%.",
        "DECISION: freeze mechanical interface for elbow joint.",
    ]
    risks = [
        "RISK: proto build might slip due to PCB re-spin.",
        "RISK: yield below target for first DVT build.",
    ]
    blockers = [
        "BLOCKER: test lab access blocked, awaiting safety approval.",
        "BLOCKER: missing critical part, awaiting SCM update.",
    ]
    generics = [
        "Synced on latest test results and next steps.",
        "Updated CAD for mounting bracket, ready for review.",
        "Reviewed firmware bring-up log, no new issues found.",
    ]

    roll = RNG.random()
    if roll < 0.15:
        txt = RNG.choice(decisions)
        return txt, True, False, False
    elif roll < 0.27:
        txt = RNG.choice(risks)
        return txt, False, True, False
    elif roll < 0.35:
        txt = RNG.choice(blockers)
        return txt, False, False, True
    else:
        return RNG.choice(generics), False, False, False

def generate_messages(
    users: List[User],
    projects: List[Project],
    days: int = 30,
    msgs_per_day: int = 80,
) -> List[Message]:
    base_ts = datetime(2025, 1, 1, 9, 0, 0)
    msgs: List[Message] = []
    msg_id = 0

    for day in range(days):
        for _ in range(msgs_per_day):
            user = RNG.choice(users)
            project = RNG.choice(projects)
            phase = current_phase(project, day)
            text, is_decision, is_risk, is_blocker = sample_message_text(user.role, phase)

            ts = base_ts + timedelta(days=day, minutes=RNG.randint(0, 8 * 60))
            reactions = []
            if is_decision or is_risk or is_blocker:
                reactions = ["thumbsup", "fire"]

            msgs.append(
                Message(
                    id=f"M{msg_id}",
                    ts=ts,
                    author_id=user.id,
                    project_id=project.id,
                    channel=f"#proj-{project.id.lower()}",
                    text=f"[{phase.upper()}] {text}",
                    reactions=reactions,
                    is_decision=is_decision,
                    is_risk=is_risk,
                    is_blocker=is_blocker,
                )
            )
            msg_id += 1
            for msg in msgs:
                # Add realistic engagement
                msg.mentions = random.sample([u.id for u in users if u.id != msg.author_id], 
                                        k=random.randint(0, 2))
                
                # 10% chance someone reacts
                if random.random() < 0.1:
                    reactors = random.sample([u.id for u in users if u.id != msg.author_id], 
                                        k=random.randint(1, 3))
                    msg.reacting_users["thumbsup"] = reactors
    return msgs
