"""Main entrypoint for robotics digest demo."""
from .clustering.clustering import cluster_relevant_period
from .digest.digest import build_digest_for_user, build_focus_index
from .embeddings.embeddings import embed_texts
from .fake_data.fake_data import (
    generate_messages,
    generate_projects,
    generate_user_focus,
    generate_users,
)
from .vector_store.vector_store import MessageVectorStore


def build_index():
    users = generate_users()
    projects = generate_projects()
    messages = generate_messages(users, projects)

    texts = [m.text for m in messages]
    embeddings = embed_texts(texts)

    store = MessageVectorStore()
    #store.reset()
    store.add_messages(messages, embeddings)

    return users, projects, messages, embeddings, store


def run_demo(day: int = 18):
    """Run demo for specific day."""
    print(f"🤖 Generating demo for day {day}...")
    
    users, projects, messages, embeddings, store = build_index()
    clusters = cluster_relevant_period(messages, embeddings, day)
    focus_list = generate_user_focus(users, projects)
    focus_idx = build_focus_index(focus_list)
    
    print(f"Generated {len(messages)} messages across {len(users)} users, {len(projects)} projects")
    
    # Show digests for first 3 users
    for i, user in enumerate(users[:3]):
        digest = build_digest_for_user(
            user=user,
            day=day,
            clusters=clusters,
            projects=projects,
            messages=messages,
            embeddings=embeddings,
            focus_idx=focus_idx,
            max_items=8,
        )
        print(f"\n{'='*80}")
        print(f"DIGEST #{i+1} for {user.name} ({user.role})")
        print(f"{'='*80}")
        print(digest)
        print()

if __name__ == "__main__":
    run_demo()
