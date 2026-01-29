# Daily Digest for Slack 

Slack messages are ingested and stored in a vector db as embeddings and metadata.
For this prototype fake data is generated and used instead of the slack api and slack-sdk.
Chroma DB is used for the vector store with 30 days of fake data.
Fake data simulates project phase transitions (concept->proto->DVT)
14-day clustering to capture evolving projects, phases and topics and change user interest and focus areas.
User interest vectors from engagement signals. A current user interest vector is generated from past 14 day interaction of user like messages authored by the user, replied to by the user, mentions of the user and reactions by the user. A weighted average of all these embeddings is used to generate current user interest vector.
Cosine similarity is calcuate between user interest vector and the cluster centroids to suface the top clusters.
A scoring algorithm is used to score messages based on categories, importance and relevant to user role.
LLM digest generation (Ollama/llama3.2). Top messages are send to an LLM for formatting as a daily digest

## Run the Demo

uv run python -m robotics_digest.main

