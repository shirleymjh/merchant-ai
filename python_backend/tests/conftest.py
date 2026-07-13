from __future__ import annotations

import os


# Unit/integration tests must never spend tokens or become nondeterministic just
# because a developer has a real provider key in python_backend/.env. Tests that
# exercise the curator explicitly inject a fake LLM and enable it in Settings.
os.environ["YSHOPPING_MEMORY_CURATOR_ENABLED"] = "false"
os.environ["YSHOPPING_KNOWLEDGE_CONFLICT_ENABLED"] = "false"
