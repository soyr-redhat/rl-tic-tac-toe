import pickle
import json
import numpy as np

with open("best_agent.pkl", "rb") as f:
    q_table = pickle.load(f)

exported = {}
for (state_bytes, action), value in q_table.items():
    board = np.frombuffer(state_bytes, dtype=np.int64)
    state_key = ",".join(str(int(x)) for x in board)
    key = f"{state_key}|{action}"
    exported[key] = round(value, 6)

with open("pages/q_table.json", "w") as f:
    json.dump(exported, f)

print(f"Exported {len(exported)} entries to pages/q_table.json")
