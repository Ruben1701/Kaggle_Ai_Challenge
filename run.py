from kaggle_environments import make
import json
# run another match but with our empty agent
env = make("lux_ai_2021", configuration={"seed": 56221, "loglevel": 2}, debug=True)
print("env made")
steps = env.run(["./agent.py", ".v1/old_agent.py"])
print('steps')
replay = env.toJSON()
with open("../replay.json", "w") as f:
    json.dump(replay, f)
    print("saved to json")