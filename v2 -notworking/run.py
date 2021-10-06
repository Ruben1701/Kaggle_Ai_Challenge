from kaggle_environments import make
import json
# run another match but with our empty agent
env = make("lux_ai_2021", configuration={"seed": 56221, "loglevel": 2}, debug=True)
steps = env.run(["./agent.py", "./agent.py"])
replay = env.toJSON()
with open("../replay.json", "w") as f:
    json.dump(replay, f)
    print("saved to json")