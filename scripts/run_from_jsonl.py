import json

with open("config/run_configs.jsonl", "r") as f:
    for line in f:
        config = json.loads(line)
        print(f"Running experiment: {config['name']}")
        os.system(config["cmd"])
