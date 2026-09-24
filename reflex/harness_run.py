import json, sys
from reflex.harness import run_task
task = json.loads(open(sys.argv[sys.argv.index("--task")+1]).read())
print(json.dumps(run_task(task), indent=1)[:4000])
