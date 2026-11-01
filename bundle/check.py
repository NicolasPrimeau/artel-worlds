import os
import sys
import urllib.request

url = sys.argv[1]
for pair in os.environ["ALL"].split(","):
    agent, key = pair.split(":")
    project = agent.rsplit("-", 1)[0]
    request = urllib.request.Request(
        f"{url}/projects/{project}/join",
        method="POST",
        headers={"x-agent-id": agent, "x-api-key": key},
    )
    urllib.request.urlopen(request, timeout=5)
print(f"bundle: {len(os.environ['ALL'].split(','))} agents authenticated")
