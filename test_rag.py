import os
import subprocess
import time

# Start the app
proc = subprocess.Popen(
    ["python", "app.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    cwd=r"C:\Projects\rag-local-docs-assistant"
)

# Wait for loading
time.sleep(3)

# Send test questions
test_questions = [
    "Che cos'è bitcoin?",
    "Cos'è il mining di Bitcoin?",
]

for question in test_questions:
    proc.stdin.write(question + "\n")
    proc.stdin.flush()
    time.sleep(2)

proc.stdin.write("exit\n")
proc.stdin.flush()

# Get output
stdout, stderr = proc.communicate(timeout=30)
print(stdout)
if stderr:
    print("STDERR:")
    print(stderr)
