import subprocess, sys, time
log = open("run_pipeline_full.log", "w")
proc = subprocess.Popen(
    ["kedro", "run", "--pipeline", "load_data"],
    stdout=log, stderr=subprocess.STDOUT,
    cwd="/eos/home-i03/m/maglowac/ctl2_embedding/triggerflow_template/ctl2_model"
)
with open("pipeline.pid", "w") as f:
    f.write(str(proc.pid))
print(f"Launched PID {proc.pid}")
