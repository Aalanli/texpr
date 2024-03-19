# %%
import tvm
import tvm.testing
from tvm import te
import numpy as np
from tvm import te, auto_scheduler

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def sum():
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

    return [A, B, C]

target = tvm.target.Target("cuda")
task = auto_scheduler.SearchTask(func=sum, args=(), target=target)

# Inspect the computational graph
print("Computational DAG:")
print(task.compute_dag)

log_file = "sum.json"
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=10,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

# Run auto-tuning (search)
task.tune(tune_option)
# Apply the best schedule
sch, args = task.apply_best(log_file)
