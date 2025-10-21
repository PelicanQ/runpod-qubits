import runpod
from five.hamil import eig_even_odd
import cupy as cp

print("Cuda driverGetVersion: ", cp.cuda.runtime.driverGetVersion())
print("Cuda runtimeGetVersion: ", cp.cuda.runtime.runtimeGetVersion())
print("Cuda get_local_runtime_version: ", cp.cuda.get_local_runtime_version())


def handler(event):
    print("Worker Start")
    input = event["input"]

    M = int(input.get("M"))

    print(f"Received: {M}")

    # Replace the sleep code with your Python function to generate images, text, or run any machine learning workload
    even, odd = eig_even_odd(
        1,
        1,
        1,
        1,
        1,
        50,
        50,
        50,
        50,
        50,
        0.1,
        0.1,
        0.01,
        0.1,
        0.1,
        0.01,
        M=M,
        only_energy=True,
    )

    return {"test": 2, "a": "a", "b": [1, 2, 3]}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
