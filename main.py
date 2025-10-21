import runpod
from five.hamil import eig_even_odd
import cupy as cp

print("Cuda driverGetVersion: ", cp.cuda.runtime.driverGetVersion())
print("Cuda runtimeGetVersion: ", cp.cuda.runtime.runtimeGetVersion())
print("Cuda get_local_runtime_version: ", cp.cuda.get_local_runtime_version())


def handler(event):
    print("Worker Start")
    job = event["input"]

    print(f"Received: {job["M"]}")
    vals = eig_even_odd(
        job["Ec1"],
        job["Ec2"],
        job["Ec3"],
        job["Ec4"],
        job["Ej5"],
        job["Ej1"],
        job["Ej2"],
        job["Ej3"],
        job["Ej4"],
        job["Ej5"],
        job["Eint12"],
        job["Eint23"],
        job["Eint13"],
        job["Eint34"],
        job["Eint45"],
        job["Eint35"],
        M=job["M"],
        only_energy=True,
    )

    return {"energies": vals}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
