import argparse
import cProfile
import avagrad as ag
from functools import partial
from pathlib import Path


FUNCTION_NAMES = ["matmul", "matmul_backward", "all"]
PROFS_PATH = Path("profs")

def prof_func(func: callable, name: str, size: int):
    profiler = cProfile.Profile(subcalls=True)
    profiler.enable()
    func()
    profiler.disable()
    version = ag.__version__
    folder = PROFS_PATH / version
    folder.mkdir(exist_ok=True)
    profiler.dump_stats(folder / f"n={name}_s={size}.prof")


# -----------------------------------------------------------------------------
def test_matmul(size: int):
    def exec(a, b):
        ag.matmul(a, b)

    a = ag.rand((size, size))
    b = ag.rand((size, size))
    func = partial(exec, a=a, b=b)
    prof_func(func, "MatMul", size)


def test_matmul_backward(size: int):
    def exec(c):
        c.backward()

    a = ag.rand((size, size), track_gradient=True)
    b = ag.rand((size, size), track_gradient=True)
    c = ag.matmul(a, b)
    func = partial(exec, c=c)
    prof_func(func, "MatMul.Backward", size)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--name", type=str, default="matmul")
    parser.add_argument("--size", type=int, default=5000)

    args = parser.parse_args()
    func_name = args.name
    size = args.size

    if func_name == "matmul":
        test_matmul(size)

    elif func_name == "matmul_backward":
        test_matmul_backward(size)

    elif func_name == "all":
        test_matmul(size)
        test_matmul_backward(size)

    else:
        raise ValueError(f"Supported functions are: {FUNCTION_NAMES}")
