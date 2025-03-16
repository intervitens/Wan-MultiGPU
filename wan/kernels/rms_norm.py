import torch
import tilelang
import tilelang.language as T

_cached = {}


def rms_norm_kernel(M, N, eps, blk_m):
    dtype = "float16"
    dtype_norm = "float"

    @T.prim_func
    def main(A: T.Buffer((M, N), dtype), 
             W: T.Buffer((N,), dtype), 
             B: T.Buffer((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            A_shared = T.alloc_shared((blk_m, N), dtype_norm)
            W_local = T.alloc_fragment((N,), dtype)
            A_local = T.alloc_fragment((blk_m, N), dtype_norm)
            A_powsum = T.alloc_fragment((blk_m,), dtype_norm)

            T.copy(A[bx * blk_m:(bx + 1) * blk_m, :], A_shared)
            T.copy(W, W_local)
            for i, j in T.Parallel(blk_m, N):
                A_local[i, j] = A_shared[i, j] * A_shared[i, j]
            T.reduce_sum(A_local, A_powsum, dim=1)
            for i in T.Parallel(blk_m):
                A_powsum[i] = T.rsqrt(A_powsum[i] / N) + eps
            for i, j in T.Parallel(blk_m, N):
                A_shared[i, j] *= A_powsum[i] * W_local[j]
            T.copy(A_shared, B[bx * blk_m:(bx + 1) * blk_m, :])

    return main

def rms_norm(X, W, eps):
    global _cached
    blk_m = 1
    M, N = X.shape

    key = ("rms_norm", M, N, eps, blk_m)
    if key not in _cached:
        program = rms_norm_kernel(M, N, eps, blk_m)
        kernel = tilelang.compile(program, out_idx=-1, target="cuda", execution_backend="cython")
        _cached[key] = kernel
    else:
        kernel = _cached[key]
    return kernel(X, W)

def rms_norm_precompile(M, N, eps):
    global _cached
    blk_m = 1
    key = ("rms_norm", M, N, eps, blk_m)
    if key not in _cached:
        program = rms_norm_kernel(M, N, eps, blk_m)
        kernel = tilelang.compile(program, out_idx=-1, target="cuda", execution_backend="cython")
        _cached[key] = kernel
    else:
        kernel = _cached[key]
    return kernel


def ref_program(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-12).to(torch.float16) * w


if __name__ == "__main__":
    M, N, blk_m, blk_k = 18522, 5120, 1, 512
    program = rms_norm(M, N, blk_m, 1e-12)
    kernel = tilelang.compile(program, out_idx=-1, target="cuda", execution_backend="cython")
    profiler = kernel.get_profiler()
    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    print("All checks pass.")

    latency = profiler.do_bench(ref_program, warmup=500)
    print("Ref: {:.2f} ms".format(latency))
    latency = profiler.do_bench(profiler.mod, warmup=500)
    print("Tile-lang: {:.2f} ms".format(latency))
