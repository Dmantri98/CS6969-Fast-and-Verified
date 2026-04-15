import triton
import triton.language as tl
import triton.compiler as tc
import torch

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # 1. Generate 1D offsets
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # 2. Simplified Masking
    # Instead of (64, 1) < i32, create a 1D mask and then expand it
    mask_m = rm < M
    mask_n = rn < N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offset = k * BLOCK_SIZE_K
        
        # Pointers: Use explicit 2D expansion here
        curr_a_ptrs = a_ptr + (rm[:, None] * stride_am + (rk[None, :] + k_offset) * stride_ak)
        curr_b_ptrs = b_ptr + ((rk[:, None] + k_offset) * stride_bk + rn[None, :] * stride_bn)
        
        # 3. Use the 1D masks with expansion inside the load
        # This often avoids the materialization error because Triton 
        # handles the expansion during the LoadOp conversion.
        a = tl.load(curr_a_ptrs, mask=mask_m[:, None])
        b = tl.load(curr_b_ptrs, mask=mask_n[None, :])
        
        accumulator += tl.dot(a, b)

    # Store logic
    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator, mask=mask_m[:, None] & mask_n[None, :])

# --- Compilation and IR Export ---

# 1. Wrap the kernel in an ASTSource
src = tc.ASTSource(
    fn=matmul_kernel,
    signature={
        "a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
        "M": "i32", "N": "i32", "K": "i32",
        "stride_am": "i32", "stride_ak": "i32",
        "stride_bk": "i32", "stride_bn": "i32",
        "stride_cm": "i32", "stride_cn": "i32"
    },
    constexprs={
        "BLOCK_SIZE_M": 256, 
        "BLOCK_SIZE_N": 256, 
        "BLOCK_SIZE_K": 256
    } 
)

# 2. Define a hardware target
try:
    target = triton.runtime.driver.active.get_current_target()
except Exception:
    from triton.backends.compiler import GPUTarget
    # Standard fallback (A100/H100 class)
    target = GPUTarget("cuda", 80, 32) 

# 3. Compile the source
# We pass the target to the compiler
compiled = tc.compile(src, target=target)

# 4. Extract and save the Triton-IR (TTIR)
ttir_code = compiled.asm["ttir"]
out_filename = "matmul_kernel.ttir"

with open(out_filename, "w") as f:
    f.write(ttir_code)

print(f"Success! MatMul Triton-IR saved to '{out_filename}'.")