import triton
import triton.language as tl
import triton.compiler as tc

@triton.jit
def add_kernel(
    x_ptr,       
    y_ptr,       
    out_ptr,     
    n_elements,  
    BLOCK_SIZE: tl.constexpr, 
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)

def main():
    print("Compiling kernel to Triton-IR (TTIR) using Triton 3.x API...")
    
    # 1. Wrap the kernel in an ASTSource
    # FIX: Use exact argument names as strings
    src = tc.ASTSource(
        fn=add_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n_elements": "i32"},
        constexprs={"BLOCK_SIZE": 1024} 
    )
    
    # 2. Define a hardware target
    try:
        # Grabs the target configuration of your local GPU
        target = triton.runtime.driver.active.get_current_target()
    except Exception:
        # Fallback target if running on a machine without an active GPU
        from triton.backends.compiler import GPUTarget
        target = GPUTarget("cuda", 89, 32) 

    # 3. Compile the source
    compiled = tc.compile(src, target=target)

    # 4. Extract and save the Triton-IR
    ttir_code = compiled.asm["ttir"]
    out_filename = "add_kernel.ttir"
    
    with open(out_filename, "w") as f:
        f.write(ttir_code)
        
    print(f"Success! Triton-IR saved to '{out_filename}'.")

if __name__ == "__main__":
    main()