## Env
    conda activate triton
    cd to ~/school/CS6969-Fast-and-Verified/final_project/
    run source ./triton/.venv/bin/activate

## Lowering Path to Specialized Linalg

Run python3 mat_mult_kernel.py

run triton/build/cmake.linux-x86_64-cpython-3.10/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt --triton-to-linalg --linalg-specialize-generic-ops matmul_kernel_#tile_size.ttir > matmul_kernel_#tile_size.linalg

