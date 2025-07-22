# Spatial BLAS Routines on Cerebras WSE

This project implements and benchmarks five foundational BLAS routines (AXPY, DOT, SCAL, GER, TRSV) using the **Cerebras Software Language (CSL)** and the Cerebras SDK simulation environment. Each routine is expressed in a spatial programming model and designed to run across multiple Processing Elements (PEs) in a parallelized and modular fashion.

## Project Structure

Each routine is organized in its own directory.
Each subdirectory contains:
- `pe_program.csl`: the kernel implementation
- `layout.csl`: spatial mapping to PEs
- `run.py`: host script for simulation and validation
l
Shared timing infrastructure is imported via `syncpe.csl` and `synclayout.csl`.

## Compilation

Each routine can be compiled using the `cslc` compiler provided in the SDK. Example (for AXPY):

```bash
$ cslc layout.csl 
  --fabric-dims=11,3 
  --fabric-offsets=4,1 
  --params=N:500,width:10,N_PER_PE:50 
  --memcpy --channels=1 
  -o out
```
## Running the Simulation
Once compiled, run the simulation via Python:

`$ cs_python run.py --name out`
This will:
- Initialize input buffers
- Launch the kernel
- Collect and validate results using NumPy or SciPy
- Perform cycle counting

## Notes
This code is designed for use with the Cerebras SDK simulator only.

Large input sizes (e.g., TRSV/GER with N=5000) may exceed memory or disk limits depending on the machine.

The routines are tested for correctness and compared against NumPy/SciPy references.
