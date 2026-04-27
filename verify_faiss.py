import faiss
import numpy as np

print(f"FAISS version loaded: {faiss.__version__}")

try:
    # 1. Initialize AMD GPU Resources
    # (FAISS still uses standard "GpuResources" naming even for ROCm)
    res = faiss.StandardGpuResources()
    print("✅ ROCm GPU Resources initialized successfully!")

    # 2. Generate some dummy data
    d = 64  # dimension of vectors
    nb = 10000  # database size
    nq = 5  # number of queries
    np.random.seed(42)

    # FAISS requires float32 arrays
    xb = np.random.random((nb, d)).astype("float32")
    xq = np.random.random((nq, d)).astype("float32")

    # 3. Create a CPU index, then move it to the GPU
    cpu_index = faiss.IndexFlatL2(d)

    # The '0' here means use GPU device 0 (your 7900 XT)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    print("✅ Index successfully moved to the RX 7900 XT.")

    # 4. Add data to the GPU and search
    gpu_index.add(xb)
    print(f"✅ Indexed {gpu_index.ntotal} vectors in VRAM.")

    k = 4  # return 4 nearest neighbors
    distances, indices = gpu_index.search(xq, k)

    print("\n🎉 SUCCESS: FAISS-ROCm is fully operational! Here are the search results:")
    print("Indices of nearest neighbors:\n", indices)

except AttributeError as e:
    print(f"❌ Error: FAISS loaded, but it does not have GPU support compiled. ({e})")
except Exception as e:
    print(f"❌ Error during GPU execution: {e}")
