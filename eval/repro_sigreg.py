
import torch
from loss import SIGReg, LeJEPA

def test_sigreg_accum_invariance():
    torch.manual_seed(42)
    device = "cpu"
    D = 128
    N = 256 # Large batch
    
    # Random projections
    proj = torch.randn(N, 1, D, requires_grad=True)
    
    # 1. Full Batch Loss
    sigreg = SIGReg()
    # LeJEPA expects all_proj as (N, V, D)
    # We'll just look at SIGReg direct usage or LeJEPA usage
    # LeJEPA calls: sigreg(all_proj.reshape(N*V, D))
    
    # Direct SIGReg
    flat_proj = proj.reshape(-1, D)
    loss_full = sigreg(flat_proj)
    
    print(f"Full Batch Loss (N={N}): {loss_full.item()}")
    
    # 2. Accumulated Loss (Simulating grad_accum=2)
    # Split into 2 micro-batches
    micro_N = N // 2
    proj1 = proj[:micro_N]
    proj2 = proj[micro_N:]
    
    flat_proj1 = proj1.reshape(-1, D)
    flat_proj2 = proj2.reshape(-1, D)
    
    loss1 = sigreg(flat_proj1)
    loss2 = sigreg(flat_proj2)
    
    loss_accum = (loss1 + loss2) / 2
    
    print(f"Accumulated Loss (Avg of 2 batches of {micro_N}): {loss_accum.item()}")
    
    diff = abs(loss_full.item() - loss_accum.item())
    print(f"Difference: {diff}")
    
    if diff > 1e-4:
        print("FAIL: SIGReg is not invariant to batch splitting!")
    else:
        print("PASS: SIGReg is invariant.")

if __name__ == "__main__":
    test_sigreg_accum_invariance()
