import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import os
from torch.utils.data import IterableDataset, DataLoader

# ==========================================
# 0. CONFIGURATION
# ==========================================
CURRICULUM_SCHEDULE = [4, 6, 8, 12, 16, 20] 
STEPS_PER_LEVEL = 2000 
BATCH_SIZE = 512
LR = 3e-4
MODEL_PATH = "rubik_master.pth"

MOVE_MAP = {
    0: "U", 1: "U'", 2: "L", 3: "L'", 4: "F", 5: "F'",
    6: "R", 7: "R'", 8: "B", 9: "B'", 10: "D", 11: "D'"
}

# ==========================================
# 1. PHYSICS ENGINE
# ==========================================
class PocketCube:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.state = np.array([c for c in range(6) for _ in range(4)], dtype=np.uint8)
        
    def is_solved(self):
        base = np.array([c for c in range(6) for _ in range(4)], dtype=np.uint8)
        return np.array_equal(self.state, base)
    
    def get_state(self):
        return self.state.copy()
    
    def apply_move(self, move_idx):
        s = self.state.copy()
        if move_idx == 0: # U
            s[0], s[1], s[2], s[3] = s[2], s[0], s[3], s[1]
            s[8],s[9], s[4],s[5], s[16],s[17], s[12],s[13] = s[12],s[13], s[8],s[9], s[4],s[5], s[16],s[17]
        elif move_idx == 2: # L
            s[4], s[5], s[6], s[7] = s[6], s[4], s[7], s[5]
            s[0],s[2], s[20],s[22], s[19],s[17], s[8],s[10] = s[19],s[17], s[8],s[10], s[20],s[22], s[0],s[2]
        elif move_idx == 4: # F
            s[8], s[9], s[10], s[11] = s[10], s[8], s[11], s[9]
            s[2],s[3], s[14],s[12], s[21],s[20], s[5],s[7] = s[5],s[7], s[2],s[3], s[14],s[12], s[21],s[20]
        elif move_idx == 6: # R
            s[12], s[13], s[14], s[15] = s[14], s[12], s[15], s[13]
            s[1],s[3], s[18],s[16], s[21],s[23], s[9],s[11] = s[9],s[11], s[1],s[3], s[18],s[16], s[21],s[23]
        elif move_idx == 8: # B
            s[16], s[17], s[18], s[19] = s[18], s[16], s[19], s[17]
            s[0],s[1], s[6],s[4], s[23],s[22], s[13],s[15] = s[13],s[15], s[23],s[22], s[6],s[4], s[0],s[1]
        elif move_idx == 10: # D
            s[20], s[21], s[22], s[23] = s[22], s[20], s[23], s[21]
            s[10],s[11], s[6],s[7], s[18],s[19], s[14],s[15] = s[6],s[7], s[18],s[19], s[14],s[15], s[10],s[11]
        else: # Primes
            base_move = move_idx - 1
            self.state = s
            for _ in range(3):
                self.apply_move(base_move)
            return
        self.state = s

# ==========================================
# 2. MODEL
# ==========================================
class RecursiveReasoningBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

class RubikTRM(nn.Module):
    def __init__(self, d_model=256, n_heads=8):
        super().__init__()
        self.color_embed = nn.Embedding(6, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 24, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.core = RecursiveReasoningBlock(d_model, n_heads)
        self.head = nn.Linear(d_model, 12)

    def forward(self, x, steps=4):
        B = x.size(0)
        h = self.color_embed(x) + self.pos_embed
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)
        
        for _ in range(steps):
            h = self.core(h)
            
        return self.head(h[:, 0, :])

# ==========================================
# 3. DATASET
# ==========================================
class InfiniteRubiksDS(IterableDataset):
    def __init__(self, max_scramble_depth=10):
        self.cube = PocketCube()
        self.max_depth = max_scramble_depth
    def __iter__(self):
        while True:
            self.cube.reset()
            scramble_moves = []
            depth = random.randint(1, self.max_depth)
            for _ in range(depth):
                move = random.randint(0, 11)
                self.cube.apply_move(move)
                scramble_moves.append(move)
            x = self.cube.get_state()
            last_move = scramble_moves[-1]
            y = (last_move + 1) if last_move % 2 == 0 else (last_move - 1)
            yield x, y

# ==========================================
# 4. SOLVER (BEAM + PENALTY)
# ==========================================
def solve_live(model, device, scramble_depth=10, max_moves=20, beam_width=5):
    print(f"\n>>> LIVE SOLVER (Beam k={beam_width}, Scramble Depth {scramble_depth}) <<<")
    
    cube = PocketCube()
    print("Scramble: ", end="")
    for _ in range(scramble_depth):
        m = random.randint(0, 11)
        cube.apply_move(m)
        print(f"{MOVE_MAP[m]} ", end="")
    print("\n---------------------------------------------------")
    
    beams = [(0.0, cube.get_state(), [])] 
    model.eval()
    
    for step in range(max_moves):
        candidates = []
        
        for score, state, history in beams:
            # Check solved
            temp_cube = PocketCube()
            temp_cube.state = state
            if temp_cube.is_solved():
                print(f"!!! SOLVED in {len(history)} moves !!!")
                print(f"Solution: {' '.join(history)}")
                return True
            
            x_tensor = torch.tensor(state).unsqueeze(0).long().to(device)
            with torch.no_grad():
                logits = model(x_tensor, steps=20)
                
                # --- DIVERSITY INJECTION ---
                # Apply a penalty to moves we have done recently
                # If we just did R, R, R... doing R again costs extra
                if len(history) > 3:
                    last_3 = history[-3:]
                    for move_idx in range(12):
                        move_str = MOVE_MAP[move_idx]
                        count = last_3.count(move_str)
                        if count >= 2:
                            logits[0, move_idx] -= 2.0 * count # Heavy penalty!
                # ---------------------------

                probs = torch.softmax(logits, dim=1)
                # Expand TOP 6 (more width)
                topk_probs, topk_indices = torch.topk(probs, 6, dim=1)
                
            for i in range(6):
                move_idx = topk_indices[0, i].item()
                move_prob = topk_probs[0, i].item()
                move_str = MOVE_MAP[move_idx]
                
                # Immediate Inverse Check (U -> U')
                if len(history) > 0:
                    last_move = history[-1]
                    last_idx = -1
                    for k,v in MOVE_MAP.items(): 
                        if v == last_move: last_idx = k
                    
                    is_inverse = False
                    if move_idx % 2 == 0 and move_idx + 1 == last_idx: is_inverse = True
                    if move_idx % 2 == 1 and move_idx - 1 == last_idx: is_inverse = True
                    if is_inverse: continue 

                new_cube = PocketCube()
                new_cube.state = state.copy()
                new_cube.apply_move(move_idx)
                
                new_score = score - np.log(move_prob + 1e-9) 
                new_history = history + [move_str]
                candidates.append((new_score, new_cube.get_state(), new_history))
        
        # Sort and prune
        candidates.sort(key=lambda x: x[0])
        # Diversity Pruning: Don't keep 5 beams that are identical state
        unique_candidates = []
        seen_states = set()
        for c in candidates:
            s_tuple = tuple(c[1])
            if s_tuple not in seen_states:
                unique_candidates.append(c)
                seen_states.add(s_tuple)
            if len(unique_candidates) >= beam_width: break
            
        beams = unique_candidates
        
        if beams:
            best_beam = beams[0]
            conf = np.exp(-best_beam[0] / (len(best_beam[2]) + 1e-9)) 
            print(f"Step {step+1}: Best Path (Avg Conf {conf:.1%}) -> {best_beam[2]}")

    print(f"... Failed to solve within {max_moves} moves.")
    return False

# ==========================================
# 5. MAIN ROUTINE
# ==========================================
def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {DEVICE}")

    model = RubikTRM().to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        print(f"Found saved model {MODEL_PATH}. Loading...")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("No saved model found. Starting CURRICULUM TRAINING...")
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        ds = InfiniteRubiksDS(max_scramble_depth=12)
        dl = DataLoader(ds, batch_size=BATCH_SIZE)
        
        for depth in CURRICULUM_SCHEDULE:
            print(f"--- Training Recursive Depth {depth} ---")
            model.train()
            data_iter = iter(dl)
            for step in range(STEPS_PER_LEVEL):
                try: x, y = next(data_iter)
                except: data_iter = iter(dl); x, y = next(data_iter)
                x, y = x.long().to(DEVICE), y.long().to(DEVICE)
                optimizer.zero_grad()
                logits = model(x, steps=depth) 
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if step % 1000 == 0:
                    print(f"   Step {step} Loss: {loss.item():.4f}")
        
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Training Complete. Model saved to {MODEL_PATH}")

    print("\nModel is ready. Let's play!")
    
    # Test 1: Easy
    solve_live(model, DEVICE, scramble_depth=5, max_moves=10, beam_width=3)
    
    # Test 2: Medium
    solve_live(model, DEVICE, scramble_depth=10, max_moves=20, beam_width=5)
    
    # Test 3: Hard (Depth 20) with wider beam and penalty
    solve_live(model, DEVICE, scramble_depth=20, max_moves=60, beam_width=12)

if __name__ == "__main__":
    main()