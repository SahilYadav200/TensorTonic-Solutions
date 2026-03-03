import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Computes the sinusoidal positional encoding matrix.
    """
    # 1. Create a column vector for positions: shape (seq_len, 1)
    positions = np.arange(seq_len)[:, np.newaxis]
    
    # 2. Create an array of dimension indices: shape (d_model,)
    dims = np.arange(d_model)
    
    # 3. Calculate the exponent for the base divisor
    # We use integer division (//) so that pairs of indices (0,1), (2,3) get the same exponent
    # Formula equivalent: 2i / d_model
    exponent = (dims // 2) * 2 / d_model
    divisors = base ** exponent
    
    # 4. Calculate angles using broadcasting: (seq_len, 1) / (d_model,) -> (seq_len, d_model)
    angles = positions / divisors
    
    # 5. Initialize the positional encoding matrix
    pe = np.zeros((seq_len, d_model), dtype=float)
    
    # 6. Apply sine to even dimension indices (0, 2, 4...)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    
    # 7. Apply cosine to odd dimension indices (1, 3, 5...)
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    
    return pe