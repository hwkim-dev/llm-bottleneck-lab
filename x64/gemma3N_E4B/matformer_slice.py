import numpy as np

def slice_weight_int4(w_tuple, row_slice=None, col_slice_half=None):
    packed, scale = w_tuple
    if row_slice is not None:
        packed = packed[:row_slice, :]
        scale = scale[:row_slice]
    if col_slice_half is not None:
        packed = packed[:, :col_slice_half]
        # scale stays the same since it's per-row
    
    # We must ensure the returned packed array is contiguous because Vulkan/C++ might require it
    return (np.ascontiguousarray(packed), np.ascontiguousarray(scale))

def slice_weight_fp(w, row_slice=None, col_slice=None):
    if row_slice is not None:
        w = w[:row_slice, :]
    if col_slice is not None:
        w = w[:, :col_slice]
    return np.ascontiguousarray(w)

def slice_to_e2b(W: dict, num_layers=35, intermediate_size=8192) -> dict:
    """
    Slices the E4B weights to MatFormer E2B weights.
    By default:
      - keeps `num_layers` (default 35, or could be 17/18 for smaller drafts)
      - shrinks intermediate size of FFN from 16384 to 8192.
    """
    W_sliced = {}
    
    # Just copy over anything that is not layer-dependent, or doesn't need slicing
    for k, v in W.items():
        W_sliced[k] = v
        
        if k in ["W_gate", "W_up", "W_down"]:
            # Need to slice these lists of weights
            new_list = []
            for i in range(num_layers):
                w = W[k][i]
                is_int4 = isinstance(w, tuple)
                if k in ["W_gate", "W_up"]:
                    # output dim is reduced
                    if is_int4:
                        new_list.append(slice_weight_int4(w, row_slice=intermediate_size))
                    else:
                        new_list.append(slice_weight_fp(w, row_slice=intermediate_size))
                elif k == "W_down":
                    # input dim is reduced
                    if is_int4:
                        new_list.append(slice_weight_int4(w, col_slice_half=intermediate_size // 2))
                    else:
                        new_list.append(slice_weight_fp(w, col_slice=intermediate_size))
            W_sliced[k] = new_list
        elif isinstance(v, list) and len(v) == 35:
            # For other layer-specific weights, just take the first `num_layers`
            W_sliced[k] = v[:num_layers]
            
    return W_sliced
