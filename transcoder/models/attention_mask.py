import torch


def get_attention_mask(sequence_length, device, mask_type="block-causal", **kwargs):
    if mask_type.lower() == 'none' or mask_type is None:
        return None
    elif mask_type.lower() == 'block-causal':
        return _block_caulsal_mask_impl(sequence_length, device, **kwargs)
    elif mask_type.lower() == 'causal':
        return _caulsal_mask_impl(sequence_length, device, **kwargs)
    else:
        raise NotImplementedError(f"Mask type {mask_type} not implemented")


def _block_caulsal_mask_impl(sequence_length, device, block_size=16, **kwargs):
    """
    Create a block-causal mask
    """
    assert sequence_length % block_size == 0, "for block causal masks sequence length must be divisible by block size"
    blocks = torch.ones(sequence_length // block_size, block_size, block_size, device=device)
    block_diag_enable_mask = torch.block_diag(*blocks)
    causal_enable_mask = torch.ones(sequence_length, sequence_length, device=device).tril_(0)
    disable_mask = ((block_diag_enable_mask + causal_enable_mask) < 0.5)
    return disable_mask


def _caulsal_mask_impl(sequence_length, device, **kwargs):
    """
    Create a causal mask
    """
    causal_disable_mask = torch.triu(
        torch.full((sequence_length, sequence_length), float('-inf'), dtype=torch.float32, device=device),
        diagonal=1,
    )
    return causal_disable_mask


if __name__ == '__main__':
    mask = get_attention_mask(9, "cuda", mask_type="block-causal", block_size=3)
    print(mask)
    mask = get_attention_mask(9, "cuda", mask_type="causal")
    print(mask)