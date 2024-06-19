from collections import namedtuple

import torch

# LensInfo = namedtuple("LensInfo", "lens", "is_absolute")
# #FIXME create an object to handle easily absolute and relative padding lens


def pad_right_to(
    tensor: torch.Tensor,
    target_shape: (list, tuple),
    mode="constant",
    value=-1,
    absolute_len=False,
):
    """
    This function takes a torch tensor of arbitrary shape and pads it to target
    shape by appending values on the right.
    Parameters
    ----------
    tensor : input torch tensor
        Input tensor whose dimension we need to pad.
    target_shape : (list, tuple)
        Target shape we want for the target tensor its len must be equal to tensor.ndim
    mode : str
        Pad mode, please refer to torch.nn.functional.pad documentation.
    value : float
        Pad value, please refer to torch.nn.functional.pad documentation.
    Returns
    -------
    tensor : torch.Tensor
        Padded tensor.
    valid_vals : list
        List containing proportion for each dimension of original, non-padded values.
    """
    assert len(target_shape) == tensor.ndim
    pads = []  # this contains the abs length of the padding for each dimension.
    valid_vals = []  # this contains the relative lengths for each dimension.
    i = len(target_shape) - 1  # iterating over target_shape ndims
    j = 0
    while i >= 0:
        assert (
            target_shape[i] >= tensor.shape[i]
        ), "Target shape must be >= original shape for every dim"
        pads.extend([0, target_shape[i] - tensor.shape[i]])
        if absolute_len:
            valid_vals.append(tensor.shape[j])
        else:
            valid_vals.append(tensor.shape[j] / target_shape[j])
        i -= 1
        j += 1

    tensor = torch.nn.functional.pad(tensor, pads, mode=mode, value=value)

    return tensor, valid_vals


def batch_pad_right(tensors: list, mode="constant", value=0, absolute_len=False):
    """Given a list of torch tensors it batches them together by padding to the right
    on each dimension in order to get same length for all.
    Parameters
    ----------
    tensors : list
        List of tensor we wish to pad together.
    mode : str
        Padding mode see torch.nn.functional.pad documentation.
    value : float
        Padding value see torch.nn.functional.pad documentation.
    Returns
    -------
    tensor : torch.Tensor
        Padded tensor.
    valid_vals : list
        List containing proportion for each dimension of original, non-padded values.
    """

    if not len(tensors):
        raise IndexError("Tensors list must not be empty")

    if len(tensors) == 1:
        # if there is only one tensor in the batch we simply unsqueeze it.
        return tensors[0].unsqueeze(0), torch.stack(
            [torch.tensor(1.0) for x in range(tensors[0].ndim)]
        ).unsqueeze(0)

    if not (any([tensors[i].ndim == tensors[0].ndim for i in range(1, len(tensors))])):
        raise IndexError("All tensors must have same number of dimensions")
    max_shape = []
    for dim in range(tensors[0].ndim):
        # compared to speechbrain, here no if (my original implementation).
        # this enables to pad tensors with arbitrary number of dimensions.
        max_shape.append(max([x.shape[dim] for x in tensors]))

    batched = []
    valid = []
    for t in tensors:
        # for each tensor we apply pad_right_to
        padded, valid_percent = pad_right_to(
            t, max_shape, mode=mode, value=value, absolute_len=absolute_len
        )
        batched.append(padded)
        valid.append(valid_percent)
    batched = torch.stack(batched)
    return batched, torch.tensor(valid)
