from contextlib import contextmanager

def _remove_ddp(model):
    from torch.nn.parallel import DistributedDataParallel

    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


@contextmanager
def tune_nms_threshold_and_restore(model, temp_threshold):
    """Apply ema stored in `model` to model and returns a function to restore
    the weights are applied
    """
    model = _remove_ddp(model)
    old_nms_threshold = model.nms_thresh
    print(f"Changing NMS threshold from {old_nms_threshold} to {temp_threshold}")
    model.nms_thresh = temp_threshold

    yield model

    print(f"Restoring NMS threshold from {temp_threshold} to {old_nms_threshold}")
    model.nms_thresh = old_nms_threshold
