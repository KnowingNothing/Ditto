from . import _ffi_api


def grad_layer(layer):
    """
    get the gradient of a layer
    
    Args:
    ---
    layer: ditto.auto_compute.Layer
    
    Returns:
    ---
    ditto.auto_compute.Layer
    """
    return _ffi_api.GradLayer(layer)