from .net2mip import Net2LPExpected, Net2MIPExpected


def factory_model2mip(model_type):
    if model_type == 'icnn_e':
        return Net2LPExpected
    elif model_type == 'nn_e':
        return Net2MIPExpected
    else:
        raise ValueError('Invalid model type!')
