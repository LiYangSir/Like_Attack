from utils.gradient_strategy import ResizeGenerator, RandomGenerator, DCTGenerator, CenterConvGenerator


def load_pgen(task, pgen_type, constraint):
    print(f"Gradient Strategy : {pgen_type}")
    if task == 'imagenet':
        if pgen_type == 'naive':
            p_gen = None
        elif pgen_type == 'resize':
            p_gen = ResizeGenerator(factor=4.0)
        elif pgen_type == 'DCT':
            p_gen = DCTGenerator(factor=4.0)
        elif pgen_type == 'random':
            p_gen = RandomGenerator(constraint)
        elif pgen_type == 'center':
            p_gen = CenterConvGenerator(2., 112, 112)
    elif task.startswith('cifar'):
        if pgen_type == 'naive':
            p_gen = None
        elif pgen_type == 'resize':
            p_gen = ResizeGenerator(factor=2.0)
        elif pgen_type == 'DCT':
            p_gen = DCTGenerator(factor=2.0)
        elif pgen_type == 'random':
            p_gen = RandomGenerator(constraint)
        elif pgen_type == 'center':
            p_gen = CenterConvGenerator(2., 16, 16)
    else:
        if pgen_type == 'naive':
            p_gen = None
        elif pgen_type == 'resize':
            p_gen = ResizeGenerator(factor=2.0)
        elif pgen_type == 'DCT':
            p_gen = DCTGenerator(factor=1.6)
        elif pgen_type == 'random':
            p_gen = RandomGenerator(constraint)
        elif pgen_type == 'center':
            p_gen = CenterConvGenerator(2., 14, 14)
    return p_gen
