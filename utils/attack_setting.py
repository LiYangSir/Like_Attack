from utils.gradient_strategy import ResizeGenerator, RandomGenerator, DCTGenerator


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

    elif task.startswith('cifar'):
        if pgen_type == 'naive':
            p_gen = None
        elif pgen_type == 'resize':
            p_gen = ResizeGenerator(factor=2.0)
        elif pgen_type == 'DCT':
            p_gen = DCTGenerator(factor=2.0)
        elif pgen_type == 'random':
            p_gen = RandomGenerator(constraint)
    else:
        if pgen_type == 'naive':
            p_gen = None
        elif pgen_type == 'resize':
            p_gen = ResizeGenerator(factor=2.0)
        elif pgen_type == 'DCT':
            p_gen = DCTGenerator(factor=1.6)
        elif pgen_type == 'random':
            p_gen = RandomGenerator(constraint)
    return p_gen
