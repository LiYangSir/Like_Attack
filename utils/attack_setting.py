from utils.gradient_strategy import CenterConvGenerator, RandomGenerator, DCTGenerator, UpSampleGenerator, KLGenerator


def load_pgen(task, pgen_type, constraint):
    print(f"Gradient Strategy : {pgen_type}")
    if task == 'imagenet':
        if pgen_type == 'naive':
            p_gen = None
        elif pgen_type == 'upsample':
            p_gen = UpSampleGenerator(factor=4.0)
        elif pgen_type == 'DCT':
            p_gen = DCTGenerator(factor=4.0)
        elif pgen_type == 'random':
            p_gen = RandomGenerator(constraint)
        elif pgen_type == 'centerconv':
            p_gen = CenterConvGenerator(4., 56, 56)
        elif pgen_type == 'kl':
            p_gen = KLGenerator(2., 14, 14)

    elif task.startswith('cifar'):
        if pgen_type == 'naive':
            p_gen = None
        elif pgen_type == 'upsample':
            p_gen = UpSampleGenerator(factor=2.0)
        elif pgen_type == 'DCT':
            p_gen = DCTGenerator(factor=2.0)
        elif pgen_type == 'random':
            p_gen = RandomGenerator(constraint)
        elif pgen_type == 'centerconv':
            p_gen = CenterConvGenerator(2., 16, 16)
        elif pgen_type == 'kl':
            p_gen = KLGenerator(2., 14, 14)
    else:
        if pgen_type == 'naive':
            p_gen = None
        elif pgen_type == 'upsample':
            p_gen = UpSampleGenerator(factor=2.0)
        elif pgen_type == 'DCT':
            p_gen = DCTGenerator(factor=1.6)
        elif pgen_type == 'random':
            p_gen = RandomGenerator(constraint)
        elif pgen_type == 'centerconv':
            p_gen = CenterConvGenerator(2., 14, 14)
        elif pgen_type == 'kl':
            p_gen = KLGenerator(2., 14, 14)
    return p_gen
