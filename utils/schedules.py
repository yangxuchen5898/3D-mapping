def schedule_lambda2(iteration, warmup_iters, ramp_iters, lambda2_max):
    """
    Schedules the alignment loss weight lambda2 over training iterations.
    """
    if iteration < warmup_iters:
        return 0.0
    elif iteration < warmup_iters + ramp_iters:
        progress = (iteration - warmup_iters) / float(ramp_iters)
        return lambda2_max * progress
    else:
        return lambda2_max
