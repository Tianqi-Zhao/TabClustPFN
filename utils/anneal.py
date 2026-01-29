
def annealed_params(anneal_proportion, initial, final, curr_step, max_steps) -> float:
    """
    Calculates the annealed temperature (tau) for Gumbel-Softmax.

    Performs exponential annealing based on the current training step.

    Parameters
    ----------
    anneal_proportion : float
        The proportion of training steps over which to anneal the temperature.
    initial : float
        The initial temperature (tau) value.
    final : float
        The final temperature (tau) value.
    curr_step : int
        The current training step.
    max_steps : int
        The maximum number of training steps.

    Returns
    -------
    float
        The calculated tau value.
    """
    if anneal_proportion <= 0.0:
        return initial

    anneal_proportion = max(0.0, min(1.0, anneal_proportion))  # Clamp to [0, 1]
    anneal_steps = int(max_steps * anneal_proportion)

    if curr_step < anneal_steps:
        # Exponentially anneal tau
        params = initial * (final / initial) ** (curr_step / anneal_steps)
    else:
        params = final

    return params