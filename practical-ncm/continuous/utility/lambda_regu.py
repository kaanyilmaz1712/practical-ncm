import math

def lambda_regu(epoch, total_epochs, lambda_start=1, lambda_end=0.001):
    # exponentially falling λ
    decay = math.log(lambda_end / lambda_start) / total_epochs
    return lambda_start * math.exp(decay * epoch)