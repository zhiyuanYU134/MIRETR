import torch


def log_sinkhorn_normalization(scores, log_mu, log_nu, num_iter):
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(num_iter):
        u = log_mu - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(scores + u.unsqueeze(2), dim=1)
    return scores + u.unsqueeze(2) + v.unsqueeze(1)
