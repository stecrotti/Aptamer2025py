import torch

def read_multinomial(population, total_reads, sequences, return_idx = False):
    reads_idx = []
    for t in range(len(population)):
        reads_t = torch.multinomial(population[t].to(torch.float), total_reads[t], replacement=True)
        reads_idx.append(reads_t)
    sequences_oh = [sequences[id] for id in reads_idx]
    
    if return_idx:
        return sequences_oh, reads_idx
    else:
        return sequences_oh

def read_negbin(population, total_reads, sequences_unique, r):
    sequences_oh = []
    for t in range(len(population)):
        normalized_population_t = population[t] / population[t].sum()
        poisson_mean = normalized_population_t * total_reads[t]
        rt = torch.full(poisson_mean.size(), r)
        logits_t = torch.log(poisson_mean / r)
        read_counts_t = torch.distributions.negative_binomial.NegativeBinomial(rt, logits=logits_t).sample().to(torch.int)
        if not torch.all(read_counts_t >= 0):
            print(f'Found some negative counts when reading: {read_counts_t[read_counts_t<0]}')
        if torch.all(read_counts_t == 0):
            print(f'All read counts equal to zero at round {t}')
        sequences_t = torch.repeat_interleave(sequences_unique, read_counts_t, dim=0)
        sequences_oh.append(sequences_t)

    return sequences_oh

def ReadNegBin(r):
    def _read(*args):
        return read_negbin(*args, r)
    return _read

def read_poisson(population, total_reads, sequences):
    sequences_oh = []
    for t in range(len(population)):
        normalized_population_t = population[t] / population[t].sum()
        poisson_mean = normalized_population_t * total_reads[t]
        read_counts_t = torch.poisson(poisson_mean).to(torch.int)
        sequences_t = torch.repeat_interleave(sequences, read_counts_t, dim=0)
        sequences_oh.append(sequences_t)

    return sequences_oh

def generate_realistic(
    sample_round_zero,
    ps_round_t,
    initial_pop_size: int,
    n_selection_rounds: int,
    verbose = True
):
    seq_round_zero = sample_round_zero(initial_pop_size)
    if verbose: print("Extracting unique sequences and counts from initial library...")
    sequences, counts_round_zero = torch.unique(seq_round_zero, dim=0, return_counts=True)
    ps = [ps_round_t(sequences, t) for t in range(n_selection_rounds)]
    if not torch.all(torch.tensor([torch.all(pst <= 1) for pst in ps])):
        raise ValueError(f'Got some selection probabilities > 1')
    population = [counts_round_zero.to(torch.int64)]
    n_rounds = n_selection_rounds + 1
    for t in range(1, n_rounds):
        if verbose: print(f"Starting selection round {t} of {n_rounds-1}...")
        d = torch.distributions.Binomial(population[t-1], ps[t-1])
        selected = d.sample()
        if not torch.all(selected >= 0):
            raise ValueError("Binomial sampled negative values")
        amplification = 1 / ps[t-1].mean()
        population_t = (selected * amplification).to(torch.int64)
        if not torch.all(population_t >= 0):
            raise ValueError("Negative population value")
        if torch.all(population_t == 0):
            raise ValueError(f"Nothing was selected at round {t}")
        population.append(population_t)

    return population, sequences
    
def sample_realistic(
    sample_round_zero,
    ps_round_t,
    initial_pop_size: int,
    total_reads: torch.tensor,
    read = read_multinomial,
    verbose = True
):
    n_selection_rounds = len(total_reads - 1)
    population, sequences = generate_realistic(sample_round_zero, ps_round_t, initial_pop_size,
        n_selection_rounds, verbose = verbose)

    sequences_oh = read(population, total_reads, sequences)
    
    return sequences, population, sequences_oh