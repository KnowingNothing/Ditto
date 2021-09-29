### GEMM
$D[i, j] = alpha * (\sum_k{A[i, k] * B[k, j]}) + beta * C[i, j]$

### GEMVER
$B[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]\\ W[i] = w[i] + alpha * (\sum_{k}{B[i, k] * (x[i] + beta * (\sum_l{B[k, j] * y[l]}) + z[i])})$

### GESUMMV
$y[i] = alpha * (\sum_{k}{A[i, k] * x[k]}) + beta * (\sum_{l}{B[i, l] * x[l]})$

### SYMM
$D[i, j] = beta * C[i, j] + alpha * B[i, j] * A[i, i] + alpha * (\sum_k(A[i, k] * B[k, j]))\\ state[t, i, j] =\\ \ \ \ \ \ \text{ if } (0 <= i < t) \text{ then } state[t-1, i, j] + alpha * B[t, j] * A[t, i]\\ \ \ \ \ \ \text{ else if } (i >= t > 0) state[t-1, i, j]\\ \ \ \ \ \ \text{ else } D[i, j]\\ result[i, j] = state[T-1, i, j]$

### SYR2K
$D[i, j] = \text{ if } (j <= i) \text{ then } C[i, j] * beta \text{ else } C[i, j]\\ E[i, j] = \sum_k{A[j, k] * B[i, k] + B[j, k] * A[i, k]}\\ F[i, j] = \text{ if } (j <= i) \text{ then } D[i, j] + alpha * E[i, j] \text{ else } D[i, j]$

### SYRK
$D[i, j] = \text{ if } (j <= i) \text{ then } C[i, j] * beta \text{ else } C[i, j]\\ E[i, j] = \sum_k{A[j, k] * A[i, k]\\ F[i, j] = \text{ if } (j <= i) \text{ then } D[i, j] + alpha * E[i, j] \text{ else } D[i, j]$

### TRMM
$C[i, j] = alpha * (B[i, j] + (\sum_{k > i}{A[k, i] * B[k, j]}))$