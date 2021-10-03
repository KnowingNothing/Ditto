### ATAX
$y[j] = \sum_i{(\sum_k{A[i, k] * x[k]}) * A[i, j]}$

### 2MM
$F[i, j] = alpha * (\sum_l{(\sum_k{A[i, k] * B[k, l]}) * C[l, j]}) + beta * D[i, j]$

### 3MM
$G[i, j] = \sum_r{(\sum_k{A[i, k] * B[k, r]}) * (\sum_l{C[r, l] * D[l, j]})}$

### BICG
$x[i] = \sum_k{A[i, k] * p[k]}\\y[j] = \sum_l{A[l, j] * q[l]}$

### DOITGEN
$S[r, q, p] = \sum_k{A[r, q, k] * B[k, p]}$

### MVT
$x[i] = \sum_k{A[i, k] * p[k]}\\ y[j] = \sum_l{A[l, j] * q[l]}$