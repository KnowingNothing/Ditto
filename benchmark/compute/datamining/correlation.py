def correlation(N, M):
  data = tvm.te.placeholder([N, M], name='data', dtype='float32')
  reduce_N = tvm.te.reduce_axis([0, N], name="reduce_N")
  mean = tvm.te.compute([M], lambda m: tvm.te.sum(data[reduce_N, m])/N)
  std = tvm.te.compute([M], lambda m: (tvm.te.sum(
    (data[reduce_N, m]-mean[m])**2
  ))**0.5)
  normalized = tvm.compute([N, M], lambda n, m: (data[n, m]-mean[m])/std[m])
  corr = tvm.compute([M, M], lambda i, j: (tvm.sum(data[reduce_N, i]*data[reduce_N, j]))/N)
  return [corr],  [data]


def covariance(N, M):
  data = tvm.te.placeholder([N, M], name='data', dtype='float32')
  reduce_N = tvm.te.reduce_axis([0, N], name="reduce_N")
  mean = tvm.te.compute([M], lambda m: tvm.te.sum(data[reduce_N, m])/N)
  normalized = tvm.compute([N, M], lambda n, m: data[n, m]-mean[m])
  corr = tvm.compute([M, M], lambda i, j: (tvm.sum(data[reduce_N, i]*data[reduce_N, j]))/N)
  return [covariance],  [data]


