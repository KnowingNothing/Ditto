def trisolv(N):
  """
  probably buggy
  """
  
  L = tvm.te.placeholder([N, N], name="L", dtype="float32")
  b = tvm.te.placeholder([N], name="b", dtype="float32")
  init = tvm.te.compute([1, N], lambda t, i: b[i], name='init')
  def foo(t, i):
    tmp = state[t-1, i]
    reduce_axis = tvm.te.reduce_axis([0, i-1], name="reduce_axis")
    row = tvm.te.compute([N], lambda j: -L[i, j]*state[t-1, j])
    row = tvm.te.sum(row, axis = reduce_axis)
    tmp = tmp - row
    return tmp/L[i, i]
  update = tvm.te.compute([N, N], foo, name='update')
  state = tvm.te.placeholder([N], name="state", dtype="float32")
  x = tvm.scan(init, update, state)
  return [x], [L, b]