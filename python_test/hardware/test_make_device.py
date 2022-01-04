import pytest
import tvm
from ditto import hardware as hw


@pytest.mark.basic
def test_make_unit():
    """Make a hardware unit
    """
    isa = {
        "test.fma.fp32.fp32.fp32": hw.visa.scalar_multiply_add(
            2.5, tvm.runtime.DataType("float32"), "float32", "float32", name="test.fma.fp32.fp32.fp32")
    }
    unit = hw.hw_unit(isa, name="test_unit")
    print(unit)
    print(unit.isa_list)
    print(unit.isa_list["test.fma.fp32.fp32.fp32"])


@pytest.mark.basic
def test_make_local_mem():
    """Make a hardware local mem
    """
    pattern = {
        "scalar_fp32": hw.pattern.scalar_pattern(
            "float32", "", name="scalar_fp32")
    }
    mem = hw.hw_local_mem(32, pattern, name="test_mem")
    print(mem)
    print(mem.pattern_list)
    print(mem.pattern_list["scalar_fp32"])


@pytest.mark.basic
def test_make_heteroprocessor():
    """Make a hardware heterogenous processor
    """
    isa = {
        "test.fma.fp32.fp32.fp32": hw.visa.scalar_multiply_add(
            2.5, tvm.runtime.DataType("float32"), "float32", "float32", name="test.fma.fp32.fp32.fp32")
    }
    unit = hw.hw_unit(isa, name="test_unit")

    pattern = {
        "scalar_fp32": hw.pattern.scalar_pattern(
            "float32", "", name="scalar_fp32")
    }
    mem = hw.hw_local_mem(32, pattern, name="test_mem")

    topology = {
        unit: {
            mem: hw.compute_path(isa["test.fma.fp32.fp32.fp32"],
                                 pattern["scalar_fp32"], hw.visa.direct(), hw.visa.direct())
        }
    }

    proc = hw.hw_heteroprocessor([unit], [mem], topology)
    print(proc)
    print(proc.units)
    print(proc.local_mems)


@pytest.mark.basic
def test_make_homogroup():
    """Make a hardware homogeneous group
    """
    isa = {
        "test.fma.fp32.fp32.fp32": hw.visa.scalar_multiply_add(
            2.5, tvm.runtime.DataType("float32"), "float32", "float32", name="test.fma.fp32.fp32.fp32")
    }
    unit = hw.hw_unit(isa, name="test_unit")

    pattern = {
        "scalar_fp32": hw.pattern.scalar_pattern(
            "float32", "", name="scalar_fp32")
    }
    mem = hw.hw_local_mem(32, pattern, name="test_mem")

    topology = {
        unit: {
            mem: hw.compute_path(isa["test.fma.fp32.fp32.fp32"],
                                 pattern["scalar_fp32"], hw.visa.direct(), hw.visa.direct())
        }
    }

    proc = hw.hw_heteroprocessor([unit], [mem], topology, name="test_proc")

    shared_mem = hw.hw_shared_mem(128, pattern, name="test_shared")

    group = hw.hw_homogroup(proc, shared_mem, 4, name="test_group")
    print(group)
    print(group.processor)
    print(group.block_x)


@pytest.mark.basic
def test_make_device():
    """Make a hardware device
    """
    isa = {
        "test.fma.fp32.fp32.fp32": hw.visa.scalar_multiply_add(
            2.5, tvm.runtime.DataType("float32"), "float32", "float32", name="test.fma.fp32.fp32.fp32")
    }
    unit = hw.hw_unit(isa, name="test_unit")

    pattern = {
        "scalar_fp32": hw.pattern.scalar_pattern(
            "float32", "", name="scalar_fp32")
    }
    mem = hw.hw_local_mem(32, pattern, name="test_mem")

    topology = {
        unit: {
            mem: [hw.compute_path(isa["test.fma.fp32.fp32.fp32"],
                                  pattern["scalar_fp32"], hw.visa.direct(), hw.visa.direct())]
        }
    }

    proc = hw.hw_heteroprocessor([unit], [mem], topology, name="test_proc")
    print(proc.topology)

    shared_mem = hw.hw_shared_mem(128, pattern, name="test_shared")

    group = hw.hw_homogroup(proc, shared_mem, 4, name="test_group")

    global_mem = hw.hw_global_mem(40, pattern, name="test_global")
    print(global_mem.kb)

    device = hw.hw_device(group, global_mem, 108, name="test_device")
    print(device)
    print(device.group)
    print(device.grid_x)


if __name__ == "__main__":
    test_make_unit()
    test_make_local_mem()
    test_make_heteroprocessor()
    test_make_homogroup()
    test_make_device()
