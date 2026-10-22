"""Unit tests for FFT operators on MUSA backend (compare with CPU results)."""

import pytest
import torch

real_dtypes = [torch.float32, torch.float64]
complex_dtypes = [torch.complex64, torch.complex128]
norm_modes = [None, "forward", "backward", "ortho"]

fft_inputs = [
    torch.randn(8),
    torch.randn(3, 5),
    torch.randn(4, 8),
    torch.randn(2, 4, 8),
    torch.randn(2, 4, 6, 8),
]


@pytest.mark.parametrize("dtype", complex_dtypes+real_dtypes)
@pytest.mark.parametrize("norm", norm_modes)
@pytest.mark.parametrize("input_data", fft_inputs)
def test_fft(dtype, norm, input_data):
    """Compare MUSA fft result with CPU reference."""
    cpu_input = input_data.to(dtype)
    ref = torch.fft.fft(cpu_input, norm=norm)

    musa_input = cpu_input.to("musa")
    musa_out = torch.fft.fft(musa_input, norm=norm).cpu()

    assert torch.allclose(ref, musa_out, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", complex_dtypes+real_dtypes)
@pytest.mark.parametrize("norm", norm_modes)
@pytest.mark.parametrize("input_data", fft_inputs)
def test_ifft(dtype, norm, input_data):
    """Compare MUSA ifft result with CPU reference."""
    cpu_input = input_data.to(dtype)
    ref = torch.fft.ifft(cpu_input, norm=norm)

    musa_input = cpu_input.to("musa")
    musa_out = torch.fft.ifft(musa_input, norm=norm).cpu()

    assert torch.allclose(ref, musa_out, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", real_dtypes)
@pytest.mark.parametrize("norm", norm_modes)
@pytest.mark.parametrize("input_data", fft_inputs)
def test_rfft(dtype, norm, input_data):
    """Compare MUSA rfft result with CPU reference."""
    cpu_input = input_data.to(dtype)
    ref = torch.fft.rfft(cpu_input, norm=norm)

    musa_input = cpu_input.to("musa")
    musa_out = torch.fft.rfft(musa_input, norm=norm).cpu()

    assert torch.allclose(ref, musa_out, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", real_dtypes)
@pytest.mark.parametrize("norm", norm_modes)
@pytest.mark.parametrize("input_data", fft_inputs)
def test_irfft(dtype, norm, input_data):
    """Compare MUSA irfft result with CPU reference."""
    cpu_input = input_data.to(dtype)
    freq = torch.fft.rfft(cpu_input, norm=norm)
    ref = torch.fft.irfft(freq, n=cpu_input.shape[-1], norm=norm)

    musa_input = freq.to("musa")
    musa_out = torch.fft.irfft(musa_input, n=cpu_input.shape[-1], norm=norm).cpu()

    assert torch.allclose(ref, musa_out, atol=1e-4, rtol=1e-4)


def test_irfft_graph_capture():
    """MUSA irfft must remain valid when the caller captures a graph."""
    n = 16
    freq = torch.randn(2, n // 2 + 1, 4, device="musa", dtype=torch.complex64)
    expected = torch.fft.irfft(freq, n=n, dim=1)

    for _ in range(3):
        torch.fft.irfft(freq, n=n, dim=1)
    torch.musa.synchronize()

    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        output = torch.fft.irfft(freq, n=n, dim=1)
    graph.replay()
    torch.musa.synchronize()

    torch.testing.assert_close(output, expected)
