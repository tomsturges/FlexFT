# Changelog

## 0.2.0 - 2026-08-24

### Breaking changes

- Rename the 2D shape parameters and attributes from input/output-oriented
  `N`/`M` to domain-oriented `Nx`/`Nk`. Forward transforms map `Nx` to `Nk`;
  inverse transforms map `Nk` to `Nx`.
- Move ordinary FFT construction to the `FlexFT.fft`, `IFlexFT.fft`,
  `FlexFT2D.fft`, and `IFlexFT2D.fft` class factories. Flexible-grid
  constructors now accept only `"direct"` and `"bluestein"` methods.
- Require transform spacings explicitly in the flexible-grid APIs.

### Added

- Support independently sized input and output grids in fractional and
  flexible Fourier transforms.
- Add direct centered fractional-DFT evaluation alongside Bluestein
  convolution.
- Add `recommend_method` for choosing between direct and Bluestein evaluation
  using estimates or device benchmarks.
- Add reusable ordinary FFT plans with shifted-grid support in one and two
  dimensions.

### Changed

- Choose the first 2D transform axis according to intermediate-array size.
- Simplify 2D parameter handling and delegate per-axis validation to the 1D
  plans.
