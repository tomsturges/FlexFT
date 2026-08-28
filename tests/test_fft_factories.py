import inspect
import unittest

import jax.numpy as jnp

from flexft import (
    CenteredDFT,
    FlexFT,
    FlexFT2D,
    IFlexFT,
    IFlexFT2D,
    flexft,
    flexft2d,
    iflexft2d,
)


class FFTFactoryTests(unittest.TestCase):
    def test_forward_plan_exposes_its_physical_grids(self):
        transform = FlexFT(
            N=5,
            M=4,
            dx=0.2,
            dk=0.3,
            x0=1.0,
            k0=-0.5,
        )

        self.assertTrue(
            jnp.allclose(transform.x, jnp.array([0.6, 0.8, 1.0, 1.2, 1.4]))
        )
        self.assertTrue(
            jnp.allclose(transform.k, jnp.array([-1.1, -0.8, -0.5, -0.2]))
        )

    def test_shifted_plan_matches_sum_on_absolute_coordinates(self):
        samples = jnp.array([1.0 + 0.2j, -0.4j, 0.3 - 0.7j, 1.2j, -0.8])
        arguments = dict(
            N=5,
            M=4,
            dx=0.17,
            dk=0.11,
            x0=0.37,
            k0=-0.23,
        )

        for method in ("direct", "bluestein"):
            with self.subTest(method=method):
                transform = FlexFT(**arguments, method=method)
                kernel = jnp.exp(-2j * jnp.pi * jnp.outer(transform.k, transform.x))
                expected = transform.dx * kernel @ samples

                self.assertTrue(
                    jnp.allclose(
                        transform(samples), expected, rtol=2e-5, atol=2e-5
                    )
                )

    def test_inverse_plan_grids_keep_domain_names(self):
        transform = IFlexFT(
            N=4,
            M=5,
            dk=0.3,
            dx=0.2,
            x0=1.0,
            k0=-0.5,
        )

        self.assertEqual(transform.x.shape, (5,))
        self.assertEqual(transform.k.shape, (4,))
        self.assertTrue(
            jnp.allclose(transform.x, jnp.array([0.6, 0.8, 1.0, 1.2, 1.4]))
        )
        self.assertTrue(
            jnp.allclose(transform.k, jnp.array([-1.1, -0.8, -0.5, -0.2]))
        )

    def test_2d_plans_expose_axis_coordinate_vectors(self):
        forward = FlexFT2D(
            Nx=(3, 4),
            Nk=(2, 5),
            dx=(0.2, 0.4),
            dk=(0.5, 0.25),
            x0=(1.0, -1.0),
            k0=(0.5, 2.0),
        )
        inverse = IFlexFT2D(
            Nk=(2, 5),
            Nx=(3, 4),
            dk=(0.5, 0.25),
            dx=(0.2, 0.4),
            x0=(1.0, -1.0),
            k0=(0.5, 2.0),
        )

        for plan in (forward, inverse):
            self.assertEqual(tuple(axis.shape for axis in plan.x), ((3,), (4,)))
            self.assertEqual(tuple(axis.shape for axis in plan.k), ((2,), (5,)))
            self.assertTrue(jnp.allclose(plan.x[0], jnp.array([0.8, 1.0, 1.2])))
            self.assertTrue(
                jnp.allclose(plan.x[1], jnp.array([-1.8, -1.4, -1.0, -0.6]))
            )
            self.assertTrue(jnp.allclose(plan.k[0], jnp.array([0.0, 0.5])))
            self.assertTrue(
                jnp.allclose(
                    plan.k[1], jnp.array([1.5, 1.75, 2.0, 2.25, 2.5])
                )
            )

    def test_flexible_constructor_keeps_direct_and_bluestein_equivalent(self):
        N, M = 11, 3
        samples = jnp.arange(N) + 1j * jnp.arange(N)[::-1]
        arguments = dict(
            N=N,
            M=M,
            dx=0.13,
            dk=0.07,
            x0=0.2,
            k0=-0.3,
        )

        bluestein = FlexFT(**arguments, method="bluestein")
        direct = FlexFT(**arguments, method="direct")

        self.assertEqual(bluestein.method, "bluestein")
        self.assertEqual(direct.method, "direct")
        self.assertTrue(
            jnp.allclose(bluestein(samples), direct(samples), rtol=2e-5, atol=2e-5)
        )

    def test_forward_factory_has_constrained_signature_and_attributes(self):
        parameters = inspect.signature(FlexFT.fft).parameters
        self.assertEqual(set(parameters), {"N", "dx", "x0", "k0"})

        transform = FlexFT.fft(N=8, dx=0.25, x0=0.3, k0=-0.4)
        self.assertEqual(transform.N, 8)
        self.assertEqual(transform.M, 8)
        self.assertEqual(transform.dx, 0.25)
        self.assertEqual(transform.dk, 0.5)
        self.assertEqual(transform.x0, 0.3)
        self.assertEqual(transform.k0, -0.4)
        self.assertEqual(transform.method, "fft")
        self.assertIsInstance(transform.core, CenteredDFT)
        self.assertTrue(
            jnp.allclose(
                transform.x,
                0.3 + (jnp.arange(8) - 4) * 0.25,
            )
        )
        self.assertTrue(
            jnp.allclose(
                transform.k,
                -0.4 + (jnp.arange(8) - 4) * 0.5,
            )
        )

    def test_fft_matches_direct_sum_on_derived_grid_with_shifted_centres(self):
        for N in (7, 8):
            with self.subTest(N=N):
                dx = 0.19
                dk = 1.0 / (N * dx)
                x0, k0 = 0.31, -0.23
                samples = jnp.arange(N) + 1j * jnp.arange(N)[::-1]

                fft_transform = FlexFT.fft(N=N, dx=dx, x0=x0, k0=k0)
                direct_transform = FlexFT(
                    N=N,
                    M=N,
                    dx=dx,
                    dk=dk,
                    method="direct",
                    x0=x0,
                    k0=k0,
                )

                self.assertTrue(
                    jnp.allclose(
                        fft_transform(samples),
                        direct_transform(samples),
                        rtol=2e-5,
                        atol=2e-5,
                    )
                )

    def test_inverse_fft_factory_round_trips_shifted_grid(self):
        N = 9
        dx = 0.17
        x0, k0 = -0.21, 0.37
        samples = jnp.arange(N) + 1j * jnp.arange(N)[::-1]
        forward = FlexFT.fft(N=N, dx=dx, x0=x0, k0=k0)
        inverse = IFlexFT.fft(N=N, dk=forward.dk, x0=x0, k0=k0)

        self.assertEqual(inverse.method, "fft")
        self.assertAlmostEqual(inverse.dx, dx)
        self.assertTrue(
            jnp.allclose(inverse(forward(samples)), samples, rtol=3e-5, atol=3e-5)
        )

    def test_2d_fft_factories_match_direct_and_round_trip(self):
        shape = (5, 6)
        dx = (0.17, 0.23)
        dk = tuple(1.0 / (N * spacing) for N, spacing in zip(shape, dx))
        x0 = (0.2, -0.1)
        k0 = (-0.3, 0.4)
        samples = jnp.arange(shape[0] * shape[1], dtype=float).reshape(shape) + 1j

        forward = FlexFT2D.fft(Nx=shape, dx=dx, x0=x0, k0=k0)
        direct = FlexFT2D(
            Nx=shape,
            dx=dx,
            dk=dk,
            method="direct",
            x0=x0,
            k0=k0,
        )
        transformed = forward(samples)

        self.assertEqual(forward.Nk, shape)
        self.assertEqual(forward.method, ("fft", "fft"))
        self.assertTrue(
            jnp.allclose(transformed, direct(samples), rtol=4e-5, atol=4e-5)
        )

        inverse = IFlexFT2D.fft(Nk=shape, dk=dk, x0=x0, k0=k0)
        self.assertTrue(
            jnp.allclose(inverse(transformed), samples, rtol=5e-5, atol=5e-5)
        )

    def test_2d_shape_names_keep_their_domains_in_both_directions(self):
        Nx = (4, 5)
        Nk = (2, 3)
        samples_x = jnp.arange(Nx[0] * Nx[1], dtype=float).reshape(Nx)

        forward = FlexFT2D(
            Nx=Nx,
            Nk=Nk,
            dx=(0.2, 0.3),
            dk=(0.1, 0.15),
            method="direct",
        )
        samples_k = forward(samples_x)

        inverse = IFlexFT2D(
            Nk=Nk,
            Nx=Nx,
            dk=(0.1, 0.15),
            dx=(0.2, 0.3),
            method="direct",
        )

        self.assertEqual(forward.Nx, Nx)
        self.assertEqual(forward.Nk, Nk)
        self.assertEqual(samples_k.shape, Nk)
        self.assertEqual(inverse.Nx, Nx)
        self.assertEqual(inverse.Nk, Nk)
        self.assertEqual(inverse(samples_k).shape, Nx)

        one_shot_k = flexft2d(
            samples_x,
            Nk=Nk,
            dx=(0.2, 0.3),
            dk=(0.1, 0.15),
            method="direct",
        )
        one_shot_x = iflexft2d(
            samples_k,
            Nx=Nx,
            dk=(0.1, 0.15),
            dx=(0.2, 0.3),
            method="direct",
        )
        self.assertTrue(jnp.allclose(one_shot_k, samples_k))
        self.assertTrue(jnp.allclose(one_shot_x, inverse(samples_k)))

    def test_2d_signatures_name_shapes_by_domain(self):
        expected_parameters = {
            FlexFT2D: {"Nx", "Nk", "dx", "dk", "method", "x0", "k0"},
            FlexFT2D.fft: {"Nx", "dx", "x0", "k0"},
            IFlexFT2D: {"Nk", "Nx", "dk", "dx", "method", "x0", "k0"},
            IFlexFT2D.fft: {"Nk", "dk", "x0", "k0"},
            flexft2d: {"f", "Nk", "dx", "dk", "method", "x0", "k0"},
            iflexft2d: {"F", "Nx", "dk", "dx", "method", "x0", "k0"},
        }

        for callable_, expected in expected_parameters.items():
            with self.subTest(callable=callable_):
                self.assertEqual(set(inspect.signature(callable_).parameters), expected)

    def test_old_fft_constructor_paths_give_factory_guidance(self):
        calls = (
            lambda: FlexFT(N=8, dx=0.1, dk=None, method="fft"),
            lambda: IFlexFT(N=8, dk=1.0, dx=None, method="fft"),
            lambda: FlexFT2D(
                Nx=(8, 8),
                dx=(0.1, 0.1),
                dk=(0.2, None),
                method=("direct", "fft"),
            ),
            lambda: IFlexFT2D(
                Nk=(8, 8),
                dk=(1.0, 1.0),
                dx=(0.1, None),
                method=("direct", "fft"),
            ),
            lambda: flexft(jnp.ones(8), dx=0.1, dk=None, method="fft"),
        )

        for call in calls:
            with self.subTest(call=call):
                with self.assertRaisesRegex(ValueError, r"\.fft\(\) class factory"):
                    call()


if __name__ == "__main__":
    unittest.main()
