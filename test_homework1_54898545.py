"""Unit tests for Homework #1

Important
=========

Do not modify the way in which your solution functions

* homework1.exercise1.newton_step
* homework1.exercise1.newton

are imported. The actual test suite used to grade your homework will import
your functions in the exact same way.

"""

import unittest
import numpy
from numpy import sin, cos, exp, pi, dot, eye, zeros, ones, array, sign
from numpy.linalg import norm, solve
from numpy.random import rand, randn
from numpy import diag, tril, triu, dot, ones, zeros, sign
from numpy.linalg import norm
from scipy.linalg import solve_triangular

# Import the homework functions
from homework1.exercise1 import collatz_step, collatz
from homework1.exercise2 import gradient_step, gradient_descent
from homework1.exercise3 import (
    is_sdd,
    decompose,
    jacobi_step,
    jacobi_iteration,
    gauss_seidel_step,
    gauss_seidel_iteration,
)

class TestExercise1(unittest.TestCase):
    """Testing the validity of

    * homework1.exercise1.collatz_step
    * homework1.exercise1.collatz
    """
    def test_collatz_step(self):
        self.assertEqual(collatz_step(5), 16)
        self.assertEqual(collatz_step(16), 8)
        self.assertEqual(collatz_step(97), 292)

    def test_collatz_step_one(self):
        self.assertEqual(collatz_step(1), 1)

    def test_collatz_step_error(self):
        with self.assertRaises(ValueError):
            collatz_step(-1)
            collatz_step(-2)
            collatz_step(-19)

    def test_collatz(self):
        s6 = [6, 3, 10, 5, 16, 8, 4, 2, 1]
        self.assertEqual(collatz(6), s6)

        s43 = [43, 130, 65, 196, 98, 49, 148, 74, 37, 112, 56, 28, 14, 7, 22,
               11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1]
        self.assertEqual(collatz(43), s43)

        s11 = [11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1]
        self.assertEqual(collatz(11), s11)

        s29 = [29, 88, 44, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 
               2, 1]
        self.assertEqual(collatz(29), s29)


class TestExercise2(unittest.TestCase):
    """Testing the validity of

    * homework1.exercise2.gradient_step
    * homework1.exercise2.gradient_descent

    """
    def test_gradient_step(self):
        f = lambda x: 0.1*x**4-x**3+3.5*x**2-5*x+2.4
        df = lambda x: -5+7*x-3*x**2+0.4*x**3
        x1 = gradient_step(3, df, sigma=0.2)
        x1_actual = 3.04
        self.assertAlmostEqual(x1, x1_actual)
        x1 = gradient_step(10, df, sigma=0.2)
        x1_actual = -23
        self.assertAlmostEqual(x1, x1_actual)

    def test_gradient_descent(self):
        f = lambda x: x**2 - 1
        df = lambda x: 2*x
        xf = gradient_descent(f,df,1,0.5,1e-10)
        xf_actual = 0.0
        self.assertAlmostEqual(xf, xf_actual)

    def test_gradient_descent_nearmin_smallsig(self):
        f = lambda x: 0.1*x**4-x**3+3.5*x**2-5*x+2.4
        df = lambda x: -5+7*x-3*x**2+0.4*x**3
        xf = gradient_descent(f,df,1.3,0.05,1e-10)
        xf_actual = 1.381966 # 1.3819660093917199
        self.assertAlmostEqual(xf, xf_actual)

    def test_sigma_condition(self):
        f = lambda x: x**2 - 1
        df = lambda x: 2*x
        x0 = 1
        with self.assertRaises(ValueError):
            gradient_descent(f, df, x0, sigma=-100)
            gradient_descent(f, df, x0, sigma=100)

    def test_gradient_descent_robust(self):
        f = lambda x: 0.1*x**4-x**3+3.5*x**2-5*x+2.4
        df = lambda x: -5+7*x-3*x**2+0.4*x**3
        xf = gradient_descent(f,df,2.5,0.3,1e-10)
        xf_actual1 = 3.618034
        xf_actual2 = 1.381966
        self.assertTrue(round(xf-xf_actual1,7)==0 or round(xf-xf_actual2,7)==0)


class TestExercise3(unittest.TestCase):
    """Testing the validity of

    * homework1.exercise3.decompose
    * homework1.exercise3.is_sdd
    * homework1.exercise3.jacobi_step
    * homework1.exercise3.jacobi_iteration
    * homework1.exercise3.gauss_seidel_step
    * homework1.exercise3.gauss_seidel_iteration

    """
    def test_decompose(self):
        # the test written below tests if a random dense 10x10 SDD matrix A is properly
        # decomposed.
        B=array([[ 0.28384623, -0.00671327, -0.5909395 ,  0.22319892,  0.47311985,
                   0.68994718,  1.68902479,  1.31311603,  1.34724346,  0.87469211],
                 [ 1.4600519 ,  1.32770264, -0.07754125, -0.40217056, -0.58388847,
                  -1.51263798, -0.47040971,  2.30832758,  0.92357739,  0.54564659],
                 [ 0.57554678,  0.15559427,  1.90983902,  0.16232167, -0.07962706,
                  -0.54497108, -1.51698454,  0.86898915,  0.96449889,  0.33047333],
                 [-2.29421824, -0.12860935, -2.4615153 , -0.49196098, -0.39143065,
                   0.7018368 ,  0.61227205, -0.0688328 , -0.14342273,  0.49445229],
                 [ 0.20822844, -1.53027653,  1.60181548, -1.78165038, -0.5640091 ,
                   1.18444976, -0.03295762,  1.53161805,  0.86493622,  1.18826723],
                 [ 1.19243729,  0.97582135, -0.39137045,  2.20075989, -0.3963792 ,
                  -0.70617887,  0.56454132, -0.16555516,  1.059915  , -0.56712244],
                 [ 1.69045549,  0.67501015, -1.4364708 ,  0.8182705 ,  1.09775466,
                  -0.61637005, -0.23972642,  0.96483237,  1.80930647, -0.31649356],
                 [ 0.12250029,  0.53998451,  0.71529106, -0.55052708,  0.37901874,
                  -0.45185482,  1.32690993, -0.35890031,  0.86842993, -0.96008517],
                 [-0.79938861,  0.53407704,  1.14140023,  0.39350382,  0.01804328,
                   0.16847941, -0.22454093,  1.13058928,  1.6950224 ,  0.59640698],
                 [-0.16192425, -0.07277925, -0.03477962,  0.36739539,  0.15564998,
                   1.92979674,  0.84068755, -1.41742836, -0.059325  ,  0.26732909]])
        D, L, U = decompose(B)
        D_actual = array(array([[ 0.28384623,  0.        ,  0.        ,  0.        ,  0.        ,
                                  0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                                [ 0.        ,  1.32770264,  0.        ,  0.        ,  0.        ,
                                  0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                                [ 0.        ,  0.        ,  1.90983902,  0.        ,  0.        ,
                                  0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                                [ 0.        ,  0.        ,  0.        , -0.49196098,  0.        ,
                                  0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                                [ 0.        ,  0.        ,  0.        ,  0.        , -0.5640091 ,
                                  0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                                [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                                 -0.70617887,  0.        ,  0.        ,  0.        ,  0.        ],
                                [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                                  0.        , -0.23972642,  0.        ,  0.        ,  0.        ],
                                [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                                  0.        ,  0.        , -0.35890031,  0.        ,  0.        ],
                                [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                                  0.        ,  0.        ,  0.        ,  1.6950224 ,  0.        ],
                                [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                                  0.        ,  0.        ,  0.        ,  0.        ,  0.26732909]]))
        L_actual = array([[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                            0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                          [ 1.4600519 ,  0.        ,  0.        ,  0.        ,  0.        ,
                            0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                          [ 0.57554678,  0.15559427,  0.        ,  0.        ,  0.        ,
                            0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                          [-2.29421824, -0.12860935, -2.4615153 ,  0.        ,  0.        ,
                            0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                          [ 0.20822844, -1.53027653,  1.60181548, -1.78165038,  0.        ,
                            0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                          [ 1.19243729,  0.97582135, -0.39137045,  2.20075989, -0.3963792 ,
                            0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                          [ 1.69045549,  0.67501015, -1.4364708 ,  0.8182705 ,  1.09775466,
                           -0.61637005,  0.        ,  0.        ,  0.        ,  0.        ],
                          [ 0.12250029,  0.53998451,  0.71529106, -0.55052708,  0.37901874,
                           -0.45185482,  1.32690993,  0.        ,  0.        ,  0.        ],
                          [-0.79938861,  0.53407704,  1.14140023,  0.39350382,  0.01804328,
                            0.16847941, -0.22454093,  1.13058928,  0.        ,  0.        ],
                          [-0.16192425, -0.07277925, -0.03477962,  0.36739539,  0.15564998,
                            1.92979674,  0.84068755, -1.41742836, -0.059325  ,  0.        ]])
        U_actual = array([[ 0.        , -0.00671327, -0.5909395 ,  0.22319892,  0.47311985,
                            0.68994718,  1.68902479,  1.31311603,  1.34724346,  0.87469211],
                          [ 0.        ,  0.        , -0.07754125, -0.40217056, -0.58388847,
                           -1.51263798, -0.47040971,  2.30832758,  0.92357739,  0.54564659],
                          [ 0.        ,  0.        ,  0.        ,  0.16232167, -0.07962706,
                           -0.54497108, -1.51698454,  0.86898915,  0.96449889,  0.33047333],
                          [ 0.        ,  0.        ,  0.        ,  0.        , -0.39143065,
                            0.7018368 ,  0.61227205, -0.0688328 , -0.14342273,  0.49445229],
                          [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                            1.18444976, -0.03295762,  1.53161805,  0.86493622,  1.18826723],
                          [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                            0.        ,  0.56454132, -0.16555516,  1.059915  , -0.56712244],
                          [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                            0.        ,  0.        ,  0.96483237,  1.80930647, -0.31649356],
                          [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                            0.        ,  0.        ,  0.        ,  0.86842993, -0.96008517],
                          [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                            0.        ,  0.        ,  0.        ,  0.        ,  0.59640698],
                          [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                            0.        ,  0.        ,  0.        ,  0.        ,  0.        ]])
        self.assertAlmostEqual(norm(D_actual - D), 0)
        self.assertAlmostEqual(norm(L_actual - L), 0)
        self.assertAlmostEqual(norm(U_actual - U), 0)

    def test_isSDD(self):
        # this checks if the is_sdd function can correctly identify an SDD matrix
        B=array([[ 0.28384623, -0.00671327, -0.5909395 ,  0.22319892,  0.47311985,
                   0.68994718,  1.68902479,  1.31311603,  1.34724346,  0.87469211],
                 [ 1.4600519 ,  1.32770264, -0.07754125, -0.40217056, -0.58388847,
                  -1.51263798, -0.47040971,  2.30832758,  0.92357739,  0.54564659],
                 [ 0.57554678,  0.15559427,  1.90983902,  0.16232167, -0.07962706,
                  -0.54497108, -1.51698454,  0.86898915,  0.96449889,  0.33047333],
                 [-2.29421824, -0.12860935, -2.4615153 , -0.49196098, -0.39143065,
                   0.7018368 ,  0.61227205, -0.0688328 , -0.14342273,  0.49445229],
                 [ 0.20822844, -1.53027653,  1.60181548, -1.78165038, -0.5640091 ,
                   1.18444976, -0.03295762,  1.53161805,  0.86493622,  1.18826723],
                 [ 1.19243729,  0.97582135, -0.39137045,  2.20075989, -0.3963792 ,
                  -0.70617887,  0.56454132, -0.16555516,  1.059915  , -0.56712244],
                 [ 1.69045549,  0.67501015, -1.4364708 ,  0.8182705 ,  1.09775466,
                  -0.61637005, -0.23972642,  0.96483237,  1.80930647, -0.31649356],
                 [ 0.12250029,  0.53998451,  0.71529106, -0.55052708,  0.37901874,
                  -0.45185482,  1.32690993, -0.35890031,  0.86842993, -0.96008517],
                 [-0.79938861,  0.53407704,  1.14140023,  0.39350382,  0.01804328,
                   0.16847941, -0.22454093,  1.13058928,  1.6950224 ,  0.59640698],
                 [-0.16192425, -0.07277925, -0.03477962,  0.36739539,  0.15564998,
                   1.92979674,  0.84068755, -1.41742836, -0.059325  ,  0.26732909]])
        check_B=is_sdd(B)
        self.assertFalse(check_B)
        A = B+10*sign(diag(diag(B)))
        check_A=is_sdd(A)
        self.assertTrue(check_A)


    def test_jacobi_step(self):
        # the test written below tests if jacobi step produces the
        # correct output for a sompletridiagonal SDD matrix A
        A = array([[ 11.07336608,  -0.85580894,   0.        ,   0.        ,
                      0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ],
                   [  0.34408418, -11.5525474 ,   1.32434231,   0.        ,
                      0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ],
                   [  0.        ,  -1.25446038,  11.39901   ,   0.47072897,
                      0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ],
                   [  0.        ,   0.        ,   1.23226501, -10.25414388,
                     -0.3773415 ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ],
                   [  0.        ,   0.        ,   0.        ,   1.22311693,
                     10.2952351 ,   0.78994453,   0.        ,   0.        ,
                      0.        ,   0.        ],
                   [  0.        ,   0.        ,   0.        ,   0.        ,
                     -0.42566436, -10.52022012,  -1.58403561,   0.        ,
                      0.        ,   0.        ],
                   [  0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.65026657, -10.45407605,  -1.44470734,
                      0.        ,   0.        ],
                   [  0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ,   0.90469079,  10.33843249,
                     -0.49375486,   0.        ],
                   [  0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ,   0.        ,  -0.76157863,
                    -10.82581569,  -0.97784685],
                   [  0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ,   0.        ,   0.        ,
                      0.80920961,  11.84434943]])
        b = array([-0.82211229, -1.40264905, -0.10539367,  0.22763046,  0.15404998,
                    1.13328694, -2.68971004,  0.18112895, -0.56502348, -0.6756333 ])
        x0 = ones(len(b))
        D, L, U = decompose(A)
        x1 = jacobi_step(D, L, U, b, x0)
        x1_actual = array([ 0.00304304,  0.26583536,  0.05950848,  0.06117459, -0.18057009,
                           -0.29875676,  0.18129477, -0.02222842, -0.10848162, -0.12536298])
        self.assertAlmostEqual(norm(x1-x1_actual), 0)

    def test_gauss_seidel_step(self):
        # the test written below tests if Gauss-seidel step produces the
        # correct output for a sompletridiagonal SDD matrix A
        A = array([[ 11.07336608,  -0.85580894,   0.        ,   0.        ,
                      0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ],
                   [  0.34408418, -11.5525474 ,   1.32434231,   0.        ,
                      0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ],
                   [  0.        ,  -1.25446038,  11.39901   ,   0.47072897,
                      0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ],
                   [  0.        ,   0.        ,   1.23226501, -10.25414388,
                     -0.3773415 ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ],
                   [  0.        ,   0.        ,   0.        ,   1.22311693,
                     10.2952351 ,   0.78994453,   0.        ,   0.        ,
                      0.        ,   0.        ],
                   [  0.        ,   0.        ,   0.        ,   0.        ,
                     -0.42566436, -10.52022012,  -1.58403561,   0.        ,
                      0.        ,   0.        ],
                   [  0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.65026657, -10.45407605,  -1.44470734,
                      0.        ,   0.        ],
                   [  0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ,   0.90469079,  10.33843249,
                     -0.49375486,   0.        ],
                   [  0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ,   0.        ,  -0.76157863,
                    -10.82581569,  -0.97784685],
                   [  0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ,   0.        ,   0.        ,
                      0.80920961,  11.84434943]])
        b = array([-0.82211229, -1.40264905, -0.10539367,  0.22763046,  0.15404998,
                    1.13328694, -2.68971004,  0.18112895, -0.56502348, -0.6756333 ])
        x0 = ones(len(b))
        D, L, U = decompose(A)
        x1 = gauss_seidel_step(D, L, U, b, x0)
        x1_actual = array([-0.06170079,  0.16227553,  0.09662346,  0.10123638, -0.08866736,
                           -0.19775514,  0.32920744, -0.0703139 , -0.00683268, -0.12536298])
        x1_actual_correct = array([0.00304304, 0.23614172, -0.02455408, -0.06194853, -0.05440617,
                                   -0.25609386, 0.10316296,  0.05625159, -0.04209044, -0.05416704])
        self.assertAlmostEqual(min(norm(x1-x1_actual), norm(x1-x1_actual_correct)), 0)

    def test_jacobi_iteration(self):
        # the test written below tests if jacobi iteration produces the
        # correct output for a sompletridiagonal SDD matrix A
        A = array([[ 11.07336608,  -0.85580894,   0.        ,   0.        ,
                      0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ],
                   [  0.34408418, -11.5525474 ,   1.32434231,   0.        ,
                      0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ],
                   [  0.        ,  -1.25446038,  11.39901   ,   0.47072897,
                      0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ],
                   [  0.        ,   0.        ,   1.23226501, -10.25414388,
                     -0.3773415 ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ],
                   [  0.        ,   0.        ,   0.        ,   1.22311693,
                     10.2952351 ,   0.78994453,   0.        ,   0.        ,
                      0.        ,   0.        ],
                   [  0.        ,   0.        ,   0.        ,   0.        ,
                     -0.42566436, -10.52022012,  -1.58403561,   0.        ,
                      0.        ,   0.        ],
                   [  0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.65026657, -10.45407605,  -1.44470734,
                      0.        ,   0.        ],
                   [  0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ,   0.90469079,  10.33843249,
                     -0.49375486,   0.        ],
                   [  0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ,   0.        ,  -0.76157863,
                    -10.82581569,  -0.97784685],
                   [  0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ,   0.        ,   0.        ,
                      0.80920961,  11.84434943]])
        b = array([-0.82211229, -1.40264905, -0.10539367,  0.22763046,  0.15404998,
                    1.13328694, -2.68971004,  0.18112895, -0.56502348, -0.6756333 ])
        x0 = ones(len(b))
        x = jacobi_iteration(A, b, x0, 1e-4)
        x_actual = solve(A,b)
        err = norm(x-x_actual)
        true_err = 5.6298508459206761e-06
        self.assertTrue(round(err-true_err,12)==0)

    def test_gauss_seidel_iteration(self):
        # the test written below tests if Gauss-Seidel iteration produces the
        # correct output for a sompletridiagonal SDD matrix A
        A = array([[ 11.07336608,  -0.85580894,   0.        ,   0.        ,
                      0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ],
                   [  0.34408418, -11.5525474 ,   1.32434231,   0.        ,
                      0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ],
                   [  0.        ,  -1.25446038,  11.39901   ,   0.47072897,
                      0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ],
                   [  0.        ,   0.        ,   1.23226501, -10.25414388,
                     -0.3773415 ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ],
                   [  0.        ,   0.        ,   0.        ,   1.22311693,
                     10.2952351 ,   0.78994453,   0.        ,   0.        ,
                      0.        ,   0.        ],
                   [  0.        ,   0.        ,   0.        ,   0.        ,
                     -0.42566436, -10.52022012,  -1.58403561,   0.        ,
                      0.        ,   0.        ],
                   [  0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.65026657, -10.45407605,  -1.44470734,
                      0.        ,   0.        ],
                   [  0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ,   0.90469079,  10.33843249,
                     -0.49375486,   0.        ],
                   [  0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ,   0.        ,  -0.76157863,
                    -10.82581569,  -0.97784685],
                   [  0.        ,   0.        ,   0.        ,   0.        ,
                      0.        ,   0.        ,   0.        ,   0.        ,
                      0.80920961,  11.84434943]])
        b = array([-0.82211229, -1.40264905, -0.10539367,  0.22763046,  0.15404998,
                    1.13328694, -2.68971004,  0.18112895, -0.56502348, -0.6756333 ])
        x0 = ones(len(b))
        x = gauss_seidel_iteration(A, b, x0, epsilon=1e-3)
        x_actual = solve(A,b)
        err = norm(x-x_actual)
        true_err = 5.8690481842626678e-06
        true_err_correct = 8.0354012468682747e-06
        self.assertTrue(round(min(abs(err-true_err),abs(err-true_err_correct)),12)==0)



# The following code is run when this Python module / file is executed as a
# script. This happens when you enter
#
# $ python test_homework1.py
#
# in the terminal.
if __name__ == '__main__':
    unittest.main(verbosity=2) # run the above tests
