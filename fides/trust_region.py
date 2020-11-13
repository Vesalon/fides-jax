"""
Trust Region Steps
------------------
This module provides the machinery to compute different trust-region(
-reflective) step proposals and select among them based on to their
performance according to the quadratic approximation of the objective function
"""

import numpy as np
import scipy.linalg as linalg

from numpy.linalg import norm
from scipy.sparse import csc_matrix

from typing import Tuple

from .logging import logger
from .subproblem import (
    solve_1d_trust_region_subproblem, solve_nd_trust_region_subproblem
)
from .constants import SubSpaceDim


def normalize(v: np.ndarray) -> None:
    """
    Inplace normalization of a vector

    :param v:
        vector to be normalized
    """
    nv = norm(v)
    if nv > 0:
        v[:] = v/nv  # change inplace


class Step:
    """
    Base class for the computation of a proposal step

    :ivar x: current state of optimization variables
    :ivar s: proposed step
    :ivar sc: coefficients in the 1D/2D subspace that defines the affine
        transformed step ss: ss = subspace * sc
    :ivar ss: affine transformed step: s = scaling * ss
    :ivar og_s: s without step back
    :ivar og_sc: st without step back
    :ivar og_ss: ss without step back
    :ivar sg: rescaled gradient scaling * g
    :ivar hess: hessian of the objective function at x
    :ivar g_dscaling: diag(g) * dscaling
    :ivar delta: trust region radius in the transformed space defined by
        scaling matrix
    :ivar theta: controls step back, fraction of step to take if full
        step would reach breakpoint
    :ivar lb: lower boundaries for x
    :ivar ub: upper boundaries for x
    :ivar minbr: maximal fraction of step s that can be taken to reach
        first breakpoint
    :ivar ipt: index of x that specifies the variable that will hit the
        breakpoint if a step minbr * s is taken
    :ivar qpval0: value to the quadratic subproblem at x
    :ivar qpval: value of the quadratic subproblem for the proposed step
    :ivar shess: matrix of the full quadratic problem
    :ivar cg: projection of the g_hat to the subspace
    :ivar chess: projection of the B to the subspace
    """
    def __init__(self,
                 x: np.ndarray,
                 sg: np.ndarray,
                 hess: np.ndarray,
                 scaling: csc_matrix,
                 g_dscaling: csc_matrix,
                 delta: float,
                 theta: float,
                 ub: np.ndarray,
                 lb: np.ndarray):
        """

        :param x:
            Reference point
        :param sg:
            Gradient in rescaled coordinates
        :param hess:
            Hessian in unscaled coordinates
        :param scaling:
            Matrix that defines scaling transformation
        :param g_dscaling:
            Unscaled gradient multiplied by derivative of scaling
            transformation
        :param delta:
            Trust region Radius in scaled coordinates
        :param theta:
            Stepback parameter that controls how close steps are allowed to
            get to the boundary
        :param ub:
            Upper boundary
        :param lb:
            Lower boundary
        :param reflection_count:
            Number of reflections that were applied to obtain this step
        """
        self.x = x

        self.s = None
        self.sc = None
        self.ss = None

        self.og_s = None
        self.og_sc = None
        self.og_ss = None

        self.sg = sg
        self.scaling = scaling

        self.delta = delta
        self.theta = theta

        self.lb = lb
        self.ub = ub

        self.minbr = 1.0
        self.alpha = 1.0
        self.ipt = 0

        self.qpval0 = 0.0
        self.qpval = 0.0

        self.shess = scaling * hess * scaling + g_dscaling

        self.cg = None
        self.chess = None
        self.subspace = None

        self.s0 = np.zeros((hess.shape[0],))
        self.ss0 = np.zeros((hess.shape[0],))

        self.reflection_count = 0

    def step_back(self):
        """
        This function truncates the step based on the distance of the
        current point to the boundary.
        """
        # create copies of the calculated step
        self.og_s = self.s.copy()
        self.og_ss = self.ss.copy()
        self.og_sc = self.sc.copy()

        nonzero = np.abs(self.s) > 0
        if np.any(nonzero):
            # br quantifies the distance to the boundary normalized
            # by the proposed step, this indicates the fraction of the step
            # that would put the respective variable at the boundary
            # This is defined in [Coleman-Li1996] (3.1)
            br = np.max(np.vstack([
                (self.ub[nonzero] - self.x[nonzero])/self.s[nonzero],
                (self.lb[nonzero] - self.x[nonzero])/self.s[nonzero]
            ]), axis=0)
            self.ipt = np.where(br == np.min(br))[0]
            if np.isscalar(br):
                self.minbr = br
            else:
                self.minbr = br[self.ipt][0]
            # compute the minimum of the step
            self.alpha = np.min([1, self.theta * self.minbr])

        self.s *= self.alpha
        self.sc *= self.alpha
        self.ss *= self.alpha

    def reduce_to_subspace(self) -> None:
        """
        This function projects the matrix shess and the vector sg to the
        subspace
        """
        self.chess = self.subspace.T.dot(self.shess.dot(self.subspace))
        self.cg = self.subspace.T.dot(self.sg)

    def compute_step(self) -> None:
        """
        Compute the step as solution to the trust region subproblem. Special
        code is used for the special case 1-dimensional subspace case
        """
        if self.subspace.shape[1] > 1:
            self.sc, _ = solve_nd_trust_region_subproblem(self.chess, self.cg,
                                                          self.delta)
        else:
            self.sc = solve_1d_trust_region_subproblem(self.shess, self.sg,
                                                       self.subspace[:, 0],
                                                       self.delta,
                                                       self.ss0)
        self.ss = self.subspace.dot(np.real(self.sc))
        self.s = self.scaling.dot(self.ss)

    def calculate(self):
        """
        Calculates step and the expected objective function value according to
        the quadratic approximation
        """
        self.reduce_to_subspace()
        self.compute_step()
        self.step_back()
        self.qpval = self.qpval0 + self.cg.dot(self.sc) + \
            .5 * (self.sc.dot(self.chess).dot(self.sc))[0, 0]


class TRStepFull(Step):
    """
    This class provides the machinery to compute an exact solution of
    the trust region subproblem.
    """

    type = 'trnd'

    def __init__(self, x, sg, hess, scaling, g_dscaling, delta, theta,
                 ub, lb):
        super().__init__(x, sg, hess, scaling, g_dscaling, delta, theta,
                         ub, lb)
        self.subspace = np.eye(hess.shape[0])


class TRStep2D(Step):
    """
    This class provides the machinery to compute an approximate solution of
    the trust region subproblem according to a 2D subproblem
    """

    type = 'tr2d'

    def __init__(self, x, sg, hess, scaling, g_dscaling, delta, theta,
                 ub, lb, subspace):
        """
        :param subspace:
            Precomputed subspace
        """
        super().__init__(x, sg, hess, scaling, g_dscaling, delta, theta,
                         ub, lb)
        self.subspace = subspace
        if self.subspace is None:
            n = len(sg)

            s_newt = linalg.solve(hess, sg)
            posdef = s_newt.dot(hess.dot(s_newt)) > 0
            normalize(s_newt)

            if n > 1:
                if not posdef:
                    # in this case we are in Case 2 of Fig 12 in
                    # [Coleman-Li1994]
                    logger.debug('Newton direction did not have negative '
                                 'curvature adding scaling * np.sign(sg) to '
                                 '2D subspace.')
                    s_grad = scaling * np.sign(sg) + (sg == 0)
                else:
                    s_grad = sg.copy()

                # orthonormalize, this ensures that S.T.dot(S) = I and we
                # can use S/S.T for transformation
                s_grad = s_grad - s_newt * (s_newt.dot(s_grad))
                normalize(s_grad)
                # if non-zero, add s_grad to subspace
                if np.any(s_grad != 0):
                    self.subspace = np.vstack([s_newt, s_grad]).T
                    return
                else:
                    logger.debug('Singular subspace, continuing with 1D '
                                 'subspace.')

            self.subspace = np.expand_dims(s_newt, 1)


class TRStepReflected(Step):
    """
    This class provides the machinery to compute a reflected step based on
    trust region subproblem solution thathit the boundaries.
    """

    type = 'trr'

    def __init__(self, x, sg, hess, scaling, g_dscaling, delta, theta,
                 ub, lb, step: Step):
        """
        :param step:
            Trust-region step that is reflected
        """
        super().__init__(x, sg, hess, scaling, g_dscaling, delta, theta,
                         ub, lb)

        full_alpha = min(step.minbr, 1)
        self.s0 = full_alpha * step.og_s
        self.ss0 = full_alpha * step.og_ss
        # update x and at breakpoint
        self.x = x + self.s0

        # reflect the transformed step at the boundary
        nss = step.og_ss.copy()
        nss[step.ipt] *= -1
        normalize(nss)
        self.subspace = np.expand_dims(nss, 1)

        self.qpval0 = step.qpval

        self.reflection_count = step.reflection_count + 1


class GradientStep(Step):
    """
    This class provides the machinery to compute a gradient step.
    """

    type = 'g'

    def __init__(self, x, sg, hess, scaling, g_dscaling, delta, theta,
                 ub, lb):
        super().__init__(x, sg, hess, scaling, g_dscaling, delta, theta,
                         ub, lb)
        s_grad = sg.copy()
        normalize(s_grad)
        self.subspace = np.expand_dims(s_grad, 1)


def trust_region_reflective(x: np.ndarray,
                            g: np.ndarray,
                            hess: np.ndarray,
                            scaling: csc_matrix,
                            tr_subspace: np.ndarray,
                            delta: float,
                            dv: np.ndarray,
                            theta: float,
                            lb: np.ndarray,
                            ub: np.ndarray,
                            subspace_dim: SubSpaceDim) -> Tuple[np.ndarray,
                                                                np.ndarray,
                                                                float,
                                                                np.ndarray,
                                                                str]:
    """
    Compute a step according to the solution of the trust-region subproblem.
    If step-back is necessary, gradient and reflected trust region step are
    also evaluated in terms of their performance according to the local
    quadratic approximation

    :param x:
        Current values of the optimization variables
    :param g:
        Objective function gradient at x
    :param hess:
        (Approximate) objective function Hessian at x
    :param scaling:
        Scaling transformation according to distance to boundary
    :param tr_subspace:
        Precomputed subspace from previous iteration for reuse if proposed
        step was not accepted
    :param delta:
        Trust region radius, note that this applies after scaling
        transformation
    :param dv:
        derivative of scaling transformation
    :param theta:
        parameter regulating stepback
    :param lb:
        lower optimization variable boundaries
    :param ub:
        upper optimization variable boundaries
    :param subspace_dim:
        Subspace dimension in which the subproblem will be solved. Larger
        subspaces require more compute time but can yield higher quality step
        proposals.

    :return:
        s: proposed step,
        ss: rescaled proposed step,
        qpval: expected function value according to local quadratic
        approximation,
        subspace: computed subspace for reuse if proposed step is not accepted,
        steptype: type of step that was selected for proposal
    """
    sg = scaling.dot(g)
    g_dscaling = csc_matrix(np.diag(np.abs(g) * dv))

    if subspace_dim == SubSpaceDim.TWO:
        tr_step = TRStep2D(
            x, sg, hess, scaling, g_dscaling, delta, theta, ub, lb, tr_subspace
        )
    elif subspace_dim == SubSpaceDim.FULL:
        tr_step = TRStepFull(
            x, sg, hess, scaling, g_dscaling, delta, theta, ub, lb,
        )
    else:
        raise ValueError('Invalid choice of subspace dimension.')
    tr_step.calculate()

    # in case of truncation, we hit the boundary and we check both the
    # gradient and the reflected step, either of which could be better than the
    # TR step

    steps = [tr_step]
    if tr_step.alpha < 1 and len(g) > 1:
        g_step = GradientStep(x, sg, hess, scaling, g_dscaling, delta,
                              theta, ub, lb)
        g_step.calculate()

        steps.append(tr_step)

        rtr_step = TRStepReflected(x, g, hess, scaling, g_dscaling, delta,
                                   theta, ub, lb, tr_step)
        rtr_step.calculate()
        steps.append(rtr_step)
        for ireflection in range(len(x)-1):
            if rtr_step.alpha < 1 or norm(rtr_step.ss) > self.delta * 0.9:
                break
            # recursively add more reflections
            rtr_old = rtr_step
            rtr_step = TRStepReflected(x, g, hess, scaling, g_dscaling, delta,
                                       theta, ub, lb, rtr_old)
            rtr_step.calculate()
            steps.append(rtr_step)

    if len(steps) > 1:
        rcountstrs = [str(step.reflection_count)
                      * int(step.reflection_count > 0)
                      for step in steps]
        logger.debug(' | '.join([
            f'{step.type + rcountstr}: [qp:'
            f' {step.qpval:.2E}, '
            f'a: {step.alpha:.2E}]'
            for rcountstr, step in zip(rcountstrs, steps)
        ]))

    qpvals = [step.qpval for step in steps]
    step = steps[np.argmin(qpvals)]

    return step.s + step.s0, step.ss + step.ss0, step.qpval, \
        tr_step.subspace, step.type
