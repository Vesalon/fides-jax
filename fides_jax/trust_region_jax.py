from dataclasses import dataclass
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
np_eps = jnp.finfo(jnp.float64).eps
# 
# def make_newton(f, value_and_grad = False):
    # """
    # Newton's method for root-finding.
    # makes newton with no convergence criteria, just set number of iterations
    # """
    # if value_and_grad:
        # def body(it, x):
            # fx, dfx = f(x)
            # step = fx / dfx
            # new_x = x - step
            # return jax.lax.select(jnp.isfinite(new_x), new_x, x)
    # else:
        # def body(it, x):
            # fx, dfx = f(x), jax.grad(f)(x)
            # step = fx / dfx
            # new_x = x - step
            # return jax.lax.select(jnp.isfinite(new_x), new_x, x)
    # def newton(x0, num_iter):
        # return jax.lax.fori_loop(
            # 0,
            # num_iter,
            # body,
            # x0,
        # )
    # return jax.jit(newton)

@jax.jit
def normalize(v):
    nv = jnp.linalg.norm(v)
    return jax.lax.select(nv > 0, v/nv, v)

@jax.jit
def get_affine_scaling(x, grad, lb, ub):
    """
    Computes the vector v and dv, the diagonal of its Jacobian. For the
    definition of v, see Definition 2 in [Coleman-Li1994]

    :return:
        v scaling vector
        dv diagonal of the Jacobian of v wrt x
    """
    # this implements no scaling for variables that are not constrained by
    # bounds ((iii) and (iv) in Definition 2)
    _v = jnp.sign(grad) + (grad == 0)
    _dv = jnp.zeros(x.shape)

    # this implements scaling for variables that are constrained by
    # bounds ( i and ii in Definition 2) bounds is equal to ub if grad <
    # 0 lb if grad >= 0
    # bounds = -jnp.minimum(jnp.sign(grad), jnp.zeros(len(lb))) * ub + jnp.maximum(jnp.sign(grad), jnp.zeros(len(ub))) * lb
    bounds = jax.lax.select(grad < 0, ub, lb)
    bounded = jnp.isfinite(bounds)
    v = jnp.where(bounded, x - bounds, _v)
    dv = jnp.where(bounded, 1, _dv)
    return v, dv

@jax.jit
def quadratic_form(Q, p, x):
    return 0.5 * x.T.dot(Q).dot(x) + p.T.dot(x)

@jax.jit
def slam(lam, w, eigvals, eigvecs):
    el = eigvals + lam
    c = jnp.where(el != 0, w/el, w)
    return eigvecs.dot(c)

@jax.jit
def dslam(lam, w, eigvals, eigvecs):
    el = eigvals + lam
    _c = jnp.where(el != 0, w/-jnp.power(el, 2), w)
    c = jnp.where((el == 0) & (_c != 0), jnp.inf, _c)
    return eigvecs.dot(c)

@jax.jit
def secular(lam,w,eigvals,eigvecs,delta):
    res1 = jax.lax.select(lam < -jnp.min(eigvals), jnp.inf, 0.)
    s = slam(lam, w, eigvals, eigvecs)
    sn = jnp.linalg.norm(s)
    res2 = jax.lax.select(sn > 0, 1 / sn - 1 / delta, jnp.inf)
    return (res1 + res2)


@jax.jit
def dsecular(lam, w, eigvals, eigvecs, delta):
    s = slam(lam, w, eigvals, eigvecs)
    ds = dslam(lam, w, eigvals, eigvecs)
    sn = jnp.linalg.norm(s)
    return jax.lax.select(sn > 0, -s.T.dot(ds) / (jnp.linalg.norm(s) ** 3), jnp.inf)

@jax.jit
def secular_and_grad(x, w, eigvals, eigvecs, delta):
    return (
        secular(x, w, eigvals, eigvecs, delta),
        dsecular(x, w, eigvals, eigvecs, delta)
    )

@jax.jit
def secular_newton(x0, w, eigvals, eigvecs, delta, num_iter):
    """
    Newton's method for root-finding.
    no convergence criteria, just set number of iterations but if an iteration leads to inf/nan
    it is ignored and the following iterations essentially become expensive noops
    """
    def body(it, x):
        fx, dfx = secular_and_grad(x, w, eigvals, eigvecs, delta)
        step = fx / dfx
        new_x = x - step
        return jnp.where(jnp.isfinite(new_x), new_x, x)

    return jax.lax.fori_loop(
        0,
        num_iter,
        body,
        x0,
    )

# @jax.jit
# def secular_newton(x0, w, eigvals, eigvecs, delta, maxiter):
#     """
#     Newton's method for root-finding.
#     no convergence criteria, just set number of iterations but if an iteration leads to inf/nan
#     it is ignored and the following iterations essentially become expensive noops
#     """
#     def cond(it, x):
#         return jnp.logical_and(jnp.isfinite(x), it < maxiter)

#     def body(it, x):
#         fx, dfx = secular_and_grad(x, w, eigvals, eigvecs, delta)
#         step = fx / dfx
#         new_x = x - step
#         return it+1, new_x

#     return jax.lax.while_loop(
#         cond,
#         body,
#         (0, x0),
#     )

@jax.jit
def copysign(a, b):
    return jnp.abs(-a)*(jnp.sign(b) + (b == 0))

@jax.jit
def get_1d_trust_region_boundary_solution(B, g, s, s0, delta):
    a = jnp.dot(s, s)
    # a = a[0, 0]
    b = 2 * jnp.dot(s0, s)
    c = jnp.dot(s0, s0) - delta**2

    aux = b + copysign(jnp.sqrt(b**2 - 4 * a * c), b)
    ts = jnp.array([-aux / (2 * a), -2 * c / aux])
    # qs = [quadratic_form(B, g, s0 + t * s) for t in ts]

    qf = jax.vmap(quadratic_form, in_axes=(None, None, 0))
    qs = qf(B, g, s0 + jnp.outer(ts, s))

    return ts[jnp.argmin(qs)]


@jax.jit
def solve_1d_trust_region_subproblem(B, g, s, delta, s0):
    """
    Solves the special case of a one-dimensional subproblem

    :param B:
        Hessian of the quadratic subproblem
    :param g:
        Gradient of the quadratic subproblem
    :param s:
        Vector defining the one-dimensional search direction
    :param delta:
        Norm boundary for the solution of the quadratic subproblem
    :param s0:
        reference point from where search is started, also counts towards
        norm of step

    :return:
        Proposed step-length
    """
    # if delta == 0.0:
    #     return delta * jnp.ones((1,))

    # if jnp.array_equal(s, jnp.zeros_like(s)):
    #     return jnp.zeros((1,))
    
    # null_res = jnp.zeros_like(s)

    a = 0.5 * B.dot(s).dot(s)
    # if not isinstance(a, float):
    #     a = a[0, 0]
    b = s.T.dot(B.dot(s0) + g)

    minq = -b / (2 * a)

    bound_cond = jnp.logical_and(a > 0, jnp.linalg.norm(minq * s + s0) <= delta)
    tau = jax.lax.select(bound_cond, minq, get_1d_trust_region_boundary_solution(B, g, s, s0, delta))


    res = tau * jnp.ones((1,))
    null_res = jnp.zeros_like(res)

    return jax.lax.select(jnp.logical_and(delta == 0.0, jnp.array_equal(s, jnp.zeros_like(s))), null_res, res)

@jax.jit
def solve_nd_trust_region_subproblem_jitted(B, g, delta):
    # See Nocedal & Wright 2006 for details
    # INITIALIZATION

    def hard_case(w, mineig, eigvals, eigvecs, delta, laminit, jmin):
        w = jnp.where((eigvals - mineig) == 0, 0, w)
        s = slam(-mineig, w, eigvals, eigvecs)
        # we know that ||s(lam) + sigma*v_jmin|| = delta, since v_jmin is
        # orthonormal, we can just substract the difference in norm to get
        # the right length.

        sigma = jnp.sqrt(jnp.maximum(delta**2 - jnp.linalg.norm(s) ** 2, 0))
        s = s + sigma * eigvecs[:, jmin]
        # logger.debug('Found boundary 2D subproblem solution via hard case')
        return s

    # instead of a cholesky factorization, we go with an eigenvalue
    # decomposition, which works pretty well for n=2
    eigvals, eigvecs = jnp.linalg.eig(B)
    eigvals = jnp.real(eigvals)
    eigvecs = jnp.real(eigvecs)
    w = -eigvecs.T.dot(g)
    jmin = eigvals.argmin()
    mineig = eigvals[jmin]

    # since B symmetric eigenvecs V are orthonormal
    # B + lambda I = V * (E + lambda I) * V.T
    # inv(B + lambda I) = V * inv(E + lambda I) * V.T
    # w = V.T * g
    # s(lam) = V * w./(eigvals + lam)
    # ds(lam) = - V * w./((eigvals + lam)**2)
    # \phi(lam) = 1/||s(lam)|| - 1/delta
    # \phi'(lam) = - s(lam).T*ds(lam)/||s(lam)||^3
    laminit = jax.lax.select(mineig > 0, 0.0, -mineig)

    # calculate s for positive definite case
    s = jnp.real(slam(0, w, eigvals, eigvecs))
    norm_s = jnp.linalg.norm(s)
    thresh = delta + jnp.sqrt(np_eps)
    posdef_cond = jnp.logical_and((mineig > 0), (norm_s <= thresh))
    neg_sval = secular(laminit, w, eigvals, eigvecs, delta) < 0


    maxiter = 100
    root = secular_newton(laminit, w, eigvals, eigvecs, delta, maxiter)
    indef_s = slam(root, w, eigvals, eigvecs)
    is_root = jnp.linalg.norm(indef_s) <= delta + 1e-12
    indef = jnp.logical_and(neg_sval, is_root)

    other_s = jax.lax.select(indef, indef_s, hard_case(w, mineig, eigvals, eigvecs, delta, laminit, jmin))
    other_case = jax.lax.select(indef, 1, 2)
    

    s = jax.lax.select(posdef_cond, s, other_s)
    hess_case = jax.lax.select(posdef_cond, 0, other_case)
    jax.debug.print('case encountered: {case}', case = hess_case)
    return s, hess_case

def solve_nd_trust_region_subproblem(B, g, delta):
    if delta == 0:
        return jnp.zeros(g.shape), 'zero'

    cases = ['posdef', 'indef', 'hard']
    s, case_ind = solve_nd_trust_region_subproblem_jitted(B, g, delta)
    return s, cases[int(case_ind)]


@jax.jit
def tr_iteration(x, grad, hess, lb, ub, theta_max, delta):
    v, dv = get_affine_scaling(x, grad, lb, ub)

    ### trust region init ###

    scaling = jnp.diag(jnp.sqrt(jnp.abs(v)))
    theta = jnp.maximum(theta_max, 1 - jnp.linalg.norm(v * grad, jnp.inf))

    sg = scaling.dot(grad)
    # diag(g_k)*J^v_k Eq (2.5) [ColemanLi1994]
    g_dscaling = jnp.diag(jnp.abs(grad) * dv)


    ### step ###

    br = jnp.ones(sg.shape)
    minbr = 1.0
    alpha = 1.0
    iminbr = jnp.array([])

    qpval = 0.0

    # B_hat (Eq 2.5) [ColemanLi1996]
    # shess = jnp.asarray(scaling * hess * scaling + g_dscaling)
    shess = jnp.matmul(jnp.matmul((scaling), hess), (scaling)) + g_dscaling

    s0 = jnp.zeros(sg.shape)
    ss0 = jnp.zeros(sg.shape)

    ### 2D steps ###

    og_s_newt = -jnp.linalg.lstsq(shess, sg)[0]
    # lstsq only returns absolute ev values
    e, v_ = jnp.linalg.eig(shess)
    posdef = jnp.min(jnp.real(e)) > -np_eps * jnp.max(jnp.abs(e))

    # if len(sg) == 1:
    #     s_newt = -sg[0] / self.shess[0]
    #     self.subspace = np.expand_dims(s_newt, 1)
    #     return


    s_newt_ = normalize(og_s_newt)
    subspace_0 = jnp.vstack([s_newt_, jnp.zeros(s_newt_.shape)]).T


    s_newt_2 = jnp.real(v_[:, jnp.argmin(jnp.real(e))])
    s_newt = jax.lax.select(posdef, s_newt_, s_newt_2)
    s_grad = jax.lax.select(posdef, sg.copy(), scaling.dot(jnp.sign(sg) + (sg == 0)))
    s_newt = normalize(s_newt)
    s_grad = s_grad - s_newt * s_newt.dot(s_grad)
    subspace_other = jax.lax.select(jnp.linalg.norm(s_grad) > np_eps, jnp.vstack([s_newt, normalize(s_grad)]).T, jnp.vstack([s_newt, jnp.zeros(s_newt.shape)]).T)


    case0_cond = jnp.logical_and(posdef, jnp.linalg.norm(og_s_newt) < delta)
    subspace = jax.lax.select(case0_cond, subspace_0, subspace_other)


    ### reduce to subspace ###
    chess = subspace.T.dot(shess.dot(subspace))
    cg = subspace.T.dot(sg)

    ### compute step ###
    sc_nd, _ = solve_nd_trust_region_subproblem_jitted(
        chess,
        cg,
        jnp.sqrt(jnp.maximum(delta**2 - jnp.linalg.norm(ss0) ** 2, 0.0)),
    )
    sc_1 = solve_1d_trust_region_subproblem(shess, sg, subspace[:, 0], delta, ss0)
    sc_1d = jnp.zeros_like(sc_nd).at[0].set(1) * sc_1

    sc = jax.lax.select(jnp.linalg.matrix_rank(subspace) == 1, sc_1d, sc_nd)
    # sc = sc_nd

    ss = subspace.dot(jnp.real(sc))
    s = scaling.dot(ss)

    ### step back ###
    # create copies of the calculated step
    og_s = s.copy()
    og_ss = ss.copy()
    og_sc = sc.copy()

    # br quantifies the distance to the boundary normalized
    # by the proposed step, this indicates the fraction of the step
    # that would put the respective variable at the boundary
    # This is defined in [Coleman-Li1994] (3.1)
    nonzero = jnp.abs(s) > 0
    _br = jnp.inf * jnp.ones(s.shape)
    br = jnp.where(
        nonzero,
        jnp.max(jnp.vstack([(ub - x) / s,(lb - x) / s,]),axis=0),
        _br
    )

    minbr = jnp.min(br)
    iminbr = jnp.argmin(br)

    # compute the minimum of the step
    alpha = jnp.min(jnp.array([1, theta * minbr]))

    s = s * alpha
    sc = sc * alpha
    ss = ss * alpha

    qpval = quadratic_form(shess, sg, ss + ss0)

    x_new = x + s + s0
    return {
        'x_new': x_new,
        'qpval': qpval,
        'dv': dv,
        's': s,
        's0': s0,
        'ss': ss,
        'ss0': ss0,
        'scaling': scaling,
        'theta': theta,
        'alpha': alpha,
        'br': br,
        'cg': cg,
        'chess': chess,
        'delta': delta,
        'iminbr': iminbr,
        'lb': lb,
        'minbr': minbr,
        'og_s': og_s,
        'og_sc': og_sc,
        'og_ss': og_ss,
        'sc': sc,
        'shess': shess,
        'subspace': subspace,
        'ub': ub,
        'x': x,
        'posdef': posdef,
        'sg': sg,
        'sc_1d': sc_1d,
        'sc_nd': sc_nd,
    }

@dataclass
class StepInfo:
    x_new: jnp.ndarray
    qpval: jnp.ndarray
    dv: jnp.ndarray
    s: jnp.ndarray
    s0: jnp.ndarray
    ss: jnp.ndarray
    ss0: jnp.ndarray
    scaling: jnp.ndarray
    theta: jnp.ndarray
    alpha: jnp.ndarray
    br: jnp.ndarray
    cg: jnp.ndarray
    chess: jnp.ndarray
    delta: float
    iminbr: jnp.ndarray
    lb: jnp.ndarray
    minbr: jnp.ndarray
    og_s: jnp.ndarray
    og_sc: jnp.ndarray
    og_ss: jnp.ndarray
    sc: jnp.ndarray
    shess: jnp.ndarray
    subspace: jnp.ndarray
    ub: jnp.ndarray
    x: jnp.ndarray
    posdef: bool
    sg: jnp.ndarray
    sc_1d: jnp.ndarray
    sc_nd: jnp.ndarray
    
    type: str = 'tr2d'

def tr_wrapped(x, grad, hess, lb, ub, theta_max, delta):
    res = tr_iteration(x, grad, hess, lb, ub, theta_max, delta)
    return StepInfo(**res)
