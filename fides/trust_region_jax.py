import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
np_eps = jnp.finfo(jnp.float64).eps

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
    s  = jnp.real(slam(0, w, eigvals, eigvecs))
    norm_s = jnp.linalg.norm(s)
    thresh = delta + jnp.sqrt(np_eps)
    posdef_cond = jnp.all(jnp.array([(mineig > 0), (norm_s <= thresh)]))

    s = jax.lax.select(posdef_cond, s, hard_case(w, mineig, eigvals, eigvecs, delta, laminit, jmin))
    hess_case = jax.lax.select(posdef_cond, 0, 1)
    return s, hess_case

def solve_nd_trust_region_subproblem(B, g, delta):
    if delta == 0:
        return jnp.zeros(g.shape), 'zero'

    cases = ['posdef', 'hard']
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

    s_newt = -jnp.linalg.lstsq(shess, sg)[0]
    # lstsq only returns absolute ev values
    e, v_ = jnp.linalg.eig(shess)
    posdef = jnp.min(jnp.real(e)) > -np_eps * jnp.max(jnp.abs(e))

    # if len(sg) == 1:
    #     s_newt = -sg[0] / self.shess[0]
    #     self.subspace = np.expand_dims(s_newt, 1)
    #     return



    ###### NOTE                                                                                                      #####
    ###### replace case0 subspace from jnp.expand_dims(s_newt, 1) to jnp.vstack([s_newt, jnp.zeros(s_newt.shape)]).T #####
    ###### this has the effect of always forcing a "hard" solution without gradient direction                        #####


    s_newt_ = normalize(s_newt)
    subspace_0 = jnp.vstack([s_newt_, jnp.zeros(s_newt_.shape)]).T


    s_newt = jnp.real(v_[:, jnp.argmin(jnp.real(e))])
    s_grad = jax.lax.select(posdef, sg.copy(), scaling.dot(jnp.sign(sg) + (sg == 0)))
    s_newt = normalize(s_newt)
    s_grad = s_grad - s_newt * s_newt.dot(s_grad)
    subspace_other = jax.lax.select(jnp.linalg.norm(s_grad) > np_eps, jnp.vstack([s_newt, normalize(s_grad)]).T, jnp.vstack([s_newt, jnp.zeros(s_newt.shape)]).T)


    case0_cond = jnp.all(jnp.array([posdef, jnp.linalg.norm(s_newt) < delta]))
    subspace = jax.lax.select(case0_cond, subspace_0, subspace_other)


    ### reduce to subspace ###
    chess = subspace.T.dot(shess.dot(subspace))
    cg = subspace.T.dot(sg)

    ### compute step ###
    sc, _ = solve_nd_trust_region_subproblem_jitted(
        chess,
        cg,
        jnp.sqrt(jnp.maximum(delta**2 - jnp.linalg.norm(ss0) ** 2, 0.0)),
    )
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
        'theta': theta
    }

from dataclasses import dataclass

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
    type: str = 'tr2d'

def tr_wrapped(x, grad, hess, lb, ub, theta_max, delta):
    res = tr_iteration(x, grad, hess, lb, ub, theta_max, delta)
    return StepInfo(**res)
