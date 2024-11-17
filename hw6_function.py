import sympy as sp

def R_w_n(r, w_n_1, theta_dot):
    z = sp.Matrix([[0], [0], [1]])
    ans = r * w_n_1 + theta_dot * z
    return ans

def R_w_n_dot(r, w_n_1_dot, w_n_1, theta_dot, theta_dot_dot):
    z = sp.Matrix([[0], [0], [1]])
    ans = r * w_n_1_dot + r * (w_n_1.cross(theta_dot * z)) + theta_dot_dot * z
    return ans

def R_a_n(r, a_n_1, p, w, w_dot):
    ans = r * (a_n_1 + (w_dot.cross(p)) + (w.cross(w.cross(p))))
    return ans

def R_ac_n(a_n, w_dot, pc, w):
    ans = a_n + (w_dot.cross(pc)) + (w.cross(w.cross(pc)))
    return ans

def P_w_n(w_n_1):
    return w_n_1

def P_w_n_dot(r, w_n_1_dot):
    return r * w_n_1_dot

def P_a_n(r, a_n_1, w_dot, p, w, w_1_n, d_dot, d_dot_dot):
    z = sp.Matrix([[0], [0], [1]])
    ans = r * (a_n_1 + w_dot.cross(p) + (w.cross(w.cross(p)))) + 2 * (w_1_n.cross(d_dot * z)) + d_dot_dot * z
    return ans

def P_ac_n(a_n_1, w_dot, pc, w):
    ans = a_n_1 + w_dot.cross(pc) + (w.cross(w.cross(pc)))
    return ans

def T(theta_i, di, ai_1, alpha_i_1):
    ans = sp.Matrix([[sp.cos(theta_i), -sp.sin(theta_i), 0, ai_1],
                     [sp.sin(theta_i) * sp.cos(alpha_i_1), sp.cos(theta_i) * sp.cos(alpha_i_1), -sp.sin(alpha_i_1), -di * sp.sin(alpha_i_1)],
                     [sp.sin(theta_i) * sp.sin(alpha_i_1), sp.cos(theta_i) * sp.sin(alpha_i_1), sp.cos(alpha_i_1), di * sp.cos(alpha_i_1)],
                     [0, 0, 0, 1]])
    return ans

def square(x: sp.Matrix) -> sp.Matrix:
    return x.transpose() * x