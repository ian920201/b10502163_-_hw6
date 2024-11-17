import sympy as sp
import hw6_function as fc

# define transfrom matrix
theta1, theta2, theta4, d1, d2, d3, a1, a2 = sp.symbols('theta1 theta2 theta4 d1 d2 -d3 a1 a2')
t = sp.symbols('t')
theta1 = sp.Function('theta1')(t)
theta2 = sp.Function('theta2')(t)
d3 = sp.Function('-d3')(t)
T_0_1 = fc.T(theta1, d1, 0, 0)
T_1_2 = fc.T(theta2, d2, a1, 0)
T_2_3 = fc.T(0, d3, a2, 0)
T_3_4 = fc.T(theta4, 0, 0, 0)

T = T_0_1 * T_1_2 * T_2_3 * T_3_4

# define condition
g = sp.symbols('g')
M = sp.symbols('M')
a1, a2 = sp.symbols('a1 a2')
a3 = 0
r1, r2, r3 = sp.symbols('r1 r2 r3')
m1, m2, m3 = sp.symbols('m1 m2 m3')
I1, I2, I3 = sp.symbols('I1 I2 I3')
I4 = 0
theta1_dot, theta2_dot, theta4_dot = sp.symbols('theta1_dot theta2_dot theta4_dot')
theta1_dot_dot, theta2_dot_dot, theta4_dot_dot = sp.symbols('theta1_dot_dot theta2_dot_dot theta4_dot_dot')
theta3_dot = 0
theta3_dot_dot = 0
d1_dot, d2_dot, d3_dot = sp.symbols('d1_dot d2_dot -d3_dot')
d1_dot_dot, d2_dot_dot, d3_dot_dot = sp.symbols('d1_dot_dot d2_dot_dot -d3_dot_dot')
d4, d4_dot, d4_dot_dot = 0, 0, 0
theta1_dot = sp.Function('theta1_dot')(t)
theta2_dot = sp.Function('theta2_dot')(t)
d3_dot = sp.Function('-d3_dot')(t)
theta1_dot_dot = sp.Function('theta1_dot_dot')(t)
theta2_dot_dot = sp.Function('theta2_dot_dot')(t)
d3_dot_dot = sp.Function('-d3_dot_dot')(t)
w00 = sp.Matrix([[0], [0], [0]])
w00_dot = sp.Matrix([[0], [0], [0]])
a00 = sp.Matrix([[0], [0], [g]])
p01 = T_0_1[:3, -1]
p12 = T_1_2[:3, -1]
p23 = T_2_3[:3, -1]
p34 = T_3_4[:3, -1]
pc11 = sp.Matrix([[r1], [0], [0]])
pc22 = sp.Matrix([[r2], [0], [0]])
pc33 = sp.Matrix([[0], [0], [r3]])
pc44 = sp.Matrix([[0], [0], [0]])
r01 = T_0_1[:3, :3]
r12 = T_1_2[:3, :3]
r23 = T_2_3[:3, :3]
r34 = T_3_4[:3, :3]
r10 = r01.transpose()
r21 = r12.transpose()
r32 = r23.transpose()
r43 = r34.transpose()

# compute w and a
w11 = fc.R_w_n(r10, w00, theta1_dot)
w11_dot = fc.R_w_n_dot(r10, w00_dot, w00, theta1_dot, theta1_dot_dot)
a11 = fc.R_a_n(r10, a00, p01, w00, w00_dot)
ac1 = fc.R_ac_n(a11, w11_dot, pc11, w11)

w22 = fc.R_w_n(r21, w11, theta2_dot)
w22_dot = fc.R_w_n_dot(r21, w11_dot, w11, theta2_dot, theta2_dot_dot)
a22 = fc.R_a_n(r21, a11, p12, w11, w11_dot)
ac2 = fc.R_ac_n(a22, w22_dot, pc22, w22)

w33 = fc.P_w_n(w22)
w33_dot = fc.P_w_n_dot(r32, w22_dot)
a33 = fc.P_a_n(r32, a22, w22_dot, p23, w22, w33, d3_dot, d3_dot_dot)
ac3 = fc.P_ac_n(a33, w33_dot, pc33, w33)

w44 = fc.P_w_n(w33)
w44_dot = fc.P_w_n_dot(r43, w33_dot)
a44 = fc.P_a_n(r43, a33, w33_dot, p34, w33, w44, d4_dot, d4_dot_dot)
ac4 = fc.P_ac_n(a44, w44_dot, pc44, w44)

w10 = r10 * w11
w21 = r21 * w22
w20 = w21 + w10
w30 = w20
w40 = w20

# compute kinetic energy
identity = sp.eye(fc.square(w22).shape[0])
k1 = 0.5 * m1 * r1**2 * fc.square(w10) + 0.5 * I1 * fc.square(w10)
k2 = 0.5 * m2 * ((a1 * theta1_dot)**2 + (r2 * (theta1_dot + theta2_dot))**2) * identity + 0.5 * I2 * fc.square(w20)
k3 = 0.5 * m3 * (((a1 * theta1_dot)**2 + (r2 * (theta1_dot + theta2_dot))**2) + d3_dot**2) * identity + 0.5 * I3 * fc.square(w30)
k4 = 0.5 * M * (((a1 * theta1_dot)**2 + (r2 * (theta1_dot + theta2_dot))**2) + d3_dot**2) * identity
K = k1 + k2 + k3 + k4

# compute potential energy
U = m1 * (-g) * d1 + m2 * (-g) * (d1 + d2) + m3 * (-g) * (d1 + d2 - d3 + r3) + M * (-g) * (d1 + d2 - d3)

# compute lagrange equation
K_theta1_dot = sp.diff(K, theta1_dot)
K_theta2_dot = sp.diff(K, theta2_dot)
K_d3_dot = sp.diff(K, d3_dot)

replacements = {
    sp.Derivative(theta1, t): theta1_dot,
    sp.Derivative(theta2, t): theta2_dot,
    sp.Derivative(d3, t): d3_dot,
    sp.Derivative(theta1_dot, t): theta1_dot_dot,
    sp.Derivative(theta2_dot, t): theta2_dot_dot,
    sp.Derivative(d3_dot, t): d3_dot_dot,
    theta1: sp.symbols('theta1'),
    theta2: sp.symbols('theta2'),
    d3: sp.symbols('d3'),
    theta1_dot: sp.symbols('theta1_dot'),
    theta2_dot: sp.symbols('theta2_dot'),
    d3_dot: sp.symbols('-d3_dot'),
    theta1_dot_dot: sp.symbols('theta1_dot_dot'),
    theta2_dot_dot: sp.symbols('theta2_dot_dot'),
    d3_dot_dot: sp.symbols('-d3_dot_dot')
}

K_theta1_dot_t = sp.diff(K_theta1_dot, t)
K_theta2_dot_t = sp.diff(K_theta2_dot, t)
K_d3_dot_t = sp.diff(K_d3_dot, t)
K_theta1_dot_t = K_theta1_dot_t.subs(replacements)
K_theta2_dot_t = K_theta2_dot_t.subs(replacements)
K_d3_dot_t = K_d3_dot_t.subs(replacements)


K_theta1 = sp.diff(K, theta1)
K_theta2 = sp.diff(K, theta2)
K_d3 = sp.diff(K, d3)

U_theta1 = sp.Matrix([sp.diff(U, theta1)])
U_theta2 = sp.Matrix([sp.diff(U, theta2)])
U_d3 = sp.Matrix([sp.diff(U, d3)])

tau1 = K_theta1_dot_t - K_theta1 + U_theta1
tau2 = K_theta2_dot_t - K_theta2 + U_theta2
tau3 = K_d3_dot_t - K_d3 + U_d3

tau1 = sp.simplify(tau1)
tau2 = sp.simplify(tau2)
tau3 = sp.simplify(tau3)

tau1_sym, tau2_sym, tau3_sym = sp.symbols('tau1 tau2 tau3')

print('1. ', end='')
sp.pprint(tau1_sym)
print()
sp.pprint(tau1)
print()

print('2. ', end='')
sp.pprint(tau2_sym)
print()
sp.pprint(tau2)
print()

print('3. ', end='')
sp.pprint(tau3_sym)
print()
sp.pprint(tau3)
print()
