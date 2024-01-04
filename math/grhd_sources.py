# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import sympy as sp
import sympy_utils as spu
import gr_utils as gru
import rewrite_utils as rwu
sp.init_printing()

# # Generating unrolled expressions for GRHD source terms
#
# See eqs (25) in arXiv:1104.4751v3

# setting up vars
beta = sp.Matrix(sp.symbols(r'\beta{1:5}')); display(beta)
alpha = sp.symbols(r'\alpha'); display(alpha)
dalpha = sp.Matrix(sp.symbols(r'\partial\alpha{1:4}')); display(dalpha)
dbeta = sp.Matrix(3,3,sp.symbols(r'\partial\beta{1:4}{1:4}')); display(dbeta)
dgamma = [ sp.Matrix(3,3,sp.symbols(r'\partial\gamma{}{{1:4}}{{1:4}}'.format(i))) for i in range(1,4) ]; display(dgamma)
gamma = sp.Matrix(3,3,sp.symbols(r'\gamma{{1:4}{1:4}}')); display(gamma)
K = sp.Matrix(3,3,sp.symbols(r'K{{1:4}{1:4}}')); display(K)
g = sp.Matrix(4,4,sp.symbols(r'g{{1:5}{1:5}}')); display(g)
Tuu = sp.Matrix(4,4,sp.symbols(r'Tuu{{1:5}{1:5}}')); display(Tuu)
sTuu = Tuu[:-1,:-1]; display(sTuu)
Tud = sp.Matrix(4,4,sp.symbols(r'Tud{{1:5}{1:5}}')); display(Tuu)
sTud = Tud[:,:-1]; display(sTud)

# +
# dictionary with substiution rules for printing results
subs_dict = {
  r'\partial' : '∂',
  r'\gamma'   : 'γ',
  r'\beta'    : 'β',
  r'\alpha'   : 'α',
  '{'         : '',
  '}'         : '',
  '**'        : '^',
}

def subsymbols(ex):
    str_ex = str(ex)
    for key, val in subs_dict.items():
        str_ex = str_ex.replace(key, val)
    return str_ex


# -

# $$
# s_{S,k} = T^{00} (\frac{1}{2} \beta^i \beta^j \partial_k \gamma_{ij} - \alpha \partial_k \alpha) + T^{0i} \beta^j \partial_k \gamma_{ij} + T^0_i \partial_k \beta^i + \frac{1}{2} T^{ij} \partial_k \gamma_{ij}
# $$

term1 = sp.zeros(3,1)
oh = sp.Rational(1,2)
for k in range(3):
    tmp = 0
    for i in range(3):
        for j in range(3):
            tmp += oh * beta[i] * beta[j] * dgamma[k][i,j]
    tmp += - alpha * dalpha[k]
    term1[k] = Tuu[0,0] * tmp
display(term1)

term2 = sp.zeros(3,1)
for k in range(3):
    tmp = 0
    for i in range(3):
        for j in range(3):
            tmp += sTuu[0,i] * beta[j] * dgamma[k][i,j]
    term2[k] = tmp
display(term2)

term3 = sp.zeros(3,1)
for k in range(3):
    tmp = 0
    for i in range(3):
        tmp += sTud[0,i] * dbeta[k,i]
    term3[k] = tmp
display(term3)

term4 = sp.zeros(3,1)
for k in range(3):
    tmp = 0
    for i in range(3):
        for j in range(3):
            tmp += sTuu[i,j] * dgamma[k][i,j]
    term4[k] = oh * tmp
display(term4)

s_S = term1 + term2 + term3 + term4
#s_S = term2 + term3 + term4
#s_S = term4
s_S

for k in range(3):
    print(f's_S{k+1} =',subsymbols(s_S[k]))

# $$
# s_{\tau} = T^{00} (\beta^i \beta^j K_{ij} - \beta^i \partial_i \alpha) + T^{0i} (2 \beta^j K_{ij} - \partial_i \alpha) + T^{ij} K_{ij}
# $$

term1 = 0
for i in range(3):
    for j in range(3):
        term1 += beta[i] * beta[j] * K[i,j]
    term1 +=  - beta[i] * dalpha[i]
term1 *= Tuu[0,0]
display(term1)

term2 = 0
for i in range(3):
    for j in range(3):
        term2 += sTuu[0,i] * 2 * beta[j] * K[i,j]
    term2 +=  - sTuu[0,i] * dalpha[i]
display(term2)

term3 = 0
for i in range(3):
    for j in range(3):
        term3 += sTuu[i,j] * K[i,j]
display(term3)

s_tau = term1 + term2 + term3
s_tau

print(f's_tau{k+1} =', subsymbols(s_tau))

# ### Write output to file

import os
# one attempt on getting the notebook's directory path:
# https://stackoverflow.com/questions/52119454/how-to-obtain-jupyter-notebooks-path
thisdir = os.path.abspath("")
outputname = os.path.normpath(os.path.join(thisdir,"grhd_sources.ref.txt"))
print(outputname)

with open(outputname, "w") as file:
    file.write("# generated with math/grhd_sources.ipynb\n")
    for k in range(3):
        file.write(f's_S{k+1} = ' + subsymbols(s_S[k]) + '\n')
    file.write(f's_tau = ' + subsymbols(s_tau) + '\n')


