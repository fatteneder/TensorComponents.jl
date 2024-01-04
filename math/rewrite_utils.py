import sympy as sp
from sympy.core.function import UndefinedFunction, Function, AppliedUndef, Symbol


fmt_bracket = r"\{{{}\,{}\}}^{{{}}}"


def get_name(expr):
    if isinstance(expr, UndefinedFunction):
        return expr.__name__
    elif isinstance(expr, AppliedUndef):
        return expr.func.__name__
    elif isinstance(expr, Symbol):
        return expr.name
    else:
        raise Exception(f"Failed to extract name of '{expr}' which is of type '{type(expr)}'")


def bracket_plus(A, B, fwrd_rules=None, bwrd_rules=None):
    return _bracket(A, B, '+', fwrd_rules, bwrd_rules)


def bracket_minus(A, B, fwrd_rules=None, bwrd_rules=None):
    return _bracket(A, B, '-', fwrd_rules, bwrd_rules)



def _bracket(A, B, sign, fwrd_rules=[], bwrd_rules=[]):
    r, theta = sp.symbols(r"r \theta")
    #assert(isinstance(A, AppliedUndef))
    #assert(isinstance(B, AppliedUndef))
    assert(sign == '+' or sign == '-')
    name_A, name_B = get_name(A), get_name(B)
    brABs = sp.symbols(fmt_bracket.format(name_A, name_B, sign))
    Ar, Br, At, Bt = A.diff(r), A.diff(theta), B.diff(r), B.diff(theta)
    sign = 1 if sign == '+' else -1
    if isinstance(fwrd_rules, list):
        rule = (Ar, sp.sqrt(brABs - sign * At**2 / r**2))
        if rule not in fwrd_rules:
            fwrd_rules.append(rule)
    if isinstance(bwrd_rules, list):
        rule = (brABs, Ar**2 + sign * At**2 / r**2)
        if rule not in bwrd_rules:
            bwrd_rules.append(rule)
    return brABs


#######################################################################
#                                Tests                                #
# ######################################################################


def test_bracket():
    rules, bwrd_rules = [], []
    r, theta = sp.symbols(r"r \theta")
    U = sp.Function('U')(r,theta)
    brUUp = bracket_plus(U, U, rules, bwrd_rules)
    brUUp = bracket_plus(U, U, rules, bwrd_rules) # don't add things twice
    brUUm = bracket_minus(U, U, rules, bwrd_rules)
    display(brUUp)
    display(brUUm)
    display(rules)
    display(bwrd_rules)


""
test_bracket()

""

