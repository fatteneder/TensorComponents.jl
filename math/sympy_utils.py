import sympy as sp
from IPython.display import display, Math
from sympy.printing.latex import LatexPrinter, accepted_latex_functions
from sympy.core.function import UndefinedFunction, Function, AppliedUndef
from sympy import Symbol

class MyLatexPrinter(LatexPrinter):

    # Print derivatives with commas. E.g. `f(x).diff(x)` prints `U_{,x}`
    def _print_Derivative(self, expr):
        function, *vars = expr.args

        # not sure what this is supposed to do ...
        if not isinstance(type(function), UndefinedFunction): 
            #or not all(isinstance(i, Symbol) for i in vars):
            return super()._print_Derivative(expr)

        fmt = "{}_{{,{}}}"
        args = [ sp.symbols(str(v[0])*v[1]) for v in vars ]
        return fmt.format(self._print(Symbol(function.func.__name__)), 
                          ' '.join(self._print(a) for a in args))

    ### Added 'is_accepted_latex_fn' return argument. 
    def _modified_hprint_Function(self, func):
        r'''
        Logic to decide how to render a function to latex
          - if it is a recognized latex name, use the appropriate latex command
          - if it is a single letter, just use that letter
          - if it is a longer name, then put \operatorname{} around it and be
            mindful of undercores in the name
        '''
        func = self._deal_with_super_sub(func)
        is_accepted_latex_fn = False
        if func in accepted_latex_functions:
            name = r"\%s" % func
            is_accepted_latex_fn = True
        elif len(func) == 1 or func.startswith('\\'):
            name = func
        else:
            name = r"\operatorname{%s}" % func
        return name, is_accepted_latex_fn

    ### Copied from LatexPriner and replaced _hprint_Function with _modified_hprint_Function.
    # If returned value 'is_accepted_latex_fn' is true, then we print the arguments, 
    # otherwise we don't. E.g.
    # sin(x) prints as sin(x) but U(x) prints U
    def _print_Function(self, expr, exp=None):
        """
        Supress function arguments for non-standard functions.
        """
        func = expr.func.__name__
        if hasattr(self, '_print_' + func) and not isinstance(expr, AppliedUndef):
            return getattr(self, '_print_' + func)(expr, exp)

        else:
            args = [str(self._print(arg)) for arg in expr.args]
            # How inverse trig functions should be displayed, formats are:
            # abbreviated: asin, full: arcsin, power: sin^-1
            inv_trig_style = "abbreviated"
            # If we are dealing with a power-style inverse trig function
            inv_trig_power_case = False
            # If it is applicable to fold the argument brackets
            can_fold_brackets = False and len(args) == 1 and \
                not super()._needs_function_brackets(expr.args[0])

            inv_trig_table = [
                "asin", "acos", "atan", "acsc", "asec", "acot",
                "asinh", "acosh", "atanh", "acsch", "asech", "acoth",
            ]

            # If the function is an inverse trig function, handle the style
            if func in inv_trig_table:
                if inv_trig_style == "abbreviated":
                    pass
                elif inv_trig_style == "full":
                    func = ("ar" if func[-1] == "h" else "arc") + func[1:]
                elif inv_trig_style == "power":
                    func = func[1:]
                    inv_trig_power_case = True

                    # Can never fold brackets if we're raised to a power
                    if exp is not None:
                        can_fold_brackets = False

            if inv_trig_power_case:
                if func in accepted_latex_functions:
                    name = r"\%s^{-1}" % func
                else:
                    name = r"\operatorname{%s}^{-1}" % func

            elif exp is not None:
                func_tex, is_accepted_latex_fn = self._modified_hprint_Function(func)
                func_tex = super().parenthesize_super(func_tex)
                name = r'%s^{%s}' % (func_tex, exp)
            else:
                name, is_accepted_latex_fn = self._modified_hprint_Function(func)

            if is_accepted_latex_fn:
                if can_fold_brackets:
                    if func in accepted_latex_functions:
                        # Wrap argument safely to avoid parse-time conflicts
                        # with the function name itself
                        name += r" {%s}"
                    else:
                        name += r"%s"
                else:
                    name += r"{\left(%s \right)}"

                if inv_trig_power_case and exp is not None:
                    name += r"^{%s}" % exp
                return name % ",".join(args)
            else:
                return name


def my_doprint(expr, **kw):
    mpl = MyLatexPrinter()
    return mpl.doprint(expr)


def my_init_printing():
    sp.init_printing(latex_printer=my_doprint)


def display_nonzeros(T, coords, symmetric=False):
    shape = T.shape
    N = len(coords)
    assert(len(shape) == 2)
    assert(all([ si == N for si in shape ]) == True)
    for ij, Tij in enumerate(T):
        if Tij != 0:
            i = ij % N
            j = int((ij - i) / N)
            if symmetric and i > j: continue
            cij = (coords[i], coords[j])
            display(Math(my_doprint(cij) + ": " + my_doprint(Tij)))


def display_tensor2(fmt_tensor, T, coords, symmetric=False):
    shape = T.shape
    N = len(coords)
    assert(len(shape) == 2)
    assert(all([ si == N for si in shape ]) == True)
    for ij, Tij in enumerate(T):
        if Tij != 0:
            i = ij % N
            j = int((ij - i) / N)
            if symmetric and i > j: continue
            component = fmt_tensor.format(coords[i], coords[j])
            display(Math(component + my_doprint(Tij)))
    return


def display_math(*expr):
    terms = []
    for e in expr:
        if isinstance(e, str):
            terms.append(e)
        else:
            terms.append(my_doprint(e))
    display(Math(" ".join(terms)))

#######################################################################
#                                Tests                                #
# ######################################################################


def test_print_derivative():
    my_init_printing()
    x, y = sp.symbols('x y')
    f = sp.Function('f')
    dfxy = f(x,y).diff(x,y)
    return dfxy

def test_print_function_args():
    my_init_printing()
    x, y = sp.symbols('x y')
    f = sp.Function('f')(x,y)
    return f

def test_print_buildin_function_args():
    my_init_printing()
    x = sp.symbols('x')
    return sp.cos(x)
