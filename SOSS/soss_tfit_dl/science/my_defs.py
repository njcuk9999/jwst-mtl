#define trend function
def trend_func(tfit,bpass):
    trend=tfit.trends['x1'][bpass]*tfit.p0[tfit.trends['c1i'],bpass]

    for n in range(1,tfit.n_trends):
        trend+=tfit.trends[f'x{n}'][bpass]*tfit.p0[tfit.trends[f'c{n}i'],bpass]
                                           
    return trend