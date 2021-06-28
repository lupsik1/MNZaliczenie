def p_norm(v,p):
    vp = [x**p for x in v]
    return sum(vp)**(1/p)
