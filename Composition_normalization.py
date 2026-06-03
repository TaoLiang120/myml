from pymatgen.core.composition import Composition
def extract_paratheses(s):
    l = len(s)
    matches = []
    start = 0
    end = 0
    while True:
        relative_s = s[start:].find("(")
        relative_e = s[end:].find(")")
        if relative_s == -1 or relative_e == -1:
            break
        matches.append(s[start+relative_s+1:end+relative_e])
        start += 1+relative_s
        end += 1+relative_e
    return matches

def compstr2frac_formula(compstr, decimals=4):
    comp = Composition(compstr)
    newstr = ""
    for iele in range(len(comp.elements)):
        el = comp.elements[iele]
        sym = el.symbol
        frac = comp.get_atomic_fraction(el)
        frac = round(frac, decimals)
        newstr += sym + str(frac)
    return newstr

def normalize_composition(compstr):
    matches = extract_paratheses(compstr)
    if len(matches) == 0:
        compstr = compstr2frac_formula(compstr)
    else:
        for i in range(len(matches)):
            c = matches[i]
            newc = compstr2frac_formula(c)
            compstr = compstr.replace(c, newc)
    return compstr

def get_pretty_formula(compstr, multiplier=100.0):
    comp = Composition(compstr)
    newstr = ""
    for iele in range(len(comp.elements)):
        el = comp.elements[iele]
        sym = el.symbol
        pct = comp.get_atomic_fraction(el) * multiplier
        pct = int(pct)
        newstr += sym + str(pct)
    comp = Composition(newstr)
    return comp.reduced_formula