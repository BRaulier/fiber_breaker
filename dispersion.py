import numpy as np
from scipy import constants


def dispersion_totale(longueur_donde):
    """

    :param longueur_donde: [m]
    :return: Dispersion totale [ps / ( km nm)]
    """
    # Définition des constantes

    longueur_donde_um = longueur_donde*10**6
    lam = np.linspace(1, 1.75, 751)
    c = constants.c

    a = 4.6e-6

    a1si = 0.696166
    a2si = 0.407942
    a3si = 0.897479
    l1si = 0.068404
    l2si = 0.116241
    l3si = 9.896161

    a1ge = 0.806866
    a2ge = 0.718158
    a3ge = 0.854168
    l1ge = 0.068972
    l2ge = 0.153966
    l3ge = 11.84193

    # Calcul de l'indice de la gaine


    n2 = np.sqrt((a1si * lam ** 2) / (lam ** 2 - l1si ** 2) + (a2si * lam ** 2) / (lam ** 2 - l2si ** 2)
                 + (a3si * lam ** 2) / (lam ** 2 - l3si ** 2) + 1)

    # Calcul de la deuxième dérivée de n2 par rapport à lambda

    lam = lam * 1e-6
    dn2 = np.diff(n2, 1)
    dlam = np.diff(lam, 1)
    n2p = dn2 / dlam
    lamf = 0.5 * (lam[:-1] + lam[1:])
    dn2p = np.diff(n2p, 1)
    dlamf = np.diff(lamf, 1)
    n2pp = dn2p / dlamf
    lams = 0.5 * (lamf[:-1] + lamf[1:])

    # Calcul de la dispersion matérielle et conversion en ps/(km*nm)

    Dm = -lams / c * n2pp
    Dm = Dm * 10**6

    # Fibre avec 4% molaire

    # Calcul de n1

    lam = lam * 1e6
    x = 0.04
    n1 = np.sqrt((a1si + x * (a1ge - a1si)) * lam ** 2 / (lam ** 2 - (l1si + x * (l1ge - l1si)) ** 2)
                 + (a2si + x * (a2ge - a2si)) * lam ** 2 / (lam ** 2 - (l2si + x * (l2ge - l2si)) ** 2)
                 + (a3si + x * (a3ge - a3si)) * lam ** 2 / (lam ** 2 - (l3si + x * (l3ge - l3si)) ** 2) + 1)

    # Calcul de V

    lam = lam * 1e-6
    k0 = 2 * np.pi / lam
    V = k0 * np.sqrt(n1 ** 2 - n2 ** 2) * a
    Vf = 0.5 * (V[:-1] + V[1:])
    Vs = 0.5 * (Vf[:-1] + Vf[1:])

    # Calcul de u

    u_inf = 2.40483
    u = u_inf * V / (V + 1) * (1 - u_inf ** 2 / (6 * (V + 1) ** 3) - u_inf ** 4 / (20 * (V + 1) ** 5))
    uf = 0.5 * (u[:-1] + u[1:])
    us = 0.5 * (uf[:-1] + uf[1:])

    # Calcul de w et de psi

    w = np.sqrt(Vs ** 2 - us ** 2)
    psi = w / (w + 1)

    # Calcul de (Vb)' et V(Vb)''

    Vbp = 1 - (us / Vs) ** 2 * (1 - 2 * psi)
    VVbpp = 2 * (us / Vs) ** 2 * (
                psi * (1 - 2 * psi) + 2 / w * (w ** 2 + us ** 2 * psi) * np.sqrt(psi) * (psi + 1 / w * np.sqrt(psi) - 1))

    # Défintion de n2 moyenné deux fois

    n2f = 0.5 * (n2[:-1] + n2[1:])
    n2s = 0.5 * (n2f[:-1] + n2f[1:])

    # Calcul de delta

    delta = (n1 - n2) / n1
    deltaf = 0.5 * (delta[:-1] + delta[1:])
    deltas = 0.5 * (deltaf[:-1] + deltaf[1:])

    # Calcul de la dispersion de guidage

    Dw1 = -n2s * deltas / (c * lams) * VVbpp * 10 ** (6)

    lams = lams * 1e6
    D = Dm + Dw1

    idx = list(lams).index(longueur_donde_um)
    return D[idx]
