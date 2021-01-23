import numpy
from scipy import constants
import matplotlib.pyplot as plt
from dispersion import dispersion_totale
from scipy import special
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib import rcParams
from scipy.integrate import odeint

rcParams.update({'figure.autolayout': True})
font = {'size': 12}
plt.rc('font', **font)


def numerical_aperture(n1, n2):
    NA = numpy.sqrt(n1 ** 2 - n2 ** 2)
    return NA


def normed_frequncy(a, wvelenght, NA):
    """
    Retourne le paramètre V (fréquence normalisée)

    :params:
    a = rayon du coeur de la fibre
    wvelenght = longueur d'onde
    n_0 = indice de réfraction de l'air
    NA = ouverture numérique
    """
    n_0 = 1
    k_0 = 2 * numpy.pi / wvelenght
    V = a * k_0 * n_0 * NA
    return V


def carac_LHS(u, ell):
    """
    Le côté gauche de l'équation
    caractéristique J_ell / J_{ell-1}.
    """
    return special.jv(ell, u) / special.jv(ell - 1, u)


def carac_RHS(u, ell, V):
    w = numpy.sqrt(V ** 2 - u ** 2)
    return (-1) * (u / w) * (special.kn(ell, w) / special.kn(ell - 1, w))


def mult_graph(func, intervals):
    """
    Retourne plusieurs array graphables [[x],[y]]
    pour une fonction.
    """
    for interval in intervals:
        x = numpy.linspace(interval[0] + 1 * 10 ** -4, interval[1], 100, endpoint=False)
        plt.axvline(x=interval[1], color="grey", linestyle="--")
        plt.plot(x, func(x), color="k", zorder=0)


def mult_solver(guess_lst=[[2, 4.8, 7.3], [3, 6], [4.5, 7.15], [5.5], [6.5]], l_lst=[0, 1, 2, 3, 4]):
    """
    On suggère des solutions
    approximatives à notre méthode de
    résolution numérique.
    """
    # On trouve les solutions numériques pour les valeurs de u
    res = []
    for l, guess in zip(l_lst, guess_lst):
        func = lambda u: carac_RHS(u, ell=l) - carac_LHS(u, ell=l)
        res.append(fsolve(func, guess))
    return res


def n_eff(u, NA=0.14, n2=1.54):
    """
    Retourne une approximation pour
    l'indice de réfraction effectif.
    """
    # flatten = lambda l: [item for sublist in l for item in sublist]
    V = normed_frequncy()
    u = numpy.array(u)
    V = numpy.full((len(u),), V)
    w = numpy.sqrt(V ** 2 - u ** 2)
    n1 = numpy.sqrt(NA ** 2 + n2 ** 2)
    delta = (n1 ** 2 - n2 ** 2) / 2 * n1 ** 2
    return n2 * (1 + delta * (w / V) ** 2)


def approx_param_psi(u, V):
    """
    Retourne le paramètre psi
    en fonction de u et de V.
    """
    return numpy.sqrt(V ** 2 - u ** 2) / (numpy.sqrt(V ** 2 - u ** 2) + 1)


def eq_diff(u, V):
    """
    Équation différentielle
    équivalente à l'équation
    caractéristique.
    """
    psi = approx_param_psi(u, V)
    du_dv = u / V * (1 - psi)
    return du_dv


def beam_size(a, V):
    w_0 = a * (0.65 + 1.619/V**(3/2) + 2.879/V**6)
    return w_0


def ode_solve(func, l, m):
    """
    Résoud l'équation différentielle avec une condition limite à l'infini
    comme étant le premier zéro de la fonction de Bessel.
    """
    V = numpy.linspace(100000000, 11.792, 10000)
    CI = special.jn_zeros(l, m)
    sol = odeint(func, CI, V)
    return sol


def myagi_eq(l, m, V):
    """
    Équation de Myagi pour trouver les modes
    qui se propagent dans une fibre.
    """
    u_infinity = special.jn_zeros(l, m)
    u = u_infinity * V / (V + 1) * (1 - u_infinity ** 2 / (6 * (V + 1) ** 3) - u_infinity ** 4 / (20 * (V + 1) ** 5))
    return u


def param_psi(u, V, l):
    """ Retourne le paramètre psi exact
    """
    w = numpy.sqrt(V ** 2 - u ** 2)
    psi = special.kv(l, w) ** 2 / (special.kv(l + 1, w) * special.kv(l - 1, w))
    return psi


def param_gamma(u, V):
    """
    :param u:
    :param V:
    :return:
    """
    psi = param_psi(u, V, l=0)
    gamma = 1 - (u / V) ** 2 * (1 - psi)
    return gamma

def germanium_doped_silicate_refraction_index(wavelenght, mol_frac):
    """

    :param wavelenght: Longueur d'onde de la lumière [um]
    :param mol_frac: fraction molaire de GeO
    :return: Indice de réfraction
    """
    A_i_Si_arr = numpy.array([0.696166, 0.407942, 0.897479])
    A_i_Ge_arr = numpy.array([0.806866, 0.718158, 0.854168])
    lam_Si_arr = numpy.array([0.068404, 0.116241, 9.896161])
    lam_Ge_arr = numpy.array([0.068972, 0.153966, 11.84193])
    sum = 0
    # wavelenght = numpy.repeat([wavelenght], 3, axis=0)
    for i in range(3):
        if not isinstance(wavelenght, float):
            wavelenght_len = len(wavelenght)
            A_i_Si = numpy.repeat(A_i_Si_arr[i], wavelenght_len)
            A_i_Ge = numpy.repeat(A_i_Ge_arr[i], wavelenght_len)
            lam_Si = numpy.repeat(lam_Si_arr[i], wavelenght_len)
            lam_Ge = numpy.repeat(lam_Ge_arr[i], wavelenght_len)
        else:
            A_i_Si = A_i_Si_arr[i]
            A_i_Ge = A_i_Ge_arr[i]
            lam_Si = lam_Si_arr[i]
            lam_Ge = lam_Ge_arr[i]

        sum_terms = (A_i_Si + mol_frac*(A_i_Ge - A_i_Si)) * wavelenght**2 / (wavelenght**2 - (lam_Si + mol_frac*(lam_Ge - lam_Si))**2)

        sum += sum_terms
    refractive_index = numpy.sqrt(sum+1)
    return refractive_index


def curvature_pulse_separator(n1, delta_t=1*10**-12, r=62.5*10**-6, **kwargs):
    """

    :param n1: Indice de réfraction du coeur de la fibre
    :param delta_t: Temps désiré entre les deux impulsions finales
    :param r: rayon total de la fibre
    :param R: rayon de courbure
    :param z: longueur courbée
    :return:
    """
    c = constants.c
    R = kwargs.get("R", None)
    z = kwargs.get("z", None)
    if z is None:
        z = R**2 * delta_t * c /( r**2 * 0.0435 * n1**3)
        return "Solutions pour z", z
    elif R is None:
        R = numpy.sqrt( (z * n1**3 * 0.0435 * r**2) / (c * delta_t) )
        return "Solutions pour R", R


def rayon_critique(n1, n2, r_coeur, wavelenght):
    """

    :param n1:
    :param n2:
    :param r_coeur:
    :return: Rayon critique à partir duquel les pertes deviennent très importantes [m]
    """
    numerical_aperture = numpy.sqrt(n1**2 - n2**2)
    V = normed_frequncy(a=r_coeur, wvelenght=wavelenght, NA=numerical_aperture)
    u = myagi_eq(l=0, m=2, V=V)[0]
    w = numpy.sqrt(V**2 - u**2)
    deltan = n1 - n2
    K_c = 4 * w**3 * deltan / (3 * r_coeur * V**2 * n2)
    R_cr = 10 * numpy.sqrt(2) / K_c
    return R_cr


def pertes_courbure(R, n1, n2, wavelenght, r_coeur=4.6*10**-6):
    """

    :param n1: indice de réfraction du coeur de la fibre
    :param n2: indice de réfraction de la gaine
    :param R: rayon de courbure de la bobine de fibre [m]
    :param r_coeur: rayon du coeur de la fibre [m]
    :return: alpha_c [dB/m]
    """
    numerical_aperture = numpy.sqrt(n1**2 - n2**2)
    V = normed_frequncy(a=r_coeur, wvelenght=wavelenght, NA=numerical_aperture)
    u = myagi_eq(l=0, m=2, V=V)[0]
    w = numpy.sqrt(V**2 - u**2)
    deltan = n1 - n2
    K_c = 4 * w**3 * deltan / (3 * r_coeur * V**2 * n2)
    A_c = (1 / 2) * numpy.sqrt(numpy.pi / (r_coeur * w**3)) * (u / (w * special.k1(w)))**2
    alpha_c = (A_c / numpy.sqrt(R)) * numpy.exp(-K_c * R)
    return alpha_c

def birefringence(r, R, n1):
    """
    Calcul de la biréfringence B pour une fibre
    de silice sans contrainte de tension.
    :param r: rayon de la fibre (gaine)
    :param R: rayon de courbure
    :param n1: indice de réfraction du coeur
    :return: Birefringence
    """
    B = 0.0435 * n1**3 * (r/R)**2
    return B

def lame_retard(n1, r, wavelenght, retard):
    """
    Retourne le rayon R auquel il faut faire un tour de
    fibre afin d'obtenir une lame de
    :param n1: indice du coeur
    :param r: rayon de la fibre (gaine incluse)
    :param wavelenght:
    :param retard: Fraction du retard ex: retard=1/2 pour un retard de lambda/2
    :return: Rayon, un seul tour nécéssaire
    """
    K = 0.0435 * n1**3
    R_lambda = 2 * K * numpy.pi * r**2 / wavelenght
    R = R_lambda / retard
    return R


def pertes_infrarouge(wvelenght=1.55*10**-6):
    """
    Pertes dans l'infrarouge causées par les vibration de la structure
    de la silice.
    :param wvelenght: longueur d'onde [m]
    :return: alpha_IR [dB/km]
    """
    micro_wavelenght = wvelenght*10**6
    alpha_IR = 7.81 * 10**11 * numpy.exp(-48.48 / micro_wavelenght)
    return alpha_IR


def pertes_Rayleigh(delta_n_Ge, wavelenght=1.55*10**-6):
    """
    La diffusion de Rayleigh est causée par des variations microscopiques de la densité du matériau
    :param delta_n_Ge: Différence de l'indice de réfraction causée par le dopage
    :param wvelenght: longueur d'onde [m]
    :return: alpha_Rayleigh (approximation)
    """
    micro_wavelenght = wavelenght * 10**6
    alpha_Rayleigh = (0.75 + 66 * delta_n_Ge) * micro_wavelenght**-4
    return alpha_Rayleigh


def pertes_imperfections():
    """
    :return: alpha_imperfections [dB/km]
    """
    alpha_imperfections = 0.05
    return alpha_imperfections


def gaussienne(t, t_0, largeur):
    """
    Retourne un faisceau gaussien
    :param t: axe temporel
    :param t_0: position du centre de la gaussienne
    :param largeur: durée du faisceau temporel
    :return: forme gaussienne du faisceau
    """
    std = (largeur/1.699) / 2.355
    return 1/(std*numpy.sqrt(2*numpy.pi)) * numpy.exp(-(t-t_0)**2 / (2*std**2))


def dispersion_temporelle(t_0, z, wvelenght=1.55*10**-6):
    """

    :param t_0: largeur initale du faisceau gaussien [s]
    :param z: distance propagée [m]
    :param wvelenght: longueur d'onde [m]
    :return: t_prime la nouvelle largeur du faisceau gaussien
    """
    c = constants.c*10**-3
    D = dispersion_totale(wvelenght)
    nano_wvelenght = wvelenght*10**9
    beta_2 = - nano_wvelenght**2 * D / (2 * numpy.pi * c)  # [ps^2 km^-1]
    L_D = t_0**2 / abs(beta_2)
    t_prime = t_0 * numpy.sqrt(1 + (z/L_D)**2)
    print(t_0, t_prime, L_D)
    print("ATTENTION, FICHIER DISPERSION INCOMPLET")
    return t_prime


if __name__ == "__main__":
    print(beam_size(a=3.95, V=2.00158))