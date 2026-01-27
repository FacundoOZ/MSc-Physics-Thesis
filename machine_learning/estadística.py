
# Comentar

#============================================================================================================================================
# Tesis de Licenciatura | 
#============================================================================================================================================

import numpy as np

EPS = 1e-6


# ============================================================
# Helper: split window into left / center / right
# ============================================================

def _split_window(x: np.ndarray):
    n = len(x)
    if n < 3:
        return x, x, x
    return (
        x[: n // 3],
        x[n // 3 : 2 * n // 3],
        x[2 * n // 3 :]
    )


# ============================================================
# R-like quantities (distance, density proxies, etc.)
# ============================================================

def estadística_R(x: np.ndarray) -> list[float]:
    """
    Estadísticas sensibles a asimetría temporal y saltos.
    Adecuadas para detectar transiciones tipo bow shock.
    """
    if len(x) < 3:
        return [0.0] * 10

    xL, xC, xR = _split_window(x)
    dx = np.gradient(x)

    media = x.mean()

    i_max = np.argmax(np.abs(dx))
    pos_shock = i_max / len(dx)

    res = [
        media,                              # media global
        x.std(),                            # variabilidad global
        xR.mean() - xL.mean(),              # salto medio (clave)
        xR.std() / (xL.std() + EPS),         # asimetría de ruido
        np.median(x),                       # robusto
        np.percentile(x, 75) - np.percentile(x, 25),  # IQR
        x.max() - x.min(),                  # amplitud total
        pos_shock,                          # posición del salto (0–1)
        np.sum(np.abs(dx[:i_max])),         # actividad upstream
        np.sum(np.abs(dx[i_max:])),         # actividad downstream
    ]
    return res


# ============================================================
# B magnitude (|B|)
# ============================================================

def estadística_B(x: np.ndarray) -> list[float]:
    """
    Magnitud del campo magnético:
    enfatiza discontinuidades y coherencia del salto.
    """
    if len(x) < 3:
        return [0.0] * 9

    dx = np.gradient(x)
    media = x.mean()

    i_max = np.argmax(np.abs(dx))
    pos_shock = i_max / len(dx)

    res = [
        media,                              # media |B|
        x.std(),                            # fluctuaciones
        np.max(np.abs(dx)),                 # salto máximo
        dx.std(),                           # variabilidad del gradiente
        np.percentile(np.abs(dx), 95),      # gradiente fuerte típico
        pos_shock,                          # localización del shock
        np.sum(np.abs(dx[:i_max])),         # upstream activity
        np.sum(np.abs(dx[i_max:])),         # downstream activity
        np.max(np.abs(dx)) / (np.mean(np.abs(dx)) + EPS),  # sharpness
    ]
    return res


# ============================================================
# Componentes vectoriales (Bx, By, Bz, X, Y, Z)
# ============================================================

def estadística_componentes_B(x: np.ndarray) -> list[float]:
    """
    Componentes individuales del campo:
    captura coherencia direccional del salto.
    """
    if len(x) < 3:
        return [0.0] * 8

    dx = np.gradient(x)

    i_max = np.argmax(np.abs(dx))
    pos_shock = i_max / len(dx)

    res = [
        x.std(),                            # variabilidad
        np.max(np.abs(x)),                 # amplitud
        dx.std(),                           # gradiente RMS
        np.max(np.abs(dx)),                # salto máximo
        np.percentile(np.abs(dx), 95),     # gradiente extremo típico
        pos_shock,                          # dónde ocurre
        np.mean(np.sign(dx)),              # coherencia direccional
        np.max(np.abs(dx)) / (np.mean(np.abs(dx)) + EPS),  # sharpness
    ]
    return res


def estadística_componentes_R(x: np.ndarray) -> list[float]:
    """
    Componentes geométricas (Xpc, Ypc, Zpc, Xss, ...):
    menos bruscas, pero útiles para contexto.
    """
    if len(x) < 3:
        return [0.0] * 7

    xL, xC, xR = _split_window(x)

    res = [
        x.mean(),
        x.std(),
        xR.mean() - xL.mean(),              # deriva espacial
        np.median(x),
        np.percentile(x, 25),
        np.percentile(x, 75),
        x.max() - x.min(),
    ]
    return res