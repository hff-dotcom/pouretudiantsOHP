"""MesurerRaie  — ajustement de raies spectrales (application Streamlit).

Point d'entrée prêt pour Streamlit Community Cloud.

Lancement local :
    streamlit run MesurerRaie_appv4.py

L'application lit un spectre FITS 1D, reconstruit l'axe en longueur d'onde à
partir du WCS, puis ajuste un profil gaussien, une double gaussienne ou un
profil de Voigt. Le mode « détection automatique » recherche plusieurs raies
d'absorption et ajuste chacune d'elles.

Les résultats obtenus doivent être considérés comme des supports d'apprentissage 
et ne doivent pas être utilisés comme des résultats scientifiques validés.
"""

from __future__ import annotations

import io
import warnings
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # backend non interactif : indispensable côté serveur

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from astropy.io import fits
from scipy.ndimage import median_filter
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.signal import find_peaks
from scipy.special import wofz

APP_NAME = "MesurerRaie"
MIN_POINTS = 8
MAXFEV = 30000
WCS_KEYS = (
    "CRVAL1",
    "CDELT1",
    "CRPIX1",
    "CD1_1",
    "CUNIT1",
    "BUNIT",
    "OBJECT",
    "INSTRUME",
)

st.set_page_config(
    page_title=f"{APP_NAME} — mesure de raies spectrales",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

warnings.filterwarnings("ignore", category=OptimizeWarning)


# ---------------------------------------------------------------------------
# Modèles de profils
# ---------------------------------------------------------------------------


def gaussian(x, amplitude, mean, sigma, offset):
    return offset + amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma**2))


def double_gaussian(x, amp1, mean1, sigma1, amp2, mean2, sigma2, offset):
    return (
        offset
        + amp1 * np.exp(-((x - mean1) ** 2) / (2 * sigma1**2))
        + amp2 * np.exp(-((x - mean2) ** 2) / (2 * sigma2**2))
    )


def voigt(x, amplitude, mean, sigma, gamma, offset):
    """Profil de Voigt normalisé : ``amplitude`` = écart pic/continuum."""
    sigma = max(abs(sigma), 1e-9)
    gamma = abs(gamma)
    z = ((x - mean) + 1j * gamma) / (sigma * np.sqrt(2.0))
    z0 = 1j * gamma / (sigma * np.sqrt(2.0))
    profile = wofz(z).real
    norm = wofz(z0).real
    if not np.isfinite(norm) or norm == 0:
        norm = 1.0
    return offset + amplitude * profile / norm


def fwhm_from_sigma(sigma):
    return 2.3548200450309493 * abs(sigma)  # 2*sqrt(2*ln2)


def fwhm_voigt(sigma, gamma):
    """Approximation d'Olivero & Longbothum."""
    fwhm_g = fwhm_from_sigma(sigma)
    fwhm_l = 2.0 * abs(gamma)
    return 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l**2 + fwhm_g**2)


# ---------------------------------------------------------------------------
# Estimations initiales
# ---------------------------------------------------------------------------


def _span(x: np.ndarray) -> float:
    return float(np.ptp(x)) if x.size else 0.0


def guess_single(x, y):
    offset = float(np.median(y))
    imax, imin = float(np.max(y)), float(np.min(y))
    span = _span(np.asarray(x, dtype=float))
    sigma = span / 50.0 if span > 0 else 1.0

    if (imax - offset) >= (offset - imin):
        index = int(np.argmax(y))
        amplitude = imax - offset
    else:
        index = int(np.argmin(y))
        amplitude = imin - offset
    return [amplitude, float(np.asarray(x)[index]), sigma, offset]


def guess_voigt(x, y):
    amplitude, mean, sigma, offset = guess_single(x, y)
    return [amplitude, mean, max(sigma, 1e-6), max(sigma / 2.0, 1e-6), offset]


def guess_double(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    offset = float(np.median(y))
    imax, imin = float(np.max(y)), float(np.min(y))
    span = _span(x)
    sigma = span / 80.0 if span > 0 else 1.0

    emission = (imax - offset) >= (offset - imin)
    amplitude = (imax if emission else imin) - offset
    center = float(x[int(np.argmax(y) if emission else np.argmin(y))])

    # Cherche deux extrema distincts pour initialiser correctement un doublet.
    signal = y if emission else -y
    depth = abs(amplitude)
    distance = max(1, len(x) // 20)
    peaks, _ = find_peaks(signal, distance=distance, prominence=max(0.1 * depth, 1e-12))

    if peaks.size >= 2:
        strongest = np.argsort(signal[peaks])[::-1][:2]
        first, second = sorted((int(peaks[strongest[0]]), int(peaks[strongest[1]])))
        return [
            float(y[first] - offset),
            float(x[first]),
            sigma,
            float(y[second] - offset),
            float(x[second]),
            sigma,
            offset,
        ]

    # Repli : deux composantes rapprochées autour de l'extremum principal.
    delta = max(sigma, 2.0 * float(np.median(np.abs(np.diff(x)))))
    return [
        amplitude,
        center - delta / 2.0,
        sigma,
        amplitude * 0.6,
        center + delta / 2.0,
        sigma,
        offset,
    ]


PROFILE_SPECS: Dict[str, Dict[str, Any]] = {
    "Gaussienne": {
        "func": gaussian,
        "guess": guess_single,
        "lower": (-np.inf, -np.inf, 1e-9, -np.inf),
        "upper": (np.inf, np.inf, np.inf, np.inf),
    },
    "Voigt": {
        "func": voigt,
        "guess": guess_voigt,
        "lower": (-np.inf, -np.inf, 1e-9, 0.0, -np.inf),
        "upper": (np.inf, np.inf, np.inf, np.inf, np.inf),
    },
    "Gaussienne double": {
        "func": double_gaussian,
        "guess": guess_double,
        "lower": (-np.inf, -np.inf, 1e-9, -np.inf, -np.inf, 1e-9, -np.inf),
        "upper": (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
    },
}


# ---------------------------------------------------------------------------
# Lecture FITS et axe de longueur d'onde
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Lecture du fichier FITS…", max_entries=2)
def read_fits_candidates(file_bytes: bytes) -> List[Dict[str, Any]]:
    """Retourne toutes les extensions 1D exploitables du fichier FITS."""
    candidates: List[Dict[str, Any]] = []

    with fits.open(io.BytesIO(file_bytes), memmap=False) as hdul:
        for index, hdu in enumerate(hdul):
            if hdu.data is None:
                continue
            try:
                array = np.asarray(hdu.data, dtype=float).squeeze()
            except (TypeError, ValueError):
                continue

            if array.ndim != 1 or array.size < MIN_POINTS:
                continue
            if np.isfinite(array).sum() < MIN_POINTS:
                continue

            header = {
                key: hdu.header.get(key)
                for key in WCS_KEYS
                if key in hdu.header
            }
            name = hdu.name or "PRIMARY"
            candidates.append(
                {
                    "index": int(index),
                    "label": f"HDU {index} — {name} ({array.size} points)",
                    "data": np.ascontiguousarray(array, dtype=float),
                    "header": header,
                }
            )

    if not candidates:
        raise ValueError(
            "Aucune extension 1D exploitable n'a été trouvée. Le fichier doit "
            "contenir un spectre sous forme de tableau 1D (NAXIS = 1)."
        )
    return candidates


def get_wavelength_axis(
    header: Dict[str, Any], n_points: int
) -> Tuple[np.ndarray, bool, str]:
    """Reconstruit l'axe spectral depuis CRVAL1 / CDELT1 / CRPIX1.

    Repli sur CD1_1 si CDELT1 est absent. Si aucune calibration n'est
    disponible, retourne les indices de pixels.
    """
    crval1 = header.get("CRVAL1")
    crpix1 = header.get("CRPIX1", 1.0)
    cdelt1 = header.get("CDELT1", header.get("CD1_1"))

    if crval1 is None or cdelt1 is None or float(cdelt1) == 0.0:
        return (
            np.arange(n_points, dtype=float),
            False,
            "⚠️ Aucune calibration WCS exploitable (CRVAL1/CDELT1 manquants). "
            "L'axe est affiché en pixels bruts.",
        )

    pixels = np.arange(n_points, dtype=float)
    wavelength = float(crval1) + (pixels - (float(crpix1) - 1.0)) * float(cdelt1)

    unit = str(header.get("CUNIT1", "Angstrom")).strip().lower()
    factors = {
        "angstrom": 1.0,
        "ang": 1.0,
        "a": 1.0,
        "nm": 10.0,
        "nanometer": 10.0,
        "nanometre": 10.0,
        "um": 1e4,
        "micron": 1e4,
        "micrometer": 1e4,
        "micrometre": 1e4,
        "m": 1e10,
        "meter": 1e10,
        "metre": 1e10,
    }
    factor = factors.get(unit)
    note = ""
    if factor is None:
        factor = 1.0
        note = f" (unité « {unit} » supposée en Å)"
    elif factor != 1.0:
        note = f" (converti depuis « {unit} »)"

    wavelength = wavelength * factor
    message = (
        "Axe reconstruit depuis le header : "
        f"CRVAL1={float(crval1):.4f}, CDELT1={float(cdelt1):.6f}, "
        f"CRPIX1={crpix1}{note}"
    )
    return wavelength, True, message


def prepare_spectrum(
    candidate: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, bool, str, int]:
    """Nettoie les valeurs non finies et construit l'axe spectral."""
    intensity = np.array(candidate["data"], dtype=float)
    finite = np.isfinite(intensity)
    n_bad = int((~finite).sum())

    if n_bad:
        indices = np.arange(intensity.size)
        intensity[~finite] = np.interp(
            indices[~finite], indices[finite], intensity[finite]
        )

    wavelength, wcs_ok, message = get_wavelength_axis(
        candidate["header"], intensity.size
    )
    return wavelength, intensity, wcs_ok, message, n_bad


# ---------------------------------------------------------------------------
# Ajustement
# ---------------------------------------------------------------------------


def fit_profile(
    x: np.ndarray,
    y: np.ndarray,
    profile: str,
    robust: bool = False,
) -> Dict[str, Any]:
    """Ajuste un profil et retourne paramètres, erreurs et courbe."""
    spec = PROFILE_SPECS[profile]
    n_params = len(spec["lower"])

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < n_params + 2:
        raise ValueError(
            f"Pas assez de points ({x.size}) pour ajuster ce profil "
            f"({n_params} paramètres)."
        )

    lower = np.asarray(spec["lower"], dtype=float)
    upper = np.asarray(spec["upper"], dtype=float)
    p0 = np.asarray(spec["guess"](x, y), dtype=float)
    p0 = np.clip(p0, lower, upper)
    finite_lower = np.isfinite(lower)
    p0[finite_lower] = np.maximum(p0[finite_lower], lower[finite_lower] + 1e-9)

    kwargs: Dict[str, Any] = {}
    if robust:
        amplitude = float(np.ptp(y))
        kwargs["loss"] = "soft_l1"
        kwargs["f_scale"] = max(0.1 * amplitude, 1e-12)

    popt, pcov = curve_fit(
        spec["func"],
        x,
        y,
        p0=p0,
        bounds=(lower, upper),
        method="trf",
        maxfev=MAXFEV,
        **kwargs,
    )

    with np.errstate(invalid="ignore"):
        errors = np.sqrt(np.diag(pcov))
    errors = np.where(np.isfinite(errors), errors, np.nan)

    curve = spec["func"](x, *popt)
    residuals = y - curve
    dof = max(x.size - n_params, 1)
    redchi = float(np.sum(residuals**2) / dof)

    return {
        "profile": profile,
        "params": popt,
        "errors": errors,
        "curve": curve,
        "x": x,
        "y": y,
        "redchi": redchi,
        "dof": dof,
    }


# ---------------------------------------------------------------------------
# Détection multi-raies
# ---------------------------------------------------------------------------


def estimate_continuum(intensity: np.ndarray, kernel_frac: float = 0.08) -> np.ndarray:
    """Estime le continuum par filtre médian glissant."""
    n = len(intensity)
    if n < 3:
        return np.array(intensity, dtype=float)

    kernel = int(n * kernel_frac)
    if kernel % 2 == 0:
        kernel += 1
    kernel = max(kernel, 5)
    if kernel > n:
        kernel = n if n % 2 == 1 else n - 1
    return median_filter(intensity, size=kernel, mode="nearest")


def detect_and_fit(
    x: np.ndarray,
    y: np.ndarray,
    profile: str,
    prominence_pct: float,
    distance: int,
    half_width: int,
    max_peaks: int,
    smoothing: int,
    robust: bool,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Détecte les creux d'absorption puis ajuste chaque raie."""
    continuum = estimate_continuum(y)
    depth = np.clip(continuum - y, 0.0, None)

    if smoothing and smoothing > 1:
        depth_for_peaks = median_filter(depth, size=int(smoothing), mode="nearest")
    else:
        depth_for_peaks = depth

    baseline = float(np.median(np.abs(continuum)))
    if baseline <= 0:
        baseline = float(np.ptp(y))
    if baseline <= 0:
        raise ValueError("Le spectre est constant : aucune raie détectable.")

    # Bruit robuste estimé sur les différences successives (MAD).
    noise = 1.4826 * float(np.median(np.abs(np.diff(y)))) / np.sqrt(2.0)
    prominence = max(
        baseline * float(prominence_pct) / 100.0,
        2.0 * noise,
        1e-12,
    )
    peaks, _ = find_peaks(
        depth_for_peaks, prominence=prominence, distance=int(distance)
    )
    if peaks.size == 0:
        return continuum, []

    if peaks.size > max_peaks:
        strongest = np.argsort(depth[peaks])[::-1][:max_peaks]
        peaks = np.sort(peaks[strongest])

    fits: List[Dict[str, Any]] = []
    for number, peak in enumerate(peaks, start=1):
        lo = max(int(peak) - int(half_width), 0)
        hi = min(int(peak) + int(half_width) + 1, x.size)
        x_local, y_local = x[lo:hi], y[lo:hi]

        row: Dict[str, Any] = {
            "#": number,
            "pixel": int(peak),
            "centre": float(x[peak]),
            "amplitude": np.nan,
            "sigma": np.nan,
            "gamma": np.nan,
            "fwhm": np.nan,
            "centre_err": np.nan,
            "redchi": np.nan,
            "statut": "—",
            "curve_x": None,
            "curve_y": None,
        }

        try:
            result = fit_profile(x_local, y_local, profile, robust=robust)
            params = result["params"]
            errors = result["errors"]
            _, centre, sigma = params[0], params[1], params[2]
            amplitude = params[0]
            offset = params[-1]

            if profile == "Voigt":
                gamma = abs(params[3])
                fwhm = fwhm_voigt(sigma, gamma)
            else:
                gamma = np.nan
                fwhm = fwhm_from_sigma(sigma)

            row.update(
                {
                    "centre": float(centre),
                    "amplitude": float(amplitude),
                    "sigma": float(abs(sigma)),
                    "gamma": float(gamma),
                    "fwhm": float(fwhm),
                    "centre_err": float(errors[1]),
                    "redchi": result["redchi"],
                    "statut": "OK",
                    "curve_x": x_local,
                    "curve_y": result["curve"],
                    "offset": float(offset),
                }
            )
        except Exception as exc:  # ajustement non convergé, raie conservée
            row["statut"] = f"Échec ({type(exc).__name__})"

        fits.append(row)

    return continuum, fits


# ---------------------------------------------------------------------------
# Mise en forme et figures
# ---------------------------------------------------------------------------


def fmt_number(value: float, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    if value != 0 and (abs(value) >= 1e5 or abs(value) < 1e-3):
        return f"{value:.4e}"
    return f"{value:.{digits}f}"


def fmt_with_unit(value: float, unit: str = "", digits: int = 4) -> str:
    text = fmt_number(value, digits)
    if text == "—":
        return text
    return f"{text} {unit}".strip()


def fmt_compact(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:.6g}"


def error_help(error: float) -> str:
    if np.isfinite(error):
        return f"Incertitude : ± {fmt_number(error)}"
    return "Incertitude non estimée."


def figure_to_png(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    return buffer.getvalue()


def show_figure(fig, filename: str, key: str) -> None:
    st.pyplot(fig)
    png = figure_to_png(fig)
    plt.close(fig)
    st.download_button(
        "⬇️ Télécharger la figure (PNG)",
        data=png,
        file_name=filename,
        mime="image/png",
        key=key,
    )


def dataframe_for_display(frame: pd.DataFrame) -> pd.DataFrame:
    """Arrondit les colonnes numériques pour l'affichage."""
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].apply(
                lambda value: "" if pd.isna(value) else f"{value:.5g}"
            )
    return display


def plot_raw_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    x_fit: np.ndarray,
    axis_label: str,
):
    fig, ax = plt.subplots(figsize=(11.0, 3.4))
    ax.plot(x, y, color="black", linewidth=0.8, label="Spectre")
    ax.axvspan(
        float(np.min(x_fit)),
        float(np.max(x_fit)),
        color="orange",
        alpha=0.12,
        label="Plage analysée",
    )
    ax.set_xlabel(axis_label)
    ax.set_ylabel("Intensité")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_single_result(
    result: Dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    x_fit: np.ndarray,
    axis_label: str,
):
    profile = result["profile"]
    params = result["params"]
    func = PROFILE_SPECS[profile]["func"]
    y_model = func(x, *params)

    fig, (ax, ax_res) = plt.subplots(
        2,
        1,
        figsize=(11.0, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0]},
    )

    ax.plot(x, y, color="black", linewidth=0.8, label="Spectre")
    ax.plot(x, y_model, color="tab:red", linewidth=1.6, label="Ajustement")

    if profile == "Gaussienne double":
        amp1, mean1, sigma1, amp2, mean2, sigma2, offset = params
        ax.plot(
            x,
            gaussian(x, amp1, mean1, sigma1, offset),
            "--",
            color="tab:blue",
            linewidth=1.0,
            alpha=0.8,
            label="Composante 1",
        )
        ax.plot(
            x,
            gaussian(x, amp2, mean2, sigma2, offset),
            "--",
            color="tab:green",
            linewidth=1.0,
            alpha=0.8,
            label="Composante 2",
        )

    ax.axvspan(
        float(np.min(x_fit)),
        float(np.max(x_fit)),
        color="orange",
        alpha=0.08,
        label="Plage ajustée",
    )
    ax.set_ylabel("Intensité")
    ax.set_title("Ajustement du spectre")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)

    ax_res.axhline(0.0, color="gray", linewidth=0.8)
    ax_res.plot(x, y - y_model, color="tab:purple", linewidth=0.8)
    ax_res.axvspan(float(np.min(x_fit)), float(np.max(x_fit)), color="orange", alpha=0.08)
    ax_res.set_xlabel(axis_label)
    ax_res.set_ylabel("Résidus")
    ax_res.grid(alpha=0.2)

    fig.tight_layout()
    return fig


def plot_detection_result(
    x: np.ndarray,
    y: np.ndarray,
    x_fit: np.ndarray,
    continuum: np.ndarray,
    fits: List[Dict[str, Any]],
    axis_label: str,
):
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.plot(x, y, color="black", linewidth=0.8, label="Spectre")
    ax.plot(
        x_fit,
        continuum,
        "--",
        color="gray",
        linewidth=1.0,
        alpha=0.8,
        label="Continuum estimé",
    )

    colors = plt.cm.tab20.colors
    fitted = [row for row in fits if row["statut"] == "OK"]
    for index, row in enumerate(fitted):
        ax.plot(
            row["curve_x"],
            row["curve_y"],
            color=colors[index % len(colors)],
            linewidth=1.5,
            label=f"Raie {row['#']}" if len(fitted) <= 12 else None,
        )
        ax.axvline(row["centre"], color="tab:red", linestyle=":", alpha=0.25)

        local_x = np.asarray(row["curve_x"])
        local_y = np.asarray(row["curve_y"])
        if local_x.size:
            y_at_centre = local_y[int(np.argmin(np.abs(local_x - row["centre"])))]
            ax.annotate(
                f"#{row['#']}",
                (row["centre"], y_at_centre),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=7,
                color=colors[index % len(colors)],
            )

    ax.set_xlabel(axis_label)
    ax.set_ylabel("Intensité")
    ax.set_title("Raies détectées et ajustées")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Blocs d'affichage des résultats
# ---------------------------------------------------------------------------


def render_single_result(
    result: Dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    x_fit: np.ndarray,
    axis_label: str,
    unit: str,
    robust: bool,
) -> None:
    profile = result["profile"]
    params = result["params"]
    errors = result["errors"]

    if profile == "Gaussienne":
        amplitude, centre, sigma, offset = params
        amp_err, centre_err, sigma_err, offset_err = errors
        gamma = None
        gamma_err = None
        fwhm = fwhm_from_sigma(sigma)
    else:
        amplitude, centre, sigma, gamma, offset = params
        amp_err, centre_err, sigma_err, gamma_err, offset_err = errors
        gamma = abs(gamma)
        fwhm = fwhm_voigt(sigma, gamma)

    st.subheader("Résultats de l'ajustement")

    columns = st.columns(4)
    columns[0].metric(
        "Centre", fmt_with_unit(centre, unit), help=error_help(centre_err)
    )
    columns[1].metric("FWHM", fmt_with_unit(fwhm, unit))
    columns[2].metric(
        "Amplitude", fmt_number(amplitude), help=error_help(amp_err)
    )
    columns[3].metric(
        "Continuum", fmt_number(offset), help=error_help(offset_err)
    )

    rows = [
        {
            "Paramètre": "Centre",
            "Valeur": centre,
            "Erreur": centre_err,
            "Unité": unit,
        },
        {
            "Paramètre": "Amplitude",
            "Valeur": amplitude,
            "Erreur": amp_err,
            "Unité": "",
        },
        {
            "Paramètre": "σ (gaussien)",
            "Valeur": abs(sigma),
            "Erreur": sigma_err,
            "Unité": unit,
        },
    ]
    if gamma is not None:
        rows.append(
            {
                "Paramètre": "γ (lorentzien)",
                "Valeur": gamma,
                "Erreur": gamma_err,
                "Unité": unit,
            }
        )
    rows.extend(
        [
            {
                "Paramètre": "FWHM",
                "Valeur": fwhm,
                "Erreur": np.nan,
                "Unité": unit,
            },
            {
                "Paramètre": "Continuum (offset)",
                "Valeur": offset,
                "Erreur": offset_err,
                "Unité": "",
            },
            {
                "Paramètre": "χ² réduit",
                "Valeur": result["redchi"],
                "Erreur": np.nan,
                "Unité": "",
            },
        ]
    )
    table = pd.DataFrame(rows)
    st.dataframe(
        dataframe_for_display(table),
        hide_index=True,
        width="stretch",
    )

    figure = plot_single_result(result, x, y, x_fit, axis_label)
    show_figure(
        figure,
        f"ajustement_{profile.lower().replace(' ', '_')}.png",
        f"dl_single_fig_{profile}",
    )

    st.download_button(
        "⬇️ Télécharger les paramètres (CSV)",
        data=table.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"parametres_{profile.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        key=f"dl_single_csv_{profile}",
    )
    st.caption(
        f"χ² réduit = {fmt_number(result['redchi'])} · "
        f"{result['x'].size} points ajustés · "
        f"ajustement robuste : {'oui' if robust else 'non'}"
    )


def render_double_result(
    result: Dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    x_fit: np.ndarray,
    axis_label: str,
    unit: str,
    robust: bool,
) -> None:
    amp1, mean1, sigma1, amp2, mean2, sigma2, offset = result["params"]
    e = result["errors"]

    st.subheader("Résultats de l'ajustement (double gaussienne)")

    columns = st.columns(4)
    columns[0].metric(
        "Centre 1", fmt_with_unit(mean1, unit), help=error_help(e[1])
    )
    columns[1].metric(
        "Centre 2", fmt_with_unit(mean2, unit), help=error_help(e[4])
    )
    columns[2].metric("FWHM 1", fmt_with_unit(fwhm_from_sigma(sigma1), unit))
    columns[3].metric("FWHM 2", fmt_with_unit(fwhm_from_sigma(sigma2), unit))

    columns = st.columns(4)
    columns[0].metric("Amplitude 1", fmt_number(amp1), help=error_help(e[0]))
    columns[1].metric("Amplitude 2", fmt_number(amp2), help=error_help(e[3]))
    columns[2].metric("Séparation Δλ", fmt_with_unit(abs(mean2 - mean1), unit))
    columns[3].metric("Continuum", fmt_number(offset), help=error_help(e[6]))

    rows = [
        {
            "Composante": "1",
            "Paramètre": "Centre",
            "Valeur": mean1,
            "Erreur": e[1],
            "Unité": unit,
        },
        {
            "Composante": "1",
            "Paramètre": "Amplitude",
            "Valeur": amp1,
            "Erreur": e[0],
            "Unité": "",
        },
        {
            "Composante": "1",
            "Paramètre": "σ",
            "Valeur": abs(sigma1),
            "Erreur": e[2],
            "Unité": unit,
        },
        {
            "Composante": "1",
            "Paramètre": "FWHM",
            "Valeur": fwhm_from_sigma(sigma1),
            "Erreur": np.nan,
            "Unité": unit,
        },
        {
            "Composante": "2",
            "Paramètre": "Centre",
            "Valeur": mean2,
            "Erreur": e[4],
            "Unité": unit,
        },
        {
            "Composante": "2",
            "Paramètre": "Amplitude",
            "Valeur": amp2,
            "Erreur": e[3],
            "Unité": "",
        },
        {
            "Composante": "2",
            "Paramètre": "σ",
            "Valeur": abs(sigma2),
            "Erreur": e[5],
            "Unité": unit,
        },
        {
            "Composante": "2",
            "Paramètre": "FWHM",
            "Valeur": fwhm_from_sigma(sigma2),
            "Erreur": np.nan,
            "Unité": unit,
        },
        {
            "Composante": "—",
            "Paramètre": "Continuum (offset)",
            "Valeur": offset,
            "Erreur": e[6],
            "Unité": "",
        },
        {
            "Composante": "—",
            "Paramètre": "χ² réduit",
            "Valeur": result["redchi"],
            "Erreur": np.nan,
            "Unité": "",
        },
    ]
    table = pd.DataFrame(rows)
    st.dataframe(
        dataframe_for_display(table),
        hide_index=True,
        width="stretch",
    )

    figure = plot_single_result(result, x, y, x_fit, axis_label)
    show_figure(figure, "ajustement_gaussienne_double.png", "dl_double_fig")

    st.download_button(
        "⬇️ Télécharger les paramètres (CSV)",
        data=table.to_csv(index=False).encode("utf-8-sig"),
        file_name="parametres_gaussienne_double.csv",
        mime="text/csv",
        key="dl_double_csv",
    )
    st.caption(
        f"χ² réduit = {fmt_number(result['redchi'])} · "
        f"{result['x'].size} points ajustés · "
        f"ajustement robuste : {'oui' if robust else 'non'}"
    )


def render_detection_result(
    x: np.ndarray,
    y: np.ndarray,
    x_fit: np.ndarray,
    continuum: np.ndarray,
    fits: List[Dict[str, Any]],
    axis_label: str,
    unit: str,
    profile: str,
) -> None:
    ok_rows = [row for row in fits if row["statut"] == "OK"]
    if not ok_rows:
        st.error(
            "Aucune raie n'a pu être ajustée. Essaie d'élargir la fenêtre "
            "d'ajustement ou de modifier les paramètres de détection."
        )
        return

    st.subheader("Raies détectées et ajustées")
    columns = st.columns(3)
    columns[0].metric("Raies détectées", str(len(fits)))
    columns[1].metric("Raies ajustées", str(len(ok_rows)))
    columns[2].metric(
        "FWHM médian",
        fmt_with_unit(
            float(np.nanmedian([row["fwhm"] for row in ok_rows])), unit
        ),
    )

    rows = []
    for row in fits:
        rows.append(
            {
                "#": row["#"],
                f"Centre ({unit})": row["centre"],
                "Amplitude": row["amplitude"],
                "σ": row["sigma"],
                "γ": row["gamma"],
                "FWHM": row["fwhm"],
                "± centre": row["centre_err"],
                "χ² réduit": row["redchi"],
                "Statut": row["statut"],
            }
        )
    table = pd.DataFrame(rows)
    st.dataframe(
        dataframe_for_display(table),
        hide_index=True,
        width="stretch",
    )

    if len(ok_rows) < len(fits):
        st.warning(
            f"{len(fits) - len(ok_rows)} raie(s) n'ont pas convergé ; "
            "elles restent visibles dans le tableau avec leur statut."
        )

    figure = plot_detection_result(x, y, x_fit, continuum, fits, axis_label)
    show_figure(figure, "raies_detectees.png", "dl_detection_fig")

    st.download_button(
        "⬇️ Télécharger le tableau (CSV)",
        data=table.to_csv(index=False, decimal=".").encode("utf-8-sig"),
        file_name="raies_detectees.csv",
        mime="text/csv",
        key="dl_detection_csv",
    )
    st.caption(
        f"Profil utilisé : {profile} · "
        f"{len(x_fit)} points analysés · "
        "les incertitudes proviennent de la matrice de covariance."
    )


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


def render_welcome() -> None:
    st.markdown(
        """
        ### Comment procéder

        1. Charge un **spectre FITS 1D** (`.fit`, `.fits`) ci-dessus.
        2. Choisis le type d'ajustement dans la barre latérale.
        3. Restreins éventuellement la plage spectrale autour de la raie visée.
        4. Télécharge les paramètres, le tableau ou la figure.

        L'axe en longueur d'onde est reconstruit depuis les mots-clés FITS
        `CRVAL1`, `CDELT1` et `CRPIX1` (ou `CD1_1`). En l'absence de
        calibration, le spectre est affiché en pixels.
        
        ## Avertissement
        
        Cette application a été développée à des fins exclusivement pédagogiques 
        pour aider les étudiants. Elle n’est pas destinée à un usage scientifique.
        """
    )


def main() -> None:
    st.title("🔭 SpectroFit — mesure de raies spectrales")
    st.caption(
        "Ajustement gaussien, double gaussien et Voigt d'un spectre FITS, "
        "avec détection automatique de raies d'absorption."
    )

    with st.sidebar:
        st.header("Configuration")
        fit_mode = st.radio(
            "Type d'analyse",
            [
                "Gaussienne simple",
                "Gaussienne double",
                "Profil de Voigt",
                "Détection automatique",
            ],
            horizontal=False,
        )
        robust = st.checkbox(
            "Ajustement robuste",
            value=False,
            help=(
                "Utilise une perte soft_l1, moins sensible aux valeurs "
                "aberrantes (cosmiques, bruit impulsif)."
            ),
        )

        with st.expander("À propos des modèles"):
            st.markdown(
                """
                - **Gaussienne** : profil symétrique, adapté aux raies
                  instrumentales.
                - **Double gaussienne** : deux composantes centrées librement.
                - **Voigt** : convolution gaussienne × lorentzienne, adaptée
                  aux profils élargis par pression ou par vitesse.
                - **Détection automatique** : estime le continuum par filtre
                  médian, détecte les creux, puis ajuste chaque raie.
                """
            )

        st.divider()
        st.caption(
            "Les fichiers ne sont pas conservés : le traitement a lieu "
            "pendant la session Streamlit."
        )

    uploaded_file = st.file_uploader(
        "Fichier FITS du spectre",
        type=["fit", "fits", "fts"],
        help="Le fichier doit contenir un tableau 1D (NAXIS = 1).",
    )

    if uploaded_file is None:
        render_welcome()
        return

    try:
        candidates = read_fits_candidates(uploaded_file.getvalue())
    except Exception as exc:
        st.error(f"Impossible de lire le fichier FITS : {exc}")
        return

    if len(candidates) > 1:
        labels = [candidate["label"] for candidate in candidates]
        selected = st.selectbox(
            "Extension FITS à analyser",
            options=list(range(len(candidates))),
            format_func=lambda index: labels[index],
        )
    else:
        selected = 0

    candidate = candidates[selected]

    try:
        wavelength, intensity, wcs_ok, wcs_message, n_bad = prepare_spectrum(
            candidate
        )
    except Exception as exc:
        st.error(f"Impossible de préparer le spectre : {exc}")
        return

    if wcs_ok:
        st.success(wcs_message)
    else:
        st.warning(wcs_message)

    if n_bad:
        st.warning(
            f"{n_bad} valeur(s) non finie(s) ont été interpolées avant l'analyse."
        )

    unit = "Å" if wcs_ok else "px"
    axis_label = "Longueur d'onde (Å)" if wcs_ok else "Pixel"

    info_columns = st.columns(3)
    info_columns[0].metric("Points", f"{intensity.size}")
    info_columns[1].metric(
        "Plage",
        f"{fmt_compact(float(np.min(wavelength)))} – "
        f"{fmt_compact(float(np.max(wavelength)))} {unit}",
    )
    step = np.median(np.abs(np.diff(wavelength))) if wavelength.size > 1 else np.nan
    info_columns[2].metric("Pas médian", fmt_with_unit(float(step), unit, 4))

    header = candidate["header"]
    details = []
    if header.get("OBJECT"):
        details.append(f"Objet : {header['OBJECT']}")
    if header.get("INSTRUME"):
        details.append(f"Instrument : {header['INSTRUME']}")
    if header.get("BUNIT"):
        details.append(f"Unité d'intensité : {header['BUNIT']}")
    if details:
        st.caption(" · ".join(details))

    # --- Restriction de la plage spectrale --------------------------------
    wmin = float(np.min(wavelength))
    wmax = float(np.max(wavelength))
    if wmax > wmin:
        with st.expander("✂️ Restreindre la plage spectrale (optionnel)"):
            span = wmax - wmin
            range_lo, range_hi = st.slider(
                "Plage analysée",
                min_value=wmin,
                max_value=wmax,
                value=(wmin, wmax),
                step=span / 1000.0,
                format="%.6g",
            )
        mask = (wavelength >= range_lo) & (wavelength <= range_hi)
    else:
        mask = np.ones(wavelength.size, dtype=bool)

    if int(mask.sum()) < MIN_POINTS:
        st.error(
            f"La plage sélectionnée contient moins de {MIN_POINTS} points. "
            "Élargis la plage."
        )
        return

    x_fit = wavelength[mask]
    y_fit = intensity[mask]

    with st.expander("👁️ Aperçu du spectre", expanded=False):
        preview = plot_raw_spectrum(wavelength, intensity, x_fit, axis_label)
        show_figure(preview, "spectre.png", "dl_preview_fig")

    # --- Analyse -----------------------------------------------------------
    try:
        if fit_mode == "Détection automatique":
            with st.expander("⚙️ Paramètres de détection", expanded=True):
                detection_profile = st.radio(
                    "Profil utilisé pour chaque raie",
                    ["Gaussienne", "Voigt"],
                    horizontal=True,
                )
                prominence_pct = st.slider(
                    "Sensibilité de détection (proéminence, % du continuum)",
                    min_value=0.5,
                    max_value=20.0,
                    value=3.0,
                    step=0.5,
                )
                half_width = st.slider(
                    "Demi-largeur de la fenêtre d'ajustement (points)",
                    min_value=5,
                    max_value=300,
                    value=40,
                    step=5,
                )
                min_distance = st.slider(
                    "Distance minimale entre raies (points)",
                    min_value=1,
                    max_value=300,
                    value=20,
                    step=1,
                )
                smoothing = st.slider(
                    "Lissage avant détection (points, 1 = aucun)",
                    min_value=1,
                    max_value=31,
                    value=1,
                    step=2,
                )
                max_peaks = st.slider(
                    "Nombre maximal de raies ajustées",
                    min_value=1,
                    max_value=100,
                    value=30,
                    step=1,
                )

            with st.spinner("Détection et ajustement des raies…"):
                continuum, fits = detect_and_fit(
                    x_fit,
                    y_fit,
                    detection_profile,
                    prominence_pct=prominence_pct,
                    distance=min_distance,
                    half_width=half_width,
                    max_peaks=max_peaks,
                    smoothing=smoothing,
                    robust=robust,
                )

            if not fits:
                st.warning(
                    "Aucune raie d'absorption détectée avec ces paramètres. "
                    "Augmente la sensibilité (proéminence plus faible) ou "
                    "réduis la distance minimale entre raies."
                )
            else:
                render_detection_result(
                    wavelength,
                    intensity,
                    x_fit,
                    continuum,
                    fits,
                    axis_label,
                    unit,
                    detection_profile,
                )

        else:
            profile = {
                "Gaussienne simple": "Gaussienne",
                "Gaussienne double": "Gaussienne double",
                "Profil de Voigt": "Voigt",
            }[fit_mode]

            with st.spinner("Ajustement en cours…"):
                result = fit_profile(x_fit, y_fit, profile, robust=robust)

            if profile == "Gaussienne double":
                render_double_result(
                    result, wavelength, intensity, x_fit, axis_label, unit, robust
                )
            else:
                render_single_result(
                    result, wavelength, intensity, x_fit, axis_label, unit, robust
                )

    except Exception as exc:
        st.error(
            "L'ajustement a échoué : "
            f"{exc}\n\nEssaie de restreindre la plage autour de la raie visée, "
            "de choisir un autre profil ou d'activer l'ajustement robuste."
        )

    st.divider()
    st.caption(
        f"{APP_NAME} · Spectres FITS 1D · "
        "Gaussienne, double gaussienne, Voigt et détection multi-raies."
    )


if __name__ == "__main__":
    main()
