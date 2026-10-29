import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.special import wofz
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

# ---------------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------------

def gaussian(x, amplitude, mean, sigma, offset):
    return offset + amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))


def double_gaussian(x, amp1, mean1, sigma1, amp2, mean2, sigma2, offset):
    return (
        offset
        + amp1 * np.exp(-((x - mean1) ** 2) / (2 * sigma1 ** 2))
        + amp2 * np.exp(-((x - mean2) ** 2) / (2 * sigma2 ** 2))
    )


def voigt(x, amplitude, mean, sigma, gamma, offset):
    """Profil de Voigt (convolution Gauss-Lorentz), normalisé pour que
    `amplitude` corresponde directement à l'écart pic/creux par rapport
    au continuum, comme pour `gaussian` (même convention)."""
    sigma = max(abs(sigma), 1e-6)
    gamma = abs(gamma)
    z = ((x - mean) + 1j * gamma) / (sigma * np.sqrt(2))
    z0 = 1j * gamma / (sigma * np.sqrt(2))
    profile = wofz(z).real
    norm = wofz(z0).real  # valeur du profil non normalisé au centre
    return offset + amplitude * profile / norm


def fwhm_from_sigma(sigma):
    return 2.3548200450309493 * sigma  # 2*sqrt(2*ln2)


def fwhm_voigt(sigma, gamma):
    # Approximation d'Olivero & Longbothum
    fwhm_g = fwhm_from_sigma(sigma)
    fwhm_l = 2 * gamma
    return 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l ** 2 + fwhm_g ** 2)


# ---------------------------------------------------------------------------
# Lecture du header WCS pour reconstruire l'axe de longueur d'onde
# ---------------------------------------------------------------------------

def get_wavelength_axis(header, n_points):
    """Reconstruit l'axe de longueur d'onde à partir des mots-clés WCS
    standards (CRVAL1 / CDELT1 / CRPIX1), avec repli sur CD1_1 si besoin.
    Retourne (wavelength_array, ok, message)."""

    crval1 = header.get("CRVAL1")
    crpix1 = header.get("CRPIX1", 1)
    cdelt1 = header.get("CDELT1", header.get("CD1_1"))

    if crval1 is not None and cdelt1 is not None:
        pixels = np.arange(n_points)  # indices 0-based
        wavelength = crval1 + (pixels - (crpix1 - 1)) * cdelt1
        return wavelength, True, (
            f"Axe reconstruit depuis le header : CRVAL1={crval1:.4f}, "
            f"CDELT1={cdelt1:.6f}, CRPIX1={crpix1}"
        )

    # Repli : pas d'information de calibration -> indices de pixels bruts
    return np.arange(n_points), False, (
        "⚠️ Aucune information de calibration WCS trouvée dans le header "
        "(CRVAL1/CDELT1 manquants). Affichage en pixels bruts."
    )


# ---------------------------------------------------------------------------
# Estimation automatique du point de départ (p0)
# ---------------------------------------------------------------------------

def guess_single(wavelength, intensity):
    offset = np.median(intensity)
    imax, imin = np.max(intensity), np.min(intensity)
    # Choix émission / absorption selon l'écart le plus fort par rapport au continuum
    if (imax - offset) >= (offset - imin):
        amplitude = imax - offset
        mean = wavelength[np.argmax(intensity)]
    else:
        amplitude = imin - offset  # négatif -> creux d'absorption
        mean = wavelength[np.argmin(intensity)]
    sigma = (wavelength[-1] - wavelength[0]) / 50.0
    return [amplitude, mean, sigma, offset]


def guess_voigt(wavelength, intensity):
    amplitude, mean, sigma, offset = guess_single(wavelength, intensity)
    gamma = sigma / 2.0  # part lorentzienne modeste au départ
    return [amplitude, mean, sigma, gamma, offset]


def guess_double(wavelength, intensity):
    offset = np.median(intensity)
    imax, imin = np.max(intensity), np.min(intensity)
    span = wavelength[-1] - wavelength[0]
    sigma = span / 80.0

    if (imax - offset) >= (offset - imin):
        amp = imax - offset
        center = wavelength[np.argmax(intensity)]
    else:
        amp = imin - offset
        center = wavelength[np.argmin(intensity)]

    # Deux composantes symétriques autour du pic/creux principal, légèrement décalées
    delta = span / 20.0
    return [
        amp, center - delta / 2, sigma,
        amp * 0.6, center + delta / 2, sigma,
        offset,
    ]


# ---------------------------------------------------------------------------
# Détection automatique de plusieurs raies d'absorption
# ---------------------------------------------------------------------------

def estimate_continuum(intensity, kernel_frac=0.08):
    """Estime le continuum par filtre médian glissant (robuste aux raies étroites)."""
    n = len(intensity)
    kernel = int(n * kernel_frac)
    if kernel % 2 == 0:
        kernel += 1
    kernel = max(kernel, 5)
    return median_filter(intensity, size=kernel, mode="nearest")


def detect_absorption_minima(wavelength, intensity, prominence, distance_px):
    continuum = estimate_continuum(intensity)
    depth = np.clip(continuum - intensity, 0, None)  # ne garde que les creux
    peaks_idx, _ = find_peaks(depth, prominence=prominence, distance=distance_px)
    return peaks_idx, continuum


# ---------------------------------------------------------------------------
# Interface Streamlit
# ---------------------------------------------------------------------------

st.title("Ajustement Gaussien d'un Spectre (.fit)")

fit_mode = st.radio(
    "Type d'ajustement",
    ["Gaussienne simple", "Gaussienne double", "Profil de Voigt", "Détection automatique (plusieurs raies)"],
    horizontal=True,
)

uploaded_file = st.file_uploader("Upload un fichier .fit", type=["fit", "fits"])

if uploaded_file:
    with fits.open(uploaded_file) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    intensity = np.asarray(data, dtype=float).squeeze()
    wavelength, wcs_ok, wcs_message = get_wavelength_axis(header, len(intensity))

    if wcs_ok:
        st.success(wcs_message)
    else:
        st.warning(wcs_message)

    # Optionnel : restreindre la plage pour affiner l'ajustement sur la raie d'intérêt
    with st.expander("Restreindre la plage de longueur d'onde (optionnel)"):
        wmin, wmax = float(wavelength.min()), float(wavelength.max())
        lo, hi = st.slider(
            "Plage (Å)", min_value=wmin, max_value=wmax, value=(wmin, wmax)
        )
        mask = (wavelength >= lo) & (wavelength <= hi)
        wavelength_fit = wavelength[mask]
        intensity_fit = intensity[mask]

    try:
        if fit_mode == "Détection automatique (plusieurs raies)":
            with st.expander("Paramètres de détection", expanded=True):
                profile_choice = st.radio(
                    "Profil utilisé pour chaque raie", ["Gaussienne", "Voigt"], horizontal=True
                )
                prominence_pct = st.slider(
                    "Sensibilité de détection (proéminence, % du continuum)",
                    min_value=0.5, max_value=20.0, value=3.0, step=0.5,
                )
                half_width = st.slider(
                    "Demi-largeur de fenêtre d'ajustement (points)",
                    min_value=5, max_value=300, value=40, step=5,
                )
                min_distance = st.slider(
                    "Distance minimale entre raies (points)",
                    min_value=1, max_value=300, value=20, step=1,
                )

            continuum = estimate_continuum(intensity_fit)
            prominence_abs = (prominence_pct / 100.0) * np.median(np.abs(continuum))
            peaks_idx, _ = detect_absorption_minima(
                wavelength_fit, intensity_fit, prominence_abs, min_distance
            )

            if len(peaks_idx) == 0:
                st.warning(
                    "Aucune raie d'absorption détectée avec ces paramètres. "
                    "Essaie de réduire la sensibilité (proéminence)."
                )
            else:
                st.success(f"{len(peaks_idx)} raie(s) d'absorption détectée(s).")
                results = []
                fit_curves = []

                for i, pk in enumerate(peaks_idx):
                    lo_i = max(pk - half_width, 0)
                    hi_i = min(pk + half_width, len(wavelength_fit) - 1)
                    w_loc = wavelength_fit[lo_i:hi_i + 1]
                    i_loc = intensity_fit[lo_i:hi_i + 1]

                    try:
                        if profile_choice == "Gaussienne":
                            p0 = guess_single(w_loc, i_loc)
                            popt, _ = curve_fit(gaussian, w_loc, i_loc, p0=p0, maxfev=10000)
                            amplitude, mean, sigma, offset = popt
                            fwhm = fwhm_from_sigma(sigma)
                            curve = gaussian(w_loc, *popt)
                            results.append({
                                "#": i + 1, "λ (Å)": round(mean, 3),
                                "Amplitude": round(amplitude, 4), "σ (Å)": round(sigma, 4),
                                "FWHM (Å)": round(fwhm, 4),
                            })
                        else:
                            p0 = guess_voigt(w_loc, i_loc)
                            popt, _ = curve_fit(voigt, w_loc, i_loc, p0=p0, maxfev=20000)
                            amplitude, mean, sigma, gamma, offset = popt
                            fwhm = fwhm_voigt(sigma, gamma)
                            curve = voigt(w_loc, *popt)
                            results.append({
                                "#": i + 1, "λ (Å)": round(mean, 3),
                                "Amplitude": round(amplitude, 4), "σ (Å)": round(sigma, 4),
                                "γ (Å)": round(gamma, 4), "FWHM (Å)": round(fwhm, 4),
                            })
                        fit_curves.append((w_loc, curve))
                    except RuntimeError:
                        st.warning(
                            f"Raie #{i + 1} (≈{wavelength_fit[pk]:.2f} Å) : "
                            "ajustement non convergé, ignorée."
                        )

                if results:
                    st.subheader("Raies détectées et ajustées")
                    st.dataframe(results, use_container_width=True)

                    fig, ax = plt.subplots()
                    ax.plot(wavelength, intensity, color="black", linewidth=0.8, label="Spectre")
                    ax.plot(wavelength_fit, continuum, "--", color="gray", alpha=0.6, label="Continuum estimé")
                    colors = plt.cm.tab10.colors
                    for idx, (w_loc, curve) in enumerate(fit_curves):
                        ax.plot(w_loc, curve, color=colors[idx % len(colors)], linewidth=1.5,
                                label=f"Raie {idx + 1}")
                    for r in results:
                        ax.axvline(r["λ (Å)"], color="red", linestyle=":", alpha=0.4)
                    ax.set_xlabel("Longueur d'onde (Å)" if wcs_ok else "Pixel")
                    ax.set_ylabel("Intensité")
                    ax.legend(fontsize=7, ncol=2)
                    st.pyplot(fig)

        elif fit_mode == "Gaussienne simple":
            p0 = guess_single(wavelength_fit, intensity_fit)
            popt, _ = curve_fit(gaussian, wavelength_fit, intensity_fit, p0=p0, maxfev=10000)
            amplitude, mean, sigma, offset = popt
            fwhm = fwhm_from_sigma(sigma)

            st.subheader("Résultats")
            st.write(f"Position (λ) : {mean:.3f} Å")
            st.write(f"Amplitude : {amplitude:.4f}")
            st.write(f"Dispersion (σ) : {sigma:.4f} Å")
            st.write(f"FWHM : {fwhm:.4f} Å")
            st.write(f"Continuum (offset) : {offset:.4f}")

            fit_curve = gaussian(wavelength, *popt)

        elif fit_mode == "Profil de Voigt":
            p0 = guess_voigt(wavelength_fit, intensity_fit)
            popt, _ = curve_fit(voigt, wavelength_fit, intensity_fit, p0=p0, maxfev=20000)
            amplitude, mean, sigma, gamma, offset = popt
            fwhm = fwhm_voigt(sigma, gamma)

            st.subheader("Résultats")
            st.write(f"Position (λ) : {mean:.3f} Å")
            st.write(f"Amplitude : {amplitude:.4f}")
            st.write(f"σ (Gauss) : {sigma:.4f} Å")
            st.write(f"γ (Lorentz) : {gamma:.4f} Å")
            st.write(f"FWHM (Voigt) : {fwhm:.4f} Å")
            st.write(f"Continuum (offset) : {offset:.4f}")

            fit_curve = voigt(wavelength, *popt)

        else:  # Gaussienne double
            p0 = guess_double(wavelength_fit, intensity_fit)
            popt, _ = curve_fit(double_gaussian, wavelength_fit, intensity_fit, p0=p0, maxfev=20000)
            amp1, mean1, sigma1, amp2, mean2, sigma2, offset = popt

            st.subheader("Résultats")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Composante 1**")
                st.write(f"Position (λ) : {mean1:.3f} Å")
                st.write(f"Amplitude : {amp1:.4f}")
                st.write(f"σ : {sigma1:.4f} Å")
                st.write(f"FWHM : {fwhm_from_sigma(sigma1):.4f} Å")
            with col2:
                st.markdown("**Composante 2**")
                st.write(f"Position (λ) : {mean2:.3f} Å")
                st.write(f"Amplitude : {amp2:.4f}")
                st.write(f"σ : {sigma2:.4f} Å")
                st.write(f"FWHM : {fwhm_from_sigma(sigma2):.4f} Å")
            st.write(f"Continuum (offset) : {offset:.4f}")
            st.write(f"Séparation Δλ entre composantes : {abs(mean2 - mean1):.3f} Å")

            fit_curve = double_gaussian(wavelength, *popt)

        # -------------------------------------------------------------
        # Tracé (uniquement pour les modes à profil unique)
        # -------------------------------------------------------------
        if fit_mode != "Détection automatique (plusieurs raies)":
            fig, ax = plt.subplots()
            ax.plot(wavelength, intensity, label="Spectre", color="black", linewidth=0.8)
            ax.plot(wavelength, fit_curve, label="Ajustement", color="red")
            if fit_mode == "Gaussienne double":
                ax.plot(
                    wavelength,
                    gaussian(wavelength, amp1, mean1, sigma1, offset),
                    "--", color="tab:blue", alpha=0.7, label="Composante 1",
                )
                ax.plot(
                    wavelength,
                    gaussian(wavelength, amp2, mean2, sigma2, offset),
                    "--", color="tab:green", alpha=0.7, label="Composante 2",
                )
            ax.axvspan(wavelength_fit.min(), wavelength_fit.max(), color="orange", alpha=0.08, label="Plage ajustée")
            ax.set_xlabel("Longueur d'onde (Å)" if wcs_ok else "Pixel")
            ax.set_ylabel("Intensité")
            ax.legend()
            st.pyplot(fig)

    except RuntimeError as e:
        st.error(
            "L'ajustement n'a pas convergé. Essaie de restreindre la plage de "
            f"longueur d'onde autour de la raie visée.\n\nDétail : {e}"
        )

