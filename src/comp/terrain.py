"""SPEC-76/77: вода по Fmask, освещённость склона и C-коррекция (Teillet 1982) по DEM и углам солнца HLS."""
import numpy as np


def water(fmask):
    """Бит 5 Fmask HLS — вода; 255 — заполнение, не вода."""
    return (fmask != 255) & (((fmask >> 5) & 1) == 1)


def slope_deg(dem, res=30.0):
    gy, gx = np.gradient(dem, res); return np.degrees(np.arctan(np.hypot(gx, gy)))


def cos_incidence(dem, sza_deg, saa_deg, res=30.0):
    """cos угла между нормалью к поверхности и направлением на солнце. Строки идут на юг, поэтому d/dсевер = −d/dстрока."""
    dz_drow, dz_dx = np.gradient(dem, res); dz_dn = -dz_drow
    n = np.stack([-dz_dx, -dz_dn, np.ones_like(dem)]); n /= np.linalg.norm(n, axis=0)
    z, a = np.radians(sza_deg), np.radians(saa_deg)
    sun = np.stack([np.sin(z) * np.sin(a), np.sin(z) * np.cos(a), np.cos(z)])
    return (n * sun).sum(0)


def c_correct(X, cos_i, sza_deg, ok, slope, min_px=1000):
    """X (C,H,W) отражения. Для каждой полосы: ρ = a + b·cos i по пикселям ok & уклон > 5°, c = a/b,
    ρ' = ρ·(cos SZA + c)/(cos i + c), множитель ∈ [0.5, 2]. Полоса без поправки, если b ≤ 0 или пикселей мало."""
    out = X.copy(); fit = ok & (slope > 5) & np.isfinite(cos_i); cz = np.cos(np.radians(sza_deg)); used = []
    if fit.sum() < min_px: return out, used
    for k in range(X.shape[0]):
        b, a = np.polyfit(cos_i[fit], X[k][fit], 1)
        if b <= 0: continue
        c = a / b; f = np.clip((cz + c) / np.maximum(cos_i + c, 1e-3), 0.5, 2.0); f = np.where(np.isfinite(f), f, 1.0)
        out[k] = np.where(ok, X[k] * f, X[k]); used.append(k)
    return out, used
