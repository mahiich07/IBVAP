"""
Low-Light Image Enhancement Module for Night-Time Border Surveillance
Implements Multi-Scale Retinex with Color Restoration (MSRCR), Zero-DCE Curve Estimation, and Adaptive CLAHE.
"""

import cv2
import numpy as np
from typing import Optional, List


class MultiScaleRetinex:
    """
    Multi-Scale Retinex with Color Restoration (MSRCR).
    Separates illumination and reflectance components using multi-scale Gaussian surrounds.
    Dramatically boosts contrast in dark surveillance footage while preserving color balance.
    """

    def __init__(
        self,
        sigmas: List[float] = [15.0, 80.0, 250.0],
        weights: Optional[List[float]] = None,
        alpha: float = 125.0,
        beta: float = 46.0,
        gain: float = 1.0,
        offset: float = 0.0
    ):
        self.sigmas = sigmas
        self.weights = weights or [1.0 / len(sigmas)] * len(sigmas)
        self.alpha = alpha
        self.beta = beta
        self.gain = gain
        self.offset = offset

    def enhance(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Applies MSRCR to the input BGR image.
        """
        img = img_bgr.astype(np.float64) + 1.0  # Avoid log(0)
        img_sum = np.sum(img, axis=2, keepdims=True)

        # 1. Multi-Scale Retinex filtering across scales
        retinex = np.zeros_like(img)
        for sigma, weight in zip(self.sigmas, self.weights):
            blur = cv2.GaussianBlur(img, (0, 0), sigma) + 1.0
            retinex += weight * (np.log(img) - np.log(blur))

        # 2. Color Restoration Factor
        color_restoration = self.beta * (np.log(self.alpha * img) - np.log(img_sum))
        msrcr = self.gain * (retinex * color_restoration) + self.offset

        # 3. Dynamic range compression & normalization (2% - 98% percentile clipping)
        output = np.zeros_like(msrcr)
        for c in range(3):
            low = np.percentile(msrcr[:, :, c], 2)
            high = np.percentile(msrcr[:, :, c], 98)
            channel = (msrcr[:, :, c] - low) / (high - low + 1e-6) * 255.0
            output[:, :, c] = np.clip(channel, 0, 255)

        return output.astype(np.uint8)


class ZeroDCEEnhancer:
    """
    Zero-Reference Deep Curve Estimation (Zero-DCE) formulation.
    Enhances dark images via iterative higher-order curve adjustment:
        LE_n(x) = LE_{n-1}(x) + A_n(x) * LE_{n-1}(x) * (1 - LE_{n-1}(x))
    This non-linear curve increases dynamic range without over-amplifying noise or saturation.
    """

    def __init__(self, iterations: int = 8):
        self.iterations = iterations

    def enhance(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Performs iterative curve estimation on normalized image tensor.
        """
        # Normalize to [0, 1]
        x = img_bgr.astype(np.float32) / 255.0
        
        # Estimate parameter map A based on inverse illumination map
        # Darker regions get higher amplification curve slope
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        illum_map = cv2.GaussianBlur(gray, (15, 15), 5.0)
        param_map = np.clip((1.0 - illum_map) * 0.45, 0.05, 0.60)[:, :, np.newaxis]

        # Iterative curve expansion
        le = x.copy()
        for _ in range(self.iterations):
            le = le + param_map * le * (1.0 - le)

        # Scale back to [0, 255]
        enhanced = np.clip(le * 255.0, 0, 255).astype(np.uint8)
        return enhanced


class LowLightEnhancer:
    """
    Unified Low-Light Enhancer orchestrating Retinex, Zero-DCE, and Adaptive CLAHE.
    Features automatic low-lux detection to avoid unnecessary overhead during daylight.
    """

    def __init__(
        self,
        method: str = "auto",  # 'auto', 'msrcr', 'zero_dce', 'clahe', 'none'
        dark_threshold_lum: float = 65.0
    ):
        self.method = method
        self.dark_threshold_lum = dark_threshold_lum
        self.retinex = MultiScaleRetinex()
        self.zero_dce = ZeroDCEEnhancer()
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    def measure_luminance(self, img_bgr: np.ndarray) -> float:
        """Calculates average perceptual luminance (Y channel in YUV)."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def enhance_clahe(self, img_bgr: np.ndarray) -> np.ndarray:
        """High-speed LAB CLAHE for edge devices (<2ms latency)."""
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        enhanced_lab = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def process(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Selects optimal enhancement based on configuration and ambient illumination.
        """
        if self.method == "none":
            return img_bgr

        lum = self.measure_luminance(img_bgr)
        
        # In auto mode, only enhance if scene is underexposed/night
        if self.method == "auto":
            if lum >= self.dark_threshold_lum:
                return img_bgr  # Daylight or well-lit scene; pass-through
            # For low-light, Zero-DCE provides natural dynamic range recovery
            return self.zero_dce.enhance(img_bgr)

        elif self.method == "zero_dce":
            return self.zero_dce.enhance(img_bgr)

        elif self.method == "msrcr":
            return self.retinex.enhance(img_bgr)

        elif self.method == "clahe":
            return self.enhance_clahe(img_bgr)

        return img_bgr
