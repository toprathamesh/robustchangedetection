"""
Advanced Spectral Indices Module
===============================
Comprehensive calculation of spectral indices for satellite imagery analysis.
Supports multiple satellite platforms and provides robust index calculations
with quality assessment and uncertainty quantification.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum
import warnings

logger = logging.getLogger(__name__)


class SatellitePlatform(Enum):
    """Supported satellite platforms"""
    SENTINEL2 = "sentinel2"
    LANDSAT8 = "landsat8"
    LANDSAT9 = "landsat9"
    MODIS = "modis"
    GENERIC = "generic"


@dataclass
class BandConfiguration:
    """Band configuration for different satellite platforms"""
    platform: SatellitePlatform
    bands: Dict[str, int]  # Band name to index mapping
    scale_factor: float = 1.0
    offset: float = 0.0
    
    @classmethod
    def get_sentinel2_config(cls):
        return cls(
            platform=SatellitePlatform.SENTINEL2,
            bands={
                'blue': 0,      # B2 - 490nm
                'green': 1,     # B3 - 560nm  
                'red': 2,       # B4 - 665nm
                'red_edge1': 3, # B5 - 705nm
                'red_edge2': 4, # B6 - 740nm
                'red_edge3': 5, # B7 - 783nm
                'nir': 6,       # B8 - 842nm
                'red_edge4': 7, # B8A - 865nm
                'swir1': 8,     # B11 - 1610nm
                'swir2': 9      # B12 - 2190nm
            },
            scale_factor=0.0001  # Convert to reflectance
        )
    
    @classmethod
    def get_landsat8_config(cls):
        return cls(
            platform=SatellitePlatform.LANDSAT8,
            bands={
                'blue': 0,      # B2 - 482nm
                'green': 1,     # B3 - 562nm
                'red': 2,       # B4 - 655nm
                'nir': 3,       # B5 - 865nm
                'swir1': 4,     # B6 - 1610nm
                'swir2': 5      # B7 - 2200nm
            },
            scale_factor=0.0001
        )


class SpectralIndicesCalculator:
    """
    Advanced spectral indices calculator with support for multiple platforms
    and comprehensive index library
    """
    
    def __init__(self, platform: SatellitePlatform = SatellitePlatform.GENERIC):
        self.platform = platform
        self.band_config = self._get_band_configuration(platform)
        
        # Index metadata for quality assessment
        self.index_metadata = {
            'ndvi': {'range': (-1, 1), 'optimal_range': (0.1, 0.9), 'noise_threshold': 0.05},
            'ndwi': {'range': (-1, 1), 'optimal_range': (-0.3, 0.8), 'noise_threshold': 0.05},
            'ndbi': {'range': (-1, 1), 'optimal_range': (-0.5, 0.5), 'noise_threshold': 0.05},
            'evi': {'range': (-1, 1), 'optimal_range': (0.0, 1.0), 'noise_threshold': 0.05},
            'savi': {'range': (-1.5, 1.5), 'optimal_range': (0.0, 1.2), 'noise_threshold': 0.05},
            'bsi': {'range': (-1, 1), 'optimal_range': (-0.5, 0.5), 'noise_threshold': 0.05}
        }
    
    def _get_band_configuration(self, platform: SatellitePlatform) -> BandConfiguration:
        """Get band configuration for platform"""
        if platform == SatellitePlatform.SENTINEL2:
            return BandConfiguration.get_sentinel2_config()
        elif platform in [SatellitePlatform.LANDSAT8, SatellitePlatform.LANDSAT9]:
            return BandConfiguration.get_landsat8_config()
        else:
            # Generic configuration - assumes RGB + NIR + SWIR
            return BandConfiguration(
                platform=platform,
                bands={'blue': 0, 'green': 1, 'red': 2, 'nir': 3, 'swir1': 4, 'swir2': 5}
            )
    
    def calculate_all_indices(self, 
                            image: np.ndarray, 
                            mask: Optional[np.ndarray] = None,
                            return_quality_info: bool = True) -> Dict:
        """
        Calculate all available spectral indices for the image
        
        Args:
            image: Multi-band satellite image (H, W, C)
            mask: Optional mask to exclude pixels (clouds, shadows, etc.)
            return_quality_info: Include quality assessment information
            
        Returns:
            Dictionary containing all calculated indices and quality info
        """
        if len(image.shape) != 3:
            raise ValueError("Image must be 3D array (height, width, channels)")
        
        # Apply scale factor if needed
        if self.band_config.scale_factor != 1.0:
            image = image.astype(np.float32) * self.band_config.scale_factor
        
        # Add offset if needed
        if self.band_config.offset != 0.0:
            image = image + self.band_config.offset
        
        results = {}
        quality_info = {}
        
        # Vegetation indices
        vegetation_indices = self._calculate_vegetation_indices(image, mask)
        results.update(vegetation_indices)
        
        # Water indices
        water_indices = self._calculate_water_indices(image, mask)
        results.update(water_indices)
        
        # Urban/built-up indices
        urban_indices = self._calculate_urban_indices(image, mask)
        results.update(urban_indices)
        
        # Soil indices  
        soil_indices = self._calculate_soil_indices(image, mask)
        results.update(soil_indices)
        
        # Specialized indices
        specialized_indices = self._calculate_specialized_indices(image, mask)
        results.update(specialized_indices)
        
        # Calculate quality metrics if requested
        if return_quality_info:
            for index_name, index_data in results.items():
                quality_info[index_name] = self._assess_index_quality(
                    index_data, index_name, mask
                )
        
        return {
            'indices': results,
            'quality_info': quality_info if return_quality_info else {},
            'metadata': {
                'platform': self.platform.value,
                'image_shape': image.shape,
                'masked_pixels': np.sum(mask) if mask is not None else 0
            }
        }
    
    def _calculate_vegetation_indices(self, 
                                    image: np.ndarray, 
                                    mask: Optional[np.ndarray] = None) -> Dict:
        """Calculate vegetation-related spectral indices"""
        indices = {}
        bands = self.band_config.bands
        
        # NDVI - Normalized Difference Vegetation Index
        if 'red' in bands and 'nir' in bands:
            red = self._get_band(image, 'red')
            nir = self._get_band(image, 'nir')
            indices['ndvi'] = self._safe_divide(nir - red, nir + red)
        
        # EVI - Enhanced Vegetation Index
        if all(band in bands for band in ['red', 'nir', 'blue']):
            red = self._get_band(image, 'red')
            nir = self._get_band(image, 'nir')
            blue = self._get_band(image, 'blue')
            
            # EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
            numerator = 2.5 * (nir - red)
            denominator = nir + 6.0 * red - 7.5 * blue + 1.0
            indices['evi'] = self._safe_divide(numerator, denominator)
        
        # SAVI - Soil Adjusted Vegetation Index
        if 'red' in bands and 'nir' in bands:
            red = self._get_band(image, 'red')
            nir = self._get_band(image, 'nir')
            L = 0.5  # Soil brightness correction factor
            
            numerator = (nir - red) * (1.0 + L)
            denominator = nir + red + L
            indices['savi'] = self._safe_divide(numerator, denominator)
        
        # MSAVI2 - Modified Soil Adjusted Vegetation Index 2
        if 'red' in bands and 'nir' in bands:
            red = self._get_band(image, 'red')
            nir = self._get_band(image, 'nir')
            
            # MSAVI2 = (2*NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR - Red))) / 2
            term1 = 2.0 * nir + 1.0
            term2 = np.sqrt(np.maximum(0, term1**2 - 8.0 * (nir - red)))
            indices['msavi2'] = (term1 - term2) / 2.0
        
        # NDRE - Normalized Difference Red Edge (Sentinel-2 specific)
        if 'red_edge1' in bands and 'nir' in bands:
            red_edge = self._get_band(image, 'red_edge1')
            nir = self._get_band(image, 'nir')
            indices['ndre'] = self._safe_divide(nir - red_edge, nir + red_edge)
        
        # CIG - Chlorophyll Index Green
        if 'nir' in bands and 'green' in bands:
            nir = self._get_band(image, 'nir')
            green = self._get_band(image, 'green')
            indices['cig'] = self._safe_divide(nir, green) - 1.0
        
        # Apply mask if provided
        if mask is not None:
            for key in indices:
                indices[key] = np.where(mask, np.nan, indices[key])
        
        return indices
    
    def _calculate_water_indices(self, 
                               image: np.ndarray, 
                               mask: Optional[np.ndarray] = None) -> Dict:
        """Calculate water-related spectral indices"""
        indices = {}
        bands = self.band_config.bands
        
        # NDWI - Normalized Difference Water Index
        if 'green' in bands and 'nir' in bands:
            green = self._get_band(image, 'green')
            nir = self._get_band(image, 'nir')
            indices['ndwi'] = self._safe_divide(green - nir, green + nir)
        
        # MNDWI - Modified Normalized Difference Water Index
        if 'green' in bands and 'swir1' in bands:
            green = self._get_band(image, 'green')
            swir1 = self._get_band(image, 'swir1')
            indices['mndwi'] = self._safe_divide(green - swir1, green + swir1)
        
        # AWEInsh - Automated Water Extraction Index (no shadow)
        if all(band in bands for band in ['blue', 'green', 'nir', 'swir1', 'swir2']):
            blue = self._get_band(image, 'blue')
            green = self._get_band(image, 'green')
            nir = self._get_band(image, 'nir')
            swir1 = self._get_band(image, 'swir1')
            swir2 = self._get_band(image, 'swir2')
            
            indices['awei_nsh'] = 4.0 * (green - swir1) - (0.25 * nir + 2.75 * swir2)
        
        # AWEIsh - Automated Water Extraction Index (with shadow)
        if all(band in bands for band in ['blue', 'green', 'nir', 'swir1', 'swir2']):
            blue = self._get_band(image, 'blue')
            green = self._get_band(image, 'green')
            nir = self._get_band(image, 'nir')
            swir1 = self._get_band(image, 'swir1')
            swir2 = self._get_band(image, 'swir2')
            
            indices['awei_sh'] = blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2
        
        # Apply mask if provided
        if mask is not None:
            for key in indices:
                indices[key] = np.where(mask, np.nan, indices[key])
        
        return indices
    
    def _calculate_urban_indices(self, 
                               image: np.ndarray, 
                               mask: Optional[np.ndarray] = None) -> Dict:
        """Calculate urban/built-up area indices"""
        indices = {}
        bands = self.band_config.bands
        
        # NDBI - Normalized Difference Built-up Index
        if 'swir1' in bands and 'nir' in bands:
            swir1 = self._get_band(image, 'swir1')
            nir = self._get_band(image, 'nir')
            indices['ndbi'] = self._safe_divide(swir1 - nir, swir1 + nir)
        
        # EBBI - Enhanced Built-Up and Bareness Index
        if all(band in bands for band in ['red', 'nir', 'swir1']):
            red = self._get_band(image, 'red')
            nir = self._get_band(image, 'nir')
            swir1 = self._get_band(image, 'swir1')
            
            numerator = swir1 - nir
            denominator = 10.0 * np.sqrt(swir1 + red)
            indices['ebbi'] = self._safe_divide(numerator, denominator)
        
        # BUI - Built-up Index
        if all(band in bands for band in ['red', 'nir', 'swir1']):
            red = self._get_band(image, 'red')
            nir = self._get_band(image, 'nir')
            swir1 = self._get_band(image, 'swir1')
            
            numerator = (red - nir) + (swir1 - nir)
            denominator = (red + nir) + (swir1 + nir)
            indices['bui'] = self._safe_divide(numerator, denominator)
        
        # BAEI - Built-up Area Extraction Index
        if all(band in bands for band in ['red', 'green', 'nir']):
            red = self._get_band(image, 'red')
            green = self._get_band(image, 'green')
            nir = self._get_band(image, 'nir')
            
            numerator = red + 0.3
            denominator = green + nir
            indices['baei'] = self._safe_divide(numerator, denominator)
        
        # Apply mask if provided
        if mask is not None:
            for key in indices:
                indices[key] = np.where(mask, np.nan, indices[key])
        
        return indices
    
    def _calculate_soil_indices(self, 
                              image: np.ndarray, 
                              mask: Optional[np.ndarray] = None) -> Dict:
        """Calculate soil-related spectral indices"""
        indices = {}
        bands = self.band_config.bands
        
        # BSI - Bare Soil Index
        if all(band in bands for band in ['red', 'nir', 'blue', 'swir1']):
            red = self._get_band(image, 'red')
            nir = self._get_band(image, 'nir')
            blue = self._get_band(image, 'blue')
            swir1 = self._get_band(image, 'swir1')
            
            numerator = (swir1 + red) - (nir + blue)
            denominator = (swir1 + red) + (nir + blue)
            indices['bsi'] = self._safe_divide(numerator, denominator)
        
        # SI - Salinity Index
        if 'blue' in bands and 'red' in bands:
            blue = self._get_band(image, 'blue')
            red = self._get_band(image, 'red')
            indices['si'] = np.sqrt(blue * red)
        
        # BI - Brightness Index
        if all(band in bands for band in ['red', 'green', 'blue']):
            red = self._get_band(image, 'red')
            green = self._get_band(image, 'green')
            blue = self._get_band(image, 'blue')
            indices['bi'] = np.sqrt((red**2 + green**2 + blue**2) / 3.0)
        
        # Apply mask if provided
        if mask is not None:
            for key in indices:
                indices[key] = np.where(mask, np.nan, indices[key])
        
        return indices
    
    def _calculate_specialized_indices(self, 
                                     image: np.ndarray, 
                                     mask: Optional[np.ndarray] = None) -> Dict:
        """Calculate specialized indices for specific applications"""
        indices = {}
        bands = self.band_config.bands
        
        # NBR - Normalized Burn Ratio (fire detection)
        if 'nir' in bands and 'swir2' in bands:
            nir = self._get_band(image, 'nir')
            swir2 = self._get_band(image, 'swir2')
            indices['nbr'] = self._safe_divide(nir - swir2, nir + swir2)
        
        # MIRBI - Mid-Infrared Burn Index
        if 'swir1' in bands and 'swir2' in bands:
            swir1 = self._get_band(image, 'swir1')
            swir2 = self._get_band(image, 'swir2')
            indices['mirbi'] = 10.0 * swir2 - 9.8 * swir1 + 2.0
        
        # GEMI - Global Environment Monitoring Index
        if 'red' in bands and 'nir' in bands:
            red = self._get_band(image, 'red')
            nir = self._get_band(image, 'nir')
            
            eta = (2.0 * (nir**2 - red**2) + 1.5 * nir + 0.5 * red) / (nir + red + 0.5)
            indices['gemi'] = eta * (1.0 - 0.25 * eta) - ((red - 0.125) / (1.0 - red))
        
        # Apply mask if provided
        if mask is not None:
            for key in indices:
                indices[key] = np.where(mask, np.nan, indices[key])
        
        return indices
    
    def _get_band(self, image: np.ndarray, band_name: str) -> np.ndarray:
        """Get specific band from image"""
        if band_name not in self.band_config.bands:
            raise ValueError(f"Band {band_name} not available for platform {self.platform}")
        
        band_idx = self.band_config.bands[band_name]
        if band_idx >= image.shape[2]:
            raise ValueError(f"Band index {band_idx} exceeds image channels {image.shape[2]}")
        
        return image[:, :, band_idx].astype(np.float32)
    
    def _safe_divide(self, numerator: np.ndarray, denominator: np.ndarray, 
                    fill_value: float = 0.0) -> np.ndarray:
        """Safe division with handling of division by zero"""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            result = np.divide(numerator, denominator,
                             out=np.full_like(numerator, fill_value),
                             where=(denominator != 0))
        
        # Set infinite and very large values to fill_value
        result = np.where(np.isfinite(result), result, fill_value)
        result = np.clip(result, -10.0, 10.0)  # Reasonable bounds
        
        return result
    
    def _assess_index_quality(self, 
                            index_data: np.ndarray, 
                            index_name: str,
                            mask: Optional[np.ndarray] = None) -> Dict:
        """Assess quality of calculated spectral index"""
        if index_name not in self.index_metadata:
            return {'quality': 'unknown', 'message': 'No quality metadata available'}
        
        metadata = self.index_metadata[index_name]
        valid_data = index_data[np.isfinite(index_data)]
        
        if mask is not None:
            valid_data = index_data[~mask & np.isfinite(index_data)]
        
        if len(valid_data) == 0:
            return {'quality': 'poor', 'message': 'No valid data points'}
        
        # Check value range
        min_val, max_val = metadata['range']
        out_of_range = np.sum((valid_data < min_val) | (valid_data > max_val))
        out_of_range_pct = out_of_range / len(valid_data) * 100
        
        # Check optimal range
        opt_min, opt_max = metadata['optimal_range']
        in_optimal = np.sum((valid_data >= opt_min) & (valid_data <= opt_max))
        optimal_pct = in_optimal / len(valid_data) * 100
        
        # Noise assessment
        noise_level = np.std(valid_data)
        noise_threshold = metadata['noise_threshold']
        
        # Overall quality assessment
        if out_of_range_pct > 10:
            quality = 'poor'
            message = f'High out-of-range values: {out_of_range_pct:.1f}%'
        elif optimal_pct < 50:
            quality = 'fair'
            message = f'Low optimal range coverage: {optimal_pct:.1f}%'
        elif noise_level > noise_threshold * 3:
            quality = 'fair'
            message = f'High noise level: {noise_level:.3f}'
        else:
            quality = 'good'
            message = 'Index quality is good'
        
        return {
            'quality': quality,
            'message': message,
            'statistics': {
                'mean': np.mean(valid_data),
                'std': noise_level,
                'min': np.min(valid_data),
                'max': np.max(valid_data),
                'valid_pixels': len(valid_data),
                'out_of_range_pct': out_of_range_pct,
                'optimal_range_pct': optimal_pct
            }
        }
    
    def calculate_change_vector_analysis(self, 
                                       before_indices: Dict[str, np.ndarray],
                                       after_indices: Dict[str, np.ndarray],
                                       indices_to_use: Optional[List[str]] = None) -> Dict:
        """
        Perform Change Vector Analysis (CVA) using multiple spectral indices
        
        Args:
            before_indices: Spectral indices from before image
            after_indices: Spectral indices from after image
            indices_to_use: Specific indices to use (if None, uses all common indices)
            
        Returns:
            Change vector analysis results
        """
        # Find common indices
        common_indices = set(before_indices.keys()) & set(after_indices.keys())
        
        if indices_to_use:
            common_indices = common_indices & set(indices_to_use)
        
        if not common_indices:
            raise ValueError("No common indices found between before and after")
        
        common_indices = list(common_indices)
        
        # Stack indices into feature vectors
        before_stack = []
        after_stack = []
        
        for idx_name in common_indices:
            before_stack.append(before_indices[idx_name])
            after_stack.append(after_indices[idx_name])
        
        before_features = np.stack(before_stack, axis=-1)
        after_features = np.stack(after_stack, axis=-1)
        
        # Calculate change vectors
        change_vectors = after_features - before_features
        
        # Calculate change magnitude (Euclidean distance)
        change_magnitude = np.sqrt(np.sum(change_vectors**2, axis=-1))
        
        # Calculate change direction (angle)
        # Use first two principal components for visualization
        from sklearn.decomposition import PCA
        
        # Flatten for PCA
        original_shape = change_vectors.shape[:2]
        flat_vectors = change_vectors.reshape(-1, change_vectors.shape[-1])
        
        # Remove invalid pixels
        valid_mask = np.all(np.isfinite(flat_vectors), axis=1)
        valid_vectors = flat_vectors[valid_mask]
        
        if len(valid_vectors) > 10:  # Need enough samples for PCA
            pca = PCA(n_components=2)
            pca_vectors = pca.fit_transform(valid_vectors)
            
            # Calculate angles in PCA space
            angles = np.arctan2(pca_vectors[:, 1], pca_vectors[:, 0])
            
            # Map back to original shape
            full_angles = np.full(len(flat_vectors), np.nan)
            full_angles[valid_mask] = angles
            change_direction = full_angles.reshape(original_shape)
            
            explained_variance = pca.explained_variance_ratio_
        else:
            change_direction = np.full(original_shape, np.nan)
            explained_variance = [0, 0]
        
        # Statistical analysis
        valid_magnitude = change_magnitude[np.isfinite(change_magnitude)]
        
        if len(valid_magnitude) > 0:
            magnitude_stats = {
                'mean': np.mean(valid_magnitude),
                'std': np.std(valid_magnitude),
                'median': np.median(valid_magnitude),
                'percentile_95': np.percentile(valid_magnitude, 95),
                'percentile_99': np.percentile(valid_magnitude, 99)
            }
        else:
            magnitude_stats = {}
        
        return {
            'change_magnitude': change_magnitude,
            'change_direction': change_direction,
            'change_vectors': change_vectors,
            'indices_used': common_indices,
            'magnitude_statistics': magnitude_stats,
            'pca_explained_variance': explained_variance,
            'total_pixels': np.prod(original_shape),
            'valid_pixels': len(valid_magnitude)
        }
    
    def detect_change_hotspots(self, 
                             change_magnitude: np.ndarray,
                             percentile_threshold: float = 95.0,
                             min_cluster_size: int = 10) -> Dict:
        """
        Detect significant change hotspots based on change magnitude
        
        Args:
            change_magnitude: Change magnitude array from CVA
            percentile_threshold: Percentile threshold for hotspot detection
            min_cluster_size: Minimum size for change clusters
            
        Returns:
            Hotspot detection results
        """
        from scipy import ndimage
        from skimage.measure import label, regionprops
        
        # Calculate threshold
        valid_magnitudes = change_magnitude[np.isfinite(change_magnitude)]
        if len(valid_magnitudes) == 0:
            return {'hotspots': [], 'hotspot_mask': np.zeros_like(change_magnitude, dtype=bool)}
        
        threshold = np.percentile(valid_magnitudes, percentile_threshold)
        
        # Create binary mask
        hotspot_mask = change_magnitude > threshold
        
        # Remove small isolated pixels
        hotspot_mask = ndimage.binary_opening(hotspot_mask, structure=np.ones((3, 3)))
        
        # Label connected components
        labeled_hotspots = label(hotspot_mask)
        regions = regionprops(labeled_hotspots)
        
        # Filter by size
        significant_hotspots = []
        filtered_mask = np.zeros_like(hotspot_mask)
        
        for region in regions:
            if region.area >= min_cluster_size:
                significant_hotspots.append({
                    'area': region.area,
                    'centroid': region.centroid,
                    'bbox': region.bbox,
                    'mean_magnitude': np.mean(change_magnitude[labeled_hotspots == region.label]),
                    'max_magnitude': np.max(change_magnitude[labeled_hotspots == region.label])
                })
                
                # Add to filtered mask
                filtered_mask[labeled_hotspots == region.label] = True
        
        return {
            'hotspots': significant_hotspots,
            'hotspot_mask': filtered_mask,
            'threshold_used': threshold,
            'total_hotspots': len(significant_hotspots),
            'total_hotspot_area': sum(h['area'] for h in significant_hotspots)
        }


class MultiTemporalIndicesAnalyzer:
    """
    Analyzer for multi-temporal spectral indices analysis
    Performs time series analysis of spectral indices for change detection
    """
    
    def __init__(self):
        self.time_series_data = {}
        self.calculator = SpectralIndicesCalculator()
    
    def add_image_data(self, 
                      date: str, 
                      image: np.ndarray, 
                      platform: SatellitePlatform = SatellitePlatform.GENERIC,
                      mask: Optional[np.ndarray] = None):
        """Add image data for specific date"""
        self.calculator.platform = platform
        self.calculator.band_config = self.calculator._get_band_configuration(platform)
        
        results = self.calculator.calculate_all_indices(image, mask, return_quality_info=False)
        self.time_series_data[date] = results['indices']
    
    def analyze_temporal_trends(self, 
                              index_name: str,
                              roi_mask: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze temporal trends for specific spectral index
        
        Args:
            index_name: Name of spectral index to analyze
            roi_mask: Region of interest mask
            
        Returns:
            Temporal trend analysis results
        """
        if not self.time_series_data:
            return {'error': 'No time series data available'}
        
        # Extract time series
        dates = sorted(self.time_series_data.keys())
        values = []
        
        for date in dates:
            if index_name in self.time_series_data[date]:
                index_data = self.time_series_data[date][index_name]
                
                if roi_mask is not None:
                    roi_data = index_data[roi_mask]
                    values.append(np.nanmean(roi_data))
                else:
                    values.append(np.nanmean(index_data))
            else:
                values.append(np.nan)
        
        # Perform trend analysis
        from scipy import stats
        
        valid_indices = ~np.isnan(values)
        if np.sum(valid_indices) < 3:
            return {'error': 'Insufficient valid data points'}
        
        valid_dates = np.array(dates)[valid_indices]
        valid_values = np.array(values)[valid_indices]
        
        # Convert dates to timestamps for regression
        timestamps = np.array([pd.to_datetime(date).timestamp() for date in valid_dates])
        
        # Linear trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, valid_values)
        
        # Seasonal decomposition (if enough data)
        seasonal_info = {}
        if len(valid_values) >= 12:
            try:
                import pandas as pd
                from statsmodels.tsa.seasonal import seasonal_decompose
                
                df = pd.DataFrame({
                    'date': pd.to_datetime(valid_dates),
                    'value': valid_values
                })
                df = df.set_index('date').resample('M').mean()
                
                if len(df) >= 24:  # Need at least 2 years for seasonal decomposition
                    decomposition = seasonal_decompose(df['value'], model='additive', period=12)
                    seasonal_info = {
                        'seasonal_strength': np.std(decomposition.seasonal) / np.std(valid_values),
                        'trend_strength': np.std(decomposition.trend.dropna()) / np.std(valid_values),
                        'residual_strength': np.std(decomposition.resid.dropna()) / np.std(valid_values)
                    }
            except:
                pass
        
        return {
            'dates': dates,
            'values': values,
            'linear_trend': {
                'slope': slope * 365.25 * 24 * 3600,  # Convert to per year
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'is_significant': p_value < 0.05
            },
            'seasonal_analysis': seasonal_info,
            'statistics': {
                'mean': np.nanmean(values),
                'std': np.nanstd(values),
                'min': np.nanmin(values),
                'max': np.nanmax(values),
                'valid_points': np.sum(valid_indices)
            }
        } 