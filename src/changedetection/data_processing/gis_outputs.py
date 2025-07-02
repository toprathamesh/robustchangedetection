"""
Enhanced GIS Output Module
=========================
Comprehensive GIS data export functionality supporting multiple formats
with complete metadata and optimization for web usage.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import tempfile
import zipfile
from dataclasses import dataclass, field

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.profiles import default_gtiff_profile
import geopandas as gpd
from shapely.geometry import Polygon, Point, mapping, shape
import fiona
from fiona.crs import from_epsg
import simplekml

try:
    from osgeo import gdal, ogr, osr
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False
    gdal = None
    ogr = None
    osr = None

logger = logging.getLogger(__name__)


@dataclass
class ExportMetadata:
    """Metadata for GIS exports"""
    title: str
    description: str
    author: str = "Change Detection System"
    creation_date: datetime = field(default_factory=datetime.now)
    spatial_extent: Optional[Tuple[float, float, float, float]] = None
    temporal_extent: Optional[Tuple[datetime, datetime]] = None
    coordinate_system: str = "EPSG:4326"
    data_source: str = "Satellite Imagery Analysis"
    processing_level: str = "Processed"
    keywords: List[str] = field(default_factory=list)
    contact_info: Dict[str, str] = field(default_factory=dict)
    license: str = "All rights reserved"
    quality_info: Dict[str, float] = field(default_factory=dict)


class CloudOptimizedGeoTIFFExporter:
    """
    Export raster data as Cloud Optimized GeoTIFF (COG)
    """
    
    def __init__(self):
        self.cog_profile = {
            'driver': 'GTiff',
            'interleave': 'pixel',
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512,
            'compress': 'lzw',
            'predictor': 2,
            'BIGTIFF': 'IF_SAFER'
        }
    
    def export_change_detection_results(self, 
                                      change_map: np.ndarray,
                                      bounds: Tuple[float, float, float, float],
                                      output_path: str,
                                      metadata: ExportMetadata,
                                      confidence_map: Optional[np.ndarray] = None,
                                      change_types: Optional[np.ndarray] = None) -> bool:
        """
        Export change detection results as COG
        
        Args:
            change_map: Binary change detection map
            bounds: Geographic bounds (minx, miny, maxx, maxy)
            output_path: Output file path
            metadata: Export metadata
            confidence_map: Optional confidence map
            change_types: Optional change type classification map
            
        Returns:
            Success status
        """
        try:
            height, width = change_map.shape
            
            # Calculate transform
            transform = from_bounds(*bounds, width, height)
            
            # Determine number of bands
            bands_data = [change_map.astype(np.uint8)]
            band_descriptions = ['Change Detection']
            
            if confidence_map is not None:
                bands_data.append((confidence_map * 255).astype(np.uint8))
                band_descriptions.append('Confidence')
            
            if change_types is not None:
                bands_data.append(change_types.astype(np.uint8))
                band_descriptions.append('Change Types')
            
            # Create COG profile
            profile = self.cog_profile.copy()
            profile.update({
                'height': height,
                'width': width,
                'count': len(bands_data),
                'dtype': 'uint8',
                'crs': CRS.from_epsg(4326),
                'transform': transform
            })
            
            # Write temporary file first
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            with rasterio.open(temp_path, 'w', **profile) as dst:
                for i, (band_data, description) in enumerate(zip(bands_data, band_descriptions), 1):
                    dst.write(band_data, i)
                    dst.set_band_description(i, description)
                
                # Add metadata
                dst.update_tags(**self._create_gdal_metadata(metadata))
            
            # Convert to COG using GDAL
            self._convert_to_cog(temp_path, output_path)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            logger.info(f"Successfully exported COG to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting COG: {e}")
            return False
    
    def export_spectral_indices(self,
                               indices_data: Dict[str, np.ndarray],
                               bounds: Tuple[float, float, float, float],
                               output_path: str,
                               metadata: ExportMetadata) -> bool:
        """Export spectral indices as multi-band COG"""
        try:
            if not indices_data:
                return False
            
            # Get dimensions from first index
            first_index = next(iter(indices_data.values()))
            height, width = first_index.shape
            
            # Calculate transform
            transform = from_bounds(*bounds, width, height)
            
            # Prepare data
            bands_data = []
            band_descriptions = []
            
            for index_name, index_data in indices_data.items():
                # Normalize to 0-1 range and convert to uint16 for better precision
                normalized_data = np.clip((index_data + 1) / 2, 0, 1)  # Assume -1 to 1 range
                scaled_data = (normalized_data * 65535).astype(np.uint16)
                
                bands_data.append(scaled_data)
                band_descriptions.append(index_name.upper())
            
            # Create profile
            profile = self.cog_profile.copy()
            profile.update({
                'height': height,
                'width': width,
                'count': len(bands_data),
                'dtype': 'uint16',
                'crs': CRS.from_epsg(4326),
                'transform': transform,
                'nodata': 0
            })
            
            # Write temporary file
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            with rasterio.open(temp_path, 'w', **profile) as dst:
                for i, (band_data, description) in enumerate(zip(bands_data, band_descriptions), 1):
                    dst.write(band_data, i)
                    dst.set_band_description(i, description)
                
                # Add metadata
                metadata_tags = self._create_gdal_metadata(metadata)
                metadata_tags.update({
                    'SPECTRAL_INDICES': ','.join(indices_data.keys()),
                    'VALUE_RANGE': 'Scaled to 0-65535 from original -1 to 1 range'
                })
                dst.update_tags(**metadata_tags)
            
            # Convert to COG
            self._convert_to_cog(temp_path, output_path)
            os.unlink(temp_path)
            
            logger.info(f"Successfully exported spectral indices COG to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting spectral indices COG: {e}")
            return False
    
    def _convert_to_cog(self, input_path: str, output_path: str):
        """Convert GeoTIFF to Cloud Optimized GeoTIFF using GDAL"""
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # GDAL COG creation options
        cog_options = [
            '-of', 'COG',
            '-co', 'COMPRESS=LZW',
            '-co', 'PREDICTOR=2',
            '-co', 'BIGTIFF=IF_SAFER',
            '-co', 'BLOCKSIZE=512',
            '-co', 'OVERVIEW_RESAMPLING=AVERAGE',
            '-co', 'OVERVIEW_COUNT=5'
        ]
        
        # Use gdal.Translate for COG creation
        gdal.Translate(output_path, input_path, options=cog_options)
    
    def _create_gdal_metadata(self, metadata: ExportMetadata) -> Dict[str, str]:
        """Create GDAL-compatible metadata tags"""
        tags = {
            'TITLE': metadata.title,
            'DESCRIPTION': metadata.description,
            'AUTHOR': metadata.author,
            'CREATION_DATE': metadata.creation_date.isoformat(),
            'DATA_SOURCE': metadata.data_source,
            'PROCESSING_LEVEL': metadata.processing_level,
            'LICENSE': metadata.license
        }
        
        if metadata.keywords:
            tags['KEYWORDS'] = ','.join(metadata.keywords)
        
        if metadata.spatial_extent:
            minx, miny, maxx, maxy = metadata.spatial_extent
            tags['SPATIAL_EXTENT'] = f'{minx},{miny},{maxx},{maxy}'
        
        if metadata.temporal_extent:
            start, end = metadata.temporal_extent
            tags['TEMPORAL_START'] = start.isoformat()
            tags['TEMPORAL_END'] = end.isoformat()
        
        if metadata.quality_info:
            for key, value in metadata.quality_info.items():
                tags[f'QUALITY_{key.upper()}'] = str(value)
        
        return tags


class EnhancedShapefileExporter:
    """
    Enhanced Shapefile export with comprehensive metadata
    """
    
    def export_change_areas(self,
                          change_polygons: List[Polygon],
                          attributes: List[Dict],
                          output_path: str,
                          metadata: ExportMetadata,
                          coordinate_system: str = 'EPSG:4326') -> bool:
        """
        Export change detection areas as shapefile with attributes
        
        Args:
            change_polygons: List of change area polygons
            attributes: List of attribute dictionaries for each polygon
            output_path: Output shapefile path
            metadata: Export metadata
            coordinate_system: Coordinate reference system
            
        Returns:
            Success status
        """
        try:
            if not change_polygons:
                logger.warning("No change polygons to export")
                return False
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(attributes, geometry=change_polygons, crs=coordinate_system)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Export to shapefile
            gdf.to_file(output_path, driver='ESRI Shapefile')
            
            # Create metadata files
            self._create_shapefile_metadata(output_path, metadata, gdf)
            
            logger.info(f"Successfully exported shapefile to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting shapefile: {e}")
            return False
    
    def export_monitoring_points(self,
                                points: List[Point],
                                attributes: List[Dict],
                                output_path: str,
                                metadata: ExportMetadata) -> bool:
        """Export monitoring points as shapefile"""
        try:
            if not points:
                return False
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(attributes, geometry=points, crs='EPSG:4326')
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Export to shapefile
            gdf.to_file(output_path, driver='ESRI Shapefile')
            
            # Create metadata files
            self._create_shapefile_metadata(output_path, metadata, gdf)
            
            logger.info(f"Successfully exported points shapefile to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting points shapefile: {e}")
            return False
    
    def _create_shapefile_metadata(self, shapefile_path: str, metadata: ExportMetadata, gdf: gpd.GeoDataFrame):
        """Create comprehensive metadata files for shapefile"""
        base_path = os.path.splitext(shapefile_path)[0]
        
        # Create .xml metadata file (FGDC/ISO compliant)
        xml_path = f"{base_path}.xml"
        self._create_xml_metadata(xml_path, metadata, gdf)
        
        # Create .txt metadata file (human readable)
        txt_path = f"{base_path}.txt"
        self._create_text_metadata(txt_path, metadata, gdf)
        
        # Create .prj file if not exists (should be created by geopandas)
        prj_path = f"{base_path}.prj"
        if not os.path.exists(prj_path):
            self._create_prj_file(prj_path, gdf.crs)
    
    def _create_xml_metadata(self, xml_path: str, metadata: ExportMetadata, gdf: gpd.GeoDataFrame):
        """Create XML metadata file"""
        from xml.etree.ElementTree import Element, SubElement, tostring
        from xml.dom import minidom
        
        # Create root element
        root = Element('metadata')
        
        # Identification info
        idinfo = SubElement(root, 'idinfo')
        SubElement(idinfo, 'citation').text = metadata.title
        SubElement(idinfo, 'descript').text = metadata.description
        SubElement(idinfo, 'purpose').text = f"Generated by {metadata.author}"
        
        # Time period
        timeperd = SubElement(idinfo, 'timeperd')
        if metadata.temporal_extent:
            SubElement(timeperd, 'timeinfo').text = f"{metadata.temporal_extent[0]} to {metadata.temporal_extent[1]}"
        
        # Status
        status = SubElement(idinfo, 'status')
        SubElement(status, 'progress').text = 'Complete'
        SubElement(status, 'update').text = 'As needed'
        
        # Spatial domain
        spdom = SubElement(idinfo, 'spdom')
        if metadata.spatial_extent:
            bounding = SubElement(spdom, 'bounding')
            minx, miny, maxx, maxy = metadata.spatial_extent
            SubElement(bounding, 'westbc').text = str(minx)
            SubElement(bounding, 'eastbc').text = str(maxx)
            SubElement(bounding, 'northbc').text = str(maxy)
            SubElement(bounding, 'southbc').text = str(miny)
        
        # Keywords
        if metadata.keywords:
            keywords = SubElement(idinfo, 'keywords')
            theme = SubElement(keywords, 'theme')
            for keyword in metadata.keywords:
                SubElement(theme, 'themekey').text = keyword
        
        # Data quality
        if metadata.quality_info:
            dataqual = SubElement(root, 'dataqual')
            for key, value in metadata.quality_info.items():
                SubElement(dataqual, key).text = str(value)
        
        # Spatial reference
        spref = SubElement(root, 'spref')
        SubElement(spref, 'horizsys').text = str(gdf.crs)
        
        # Distribution info
        distinfo = SubElement(root, 'distinfo')
        SubElement(distinfo, 'stdorder').text = metadata.license
        
        # Metadata info
        metainfo = SubElement(root, 'metainfo')
        SubElement(metainfo, 'metd').text = metadata.creation_date.strftime('%Y%m%d')
        SubElement(metainfo, 'metc').text = metadata.author
        
        # Write XML file
        rough_string = tostring(root, 'unicode')
        reparsed = minidom.parseString(rough_string)
        
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(reparsed.toprettyxml(indent="  "))
    
    def _create_text_metadata(self, txt_path: str, metadata: ExportMetadata, gdf: gpd.GeoDataFrame):
        """Create human-readable text metadata file"""
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"SHAPEFILE METADATA\n")
            f.write(f"=" * 50 + "\n\n")
            
            f.write(f"Title: {metadata.title}\n")
            f.write(f"Description: {metadata.description}\n")
            f.write(f"Author: {metadata.author}\n")
            f.write(f"Creation Date: {metadata.creation_date}\n")
            f.write(f"Data Source: {metadata.data_source}\n")
            f.write(f"Processing Level: {metadata.processing_level}\n")
            f.write(f"License: {metadata.license}\n\n")
            
            if metadata.keywords:
                f.write(f"Keywords: {', '.join(metadata.keywords)}\n\n")
            
            if metadata.spatial_extent:
                minx, miny, maxx, maxy = metadata.spatial_extent
                f.write(f"Spatial Extent:\n")
                f.write(f"  West: {minx}\n")
                f.write(f"  East: {maxx}\n")
                f.write(f"  North: {maxy}\n")
                f.write(f"  South: {miny}\n\n")
            
            if metadata.temporal_extent:
                f.write(f"Temporal Extent: {metadata.temporal_extent[0]} to {metadata.temporal_extent[1]}\n\n")
            
            f.write(f"Coordinate System: {gdf.crs}\n")
            f.write(f"Feature Count: {len(gdf)}\n")
            f.write(f"Geometry Type: {gdf.geometry.geom_type.iloc[0] if not gdf.empty else 'Unknown'}\n\n")
            
            # Attribute information
            f.write(f"ATTRIBUTE INFORMATION\n")
            f.write(f"-" * 30 + "\n")
            for col in gdf.columns:
                if col != 'geometry':
                    dtype = str(gdf[col].dtype)
                    f.write(f"  {col}: {dtype}\n")
            
            if metadata.quality_info:
                f.write(f"\nQUALITY INFORMATION\n")
                f.write(f"-" * 30 + "\n")
                for key, value in metadata.quality_info.items():
                    f.write(f"  {key}: {value}\n")
    
    def _create_prj_file(self, prj_path: str, crs):
        """Create .prj file with projection information"""
        try:
            with open(prj_path, 'w') as f:
                f.write(crs.to_wkt())
        except Exception as e:
            logger.error(f"Error creating .prj file: {e}")


class KMLExporter:
    """
    Export data to KML/KMZ format for Google Earth
    """
    
    def export_change_areas_kml(self,
                              change_polygons: List[Polygon],
                              attributes: List[Dict],
                              output_path: str,
                              metadata: ExportMetadata,
                              create_kmz: bool = True) -> bool:
        """Export change areas as KML/KMZ"""
        try:
            kml = simplekml.Kml()
            kml.document.name = metadata.title
            kml.document.description = metadata.description
            
            # Create styles for different change types
            styles = self._create_kml_styles(kml)
            
            # Add polygons
            for polygon, attrs in zip(change_polygons, attributes):
                # Create placemark
                placemark = kml.newpolygon()
                placemark.name = attrs.get('name', f"Change Area {attrs.get('id', '')}")
                
                # Set geometry
                if hasattr(polygon, 'exterior'):
                    # Single polygon
                    coords = list(polygon.exterior.coords)
                    placemark.outerboundaryis = coords
                    
                    # Add holes if any
                    if hasattr(polygon, 'interiors'):
                        for interior in polygon.interiors:
                            placemark.innerboundaryis = list(interior.coords)
                
                # Set description with attributes
                description_html = self._create_kml_description(attrs, metadata)
                placemark.description = description_html
                
                # Set style based on change type
                change_type = attrs.get('change_type', 'unknown')
                if change_type in styles:
                    placemark.style = styles[change_type]
                
                # Set extended data
                extended_data = simplekml.ExtendedData()
                for key, value in attrs.items():
                    extended_data.newdata(key, str(value))
                placemark.extendeddata = extended_data
            
            # Save KML
            if create_kmz:
                # Create KMZ (compressed KML)
                output_path = output_path.replace('.kml', '.kmz')
                kml.savekmz(output_path)
            else:
                kml.save(output_path)
            
            logger.info(f"Successfully exported KML to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting KML: {e}")
            return False
    
    def export_monitoring_points_kml(self,
                                   points: List[Point],
                                   attributes: List[Dict],
                                   output_path: str,
                                   metadata: ExportMetadata) -> bool:
        """Export monitoring points as KML"""
        try:
            kml = simplekml.Kml()
            kml.document.name = metadata.title
            kml.document.description = metadata.description
            
            # Create point style
            point_style = simplekml.Style()
            point_style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/pushpin/red-pushpin.png'
            point_style.iconstyle.scale = 1.2
            
            # Add points
            for point, attrs in zip(points, attributes):
                placemark = kml.newpoint()
                placemark.name = attrs.get('name', f"Point {attrs.get('id', '')}")
                placemark.coords = [(point.x, point.y)]
                placemark.style = point_style
                
                # Set description
                description_html = self._create_kml_description(attrs, metadata)
                placemark.description = description_html
                
                # Set extended data
                extended_data = simplekml.ExtendedData()
                for key, value in attrs.items():
                    extended_data.newdata(key, str(value))
                placemark.extendeddata = extended_data
            
            # Save KML
            kml.save(output_path)
            
            logger.info(f"Successfully exported points KML to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting points KML: {e}")
            return False
    
    def _create_kml_styles(self, kml) -> Dict[str, simplekml.Style]:
        """Create KML styles for different change types"""
        styles = {}
        
        # Define colors for different change types
        color_map = {
            'urban_development': 'ff0000ff',    # Red
            'deforestation': 'ff00ff00',        # Green
            'mining': 'ffff0000',               # Blue
            'agriculture': 'ff00ffff',          # Yellow
            'infrastructure': 'ffff00ff',       # Magenta
            'unknown': 'ff808080'               # Gray
        }
        
        for change_type, color in color_map.items():
            style = simplekml.Style()
            style.polystyle.color = color
            style.polystyle.fill = 1
            style.polystyle.outline = 1
            style.linestyle.color = color
            style.linestyle.width = 2
            styles[change_type] = style
        
        return styles
    
    def _create_kml_description(self, attributes: Dict, metadata: ExportMetadata) -> str:
        """Create HTML description for KML placemark"""
        html = "<![CDATA["
        html += f"<h3>Change Detection Information</h3>"
        html += f"<table border='1' cellpadding='3'>"
        
        for key, value in attributes.items():
            if key != 'geometry':
                html += f"<tr><td><b>{key.replace('_', ' ').title()}</b></td><td>{value}</td></tr>"
        
        html += "</table>"
        html += f"<br><small>Generated by {metadata.author} on {metadata.creation_date.strftime('%Y-%m-%d')}</small>"
        html += "]]>"
        
        return html


class GeoJSONExporter:
    """
    Export data to GeoJSON format
    """
    
    def export_change_areas_geojson(self,
                                   change_polygons: List[Polygon],
                                   attributes: List[Dict],
                                   output_path: str,
                                   metadata: ExportMetadata) -> bool:
        """Export change areas as GeoJSON"""
        try:
            features = []
            
            for polygon, attrs in zip(change_polygons, attributes):
                feature = {
                    "type": "Feature",
                    "geometry": mapping(polygon),
                    "properties": attrs
                }
                features.append(feature)
            
            geojson = {
                "type": "FeatureCollection",
                "metadata": {
                    "title": metadata.title,
                    "description": metadata.description,
                    "author": metadata.author,
                    "creation_date": metadata.creation_date.isoformat(),
                    "data_source": metadata.data_source,
                    "license": metadata.license
                },
                "features": features
            }
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(geojson, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully exported GeoJSON to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting GeoJSON: {e}")
            return False


class ComprehensiveGISExporter:
    """
    Main class for comprehensive GIS data export
    """
    
    def __init__(self):
        self.cog_exporter = CloudOptimizedGeoTIFFExporter()
        self.shapefile_exporter = EnhancedShapefileExporter()
        self.kml_exporter = KMLExporter()
        self.geojson_exporter = GeoJSONExporter()
    
    def export_change_detection_package(self,
                                      change_data: Dict,
                                      output_dir: str,
                                      metadata: ExportMetadata,
                                      formats: List[str] = None) -> Dict[str, str]:
        """
        Export comprehensive change detection package in multiple formats
        
        Args:
            change_data: Dictionary containing change detection results
            output_dir: Output directory
            metadata: Export metadata
            formats: List of formats to export ['cog', 'shapefile', 'kml', 'geojson']
            
        Returns:
            Dictionary with format -> file path mappings
        """
        if formats is None:
            formats = ['cog', 'shapefile', 'kml', 'geojson']
        
        results = {}
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data components
        change_map = change_data.get('change_map')
        confidence_map = change_data.get('confidence_map')
        change_types = change_data.get('change_types')
        change_polygons = change_data.get('change_polygons', [])
        polygon_attributes = change_data.get('polygon_attributes', [])
        bounds = change_data.get('bounds')
        
        # Export COG
        if 'cog' in formats and change_map is not None and bounds is not None:
            cog_path = os.path.join(output_dir, f"change_detection_{metadata.creation_date.strftime('%Y%m%d')}.tif")
            if self.cog_exporter.export_change_detection_results(
                change_map, bounds, cog_path, metadata, confidence_map, change_types):
                results['cog'] = cog_path
        
        # Export Shapefile
        if 'shapefile' in formats and change_polygons:
            shp_path = os.path.join(output_dir, f"change_areas_{metadata.creation_date.strftime('%Y%m%d')}.shp")
            if self.shapefile_exporter.export_change_areas(
                change_polygons, polygon_attributes, shp_path, metadata):
                results['shapefile'] = shp_path
        
        # Export KML
        if 'kml' in formats and change_polygons:
            kml_path = os.path.join(output_dir, f"change_areas_{metadata.creation_date.strftime('%Y%m%d')}.kmz")
            if self.kml_exporter.export_change_areas_kml(
                change_polygons, polygon_attributes, kml_path, metadata):
                results['kml'] = kml_path
        
        # Export GeoJSON
        if 'geojson' in formats and change_polygons:
            geojson_path = os.path.join(output_dir, f"change_areas_{metadata.creation_date.strftime('%Y%m%d')}.geojson")
            if self.geojson_exporter.export_change_areas_geojson(
                change_polygons, polygon_attributes, geojson_path, metadata):
                results['geojson'] = geojson_path
        
        # Create summary report
        self._create_export_summary(output_dir, results, metadata)
        
        return results
    
    def export_spectral_indices_package(self,
                                       indices_data: Dict[str, np.ndarray],
                                       bounds: Tuple[float, float, float, float],
                                       output_dir: str,
                                       metadata: ExportMetadata) -> str:
        """Export spectral indices as COG with comprehensive metadata"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        cog_path = os.path.join(output_dir, f"spectral_indices_{metadata.creation_date.strftime('%Y%m%d')}.tif")
        
        success = self.cog_exporter.export_spectral_indices(
            indices_data, bounds, cog_path, metadata
        )
        
        if success:
            # Create documentation
            self._create_indices_documentation(output_dir, indices_data, metadata)
            return cog_path
        
        return None
    
    def _create_export_summary(self, output_dir: str, results: Dict[str, str], metadata: ExportMetadata):
        """Create export summary document"""
        summary_path = os.path.join(output_dir, "export_summary.txt")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("CHANGE DETECTION EXPORT SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Export Date: {datetime.now()}\n")
            f.write(f"Title: {metadata.title}\n")
            f.write(f"Description: {metadata.description}\n")
            f.write(f"Author: {metadata.author}\n")
            f.write(f"Data Source: {metadata.data_source}\n\n")
            
            f.write("EXPORTED FILES:\n")
            f.write("-" * 20 + "\n")
            for fmt, path in results.items():
                filename = os.path.basename(path)
                f.write(f"  {fmt.upper()}: {filename}\n")
            
            if metadata.spatial_extent:
                minx, miny, maxx, maxy = metadata.spatial_extent
                f.write(f"\nSPATIAL EXTENT:\n")
                f.write(f"  West: {minx}\n")
                f.write(f"  East: {maxx}\n")
                f.write(f"  North: {maxy}\n")
                f.write(f"  South: {miny}\n")
            
            if metadata.quality_info:
                f.write(f"\nQUALITY INFORMATION:\n")
                for key, value in metadata.quality_info.items():
                    f.write(f"  {key}: {value}\n")
            
            f.write(f"\nLicense: {metadata.license}\n")
    
    def _create_indices_documentation(self, output_dir: str, indices_data: Dict, metadata: ExportMetadata):
        """Create documentation for spectral indices"""
        doc_path = os.path.join(output_dir, "spectral_indices_documentation.txt")
        
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write("SPECTRAL INDICES DOCUMENTATION\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Dataset: {metadata.title}\n")
            f.write(f"Creation Date: {metadata.creation_date}\n")
            f.write(f"Author: {metadata.author}\n\n")
            
            f.write("INCLUDED INDICES:\n")
            f.write("-" * 20 + "\n")
            
            index_descriptions = {
                'ndvi': 'Normalized Difference Vegetation Index - Vegetation health and density',
                'ndwi': 'Normalized Difference Water Index - Water content and moisture',
                'ndbi': 'Normalized Difference Built-up Index - Urban and built-up areas',
                'evi': 'Enhanced Vegetation Index - Improved vegetation monitoring',
                'bsi': 'Bare Soil Index - Bare soil and exposed earth',
                'savi': 'Soil Adjusted Vegetation Index - Vegetation with soil correction'
            }
            
            for index_name in indices_data.keys():
                description = index_descriptions.get(index_name.lower(), 'Spectral index')
                f.write(f"  {index_name.upper()}: {description}\n")
            
            f.write(f"\nDATA FORMAT:\n")
            f.write(f"  File Type: Cloud Optimized GeoTIFF (COG)\n")
            f.write(f"  Data Type: 16-bit unsigned integer\n")
            f.write(f"  Value Range: 0-65535 (scaled from -1 to 1)\n")
            f.write(f"  No Data Value: 0\n")
            f.write(f"  Compression: LZW with predictor\n")
            
            f.write(f"\nUSAGE:\n")
            f.write(f"  To convert back to original values: (pixel_value / 65535) * 2 - 1\n")
            f.write(f"  Values near 0 indicate negative index values\n")
            f.write(f"  Values near 65535 indicate positive index values\n")


# Convenience functions for common exports
def export_change_detection_results(change_data: Dict, 
                                   output_dir: str,
                                   title: str = "Change Detection Results",
                                   description: str = "",
                                   formats: List[str] = None) -> Dict[str, str]:
    """
    Convenience function to export change detection results
    """
    metadata = ExportMetadata(
        title=title,
        description=description,
        keywords=['change detection', 'satellite imagery', 'remote sensing']
    )
    
    exporter = ComprehensiveGISExporter()
    return exporter.export_change_detection_package(change_data, output_dir, metadata, formats)


def export_spectral_indices(indices_data: Dict[str, np.ndarray],
                           bounds: Tuple[float, float, float, float],
                           output_dir: str,
                           title: str = "Spectral Indices") -> str:
    """
    Convenience function to export spectral indices
    """
    metadata = ExportMetadata(
        title=title,
        description="Multi-band spectral indices from satellite imagery",
        keywords=['spectral indices', 'remote sensing', 'satellite imagery']
    )
    
    exporter = ComprehensiveGISExporter()
    return exporter.export_spectral_indices_package(indices_data, bounds, output_dir, metadata) 