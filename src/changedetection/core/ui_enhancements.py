"""
UI Enhancements for AOI Management
=================================
Advanced UI components for better AOI management including:
- Shape file import/export functionality
- Coordinate validation and transformation
- Advanced drawing tools and editing
- Bulk operations and batch processing
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import tempfile
import zipfile
import io
from dataclasses import dataclass
from django.core.exceptions import ValidationError
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from decimal import Decimal, InvalidOperation

import geopandas as gpd
import fiona
import pyproj
from django.core.files.uploadedfile import UploadedFile

logger = logging.getLogger(__name__)


def validate_coordinates(lat: float, lng: float) -> bool:
    """Simple coordinate validation without PostGIS"""
    return -90 <= lat <= 90 and -180 <= lng <= 180


def validate_bbox(north: float, south: float, east: float, west: float) -> bool:
    """Validate bounding box coordinates"""
    return (validate_coordinates(north, east) and 
            validate_coordinates(south, west) and
            north > south and east > west)


class CoordinateValidator:
    """Simple coordinate validation utilities without PostGIS dependencies"""
    
    def __init__(self):
        self.supported_crs = {
            'EPSG:4326': 'WGS84 Geographic',
            'EPSG:3857': 'Web Mercator'
        }
    
    def validate_coordinates(self, coordinates: List[List[float]], source_crs: str = 'EPSG:4326') -> Dict:
        """
        Basic coordinate validation
        
        Args:
            coordinates: List of coordinate pairs [[lon, lat], ...]
            source_crs: Source coordinate reference system
            
        Returns:
            Validation result with errors and warnings
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'corrected_coordinates': None,
            'metadata': {
                'total_points': len(coordinates),
                'source_crs': source_crs,
                'geometry_type': 'Polygon'
            }
        }
        
        try:
            # Basic structure validation
            if len(coordinates) < 3:
                result['errors'].append("Polygon must have at least 3 coordinates")
                result['valid'] = False
                return result
            
            # Close polygon if not closed
            corrected_coords = list(coordinates)
            if coordinates[0] != coordinates[-1]:
                corrected_coords.append(coordinates[0])
                result['warnings'].append("Polygon was not closed, automatically closed")
            
            # Basic bounds checking for WGS84
            if source_crs == 'EPSG:4326':
                for i, coord in enumerate(corrected_coords):
                    lon, lat = coord[0], coord[1]
                    if not (-180 <= lon <= 180):
                        result['errors'].append(f"Point {i}: Longitude {lon} out of valid range [-180, 180]")
                    if not (-90 <= lat <= 90):
                        result['errors'].append(f"Point {i}: Latitude {lat} out of valid range [-90, 90]")
            
            result['corrected_coordinates'] = corrected_coords
            
            # Calculate basic metadata
            if corrected_coords:
                lons = [c[0] for c in corrected_coords]
                lats = [c[1] for c in corrected_coords]
                result['metadata'].update({
                    'bounds': [min(lons), min(lats), max(lons), max(lats)],
                    'centroid': [sum(lons)/len(lons), sum(lats)/len(lats)]
                })
            
            # Final validation
            if result['errors']:
                result['valid'] = False
            
        except Exception as e:
            result['errors'].append(f"Coordinate validation failed: {str(e)}")
            result['valid'] = False
        
        return result


class ShapeFileManager:
    """Handle shapefile import/export operations"""
    
    def __init__(self):
        self.supported_formats = ['.shp', '.geojson', '.kml', '.gpx']
        self.coordinate_validator = CoordinateValidator()
    
    def import_shapefile(self, uploaded_file: UploadedFile) -> Dict:
        """
        Import AOIs from uploaded shapefile or geospatial file
        
        Args:
            uploaded_file: Django uploaded file object
            
        Returns:
            Import result with extracted AOIs
        """
        result = {
            'success': False,
            'aois': [],
            'errors': [],
            'warnings': [],
            'metadata': {}
        }
        
        try:
            # Create temporary directory for extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Handle different file types
                file_extension = Path(uploaded_file.name).suffix.lower()
                
                if file_extension == '.zip':
                    # Extract shapefile components
                    extracted_path = self._extract_shapefile_zip(uploaded_file, temp_path)
                    if not extracted_path:
                        result['errors'].append("No valid shapefile found in ZIP archive")
                        return result
                elif file_extension in ['.geojson', '.kml', '.gpx']:
                    # Save single file
                    file_path = temp_path / uploaded_file.name
                    with open(file_path, 'wb') as f:
                        for chunk in uploaded_file.chunks():
                            f.write(chunk)
                    extracted_path = file_path
                else:
                    result['errors'].append(f"Unsupported file format: {file_extension}")
                    return result
                
                # Read geospatial data
                try:
                    gdf = gpd.read_file(extracted_path)
                    
                    # Validate and process features
                    result['metadata'] = {
                        'feature_count': len(gdf),
                        'original_crs': str(gdf.crs) if gdf.crs else 'Unknown',
                        'columns': list(gdf.columns)
                    }
                    
                    # Transform to WGS84 if needed
                    if gdf.crs and gdf.crs != 'EPSG:4326':
                        gdf = gdf.to_crs('EPSG:4326')
                    
                    # Extract AOI data
                    for idx, row in gdf.iterrows():
                        aoi_data = self._extract_aoi_from_feature(row, idx)
                        if aoi_data['valid']:
                            result['aois'].append(aoi_data)
                        else:
                            result['warnings'].extend(aoi_data['errors'])
                    
                    if result['aois']:
                        result['success'] = True
                    else:
                        result['errors'].append("No valid AOI features found in file")
                
                except Exception as e:
                    result['errors'].append(f"Error reading geospatial file: {str(e)}")
        
        except Exception as e:
            result['errors'].append(f"File processing error: {str(e)}")
        
        return result
    
    def _extract_shapefile_zip(self, zip_file: UploadedFile, extract_path: Path) -> Optional[Path]:
        """Extract shapefile from ZIP archive"""
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            # Find .shp file
            shp_files = list(extract_path.glob('*.shp'))
            if shp_files:
                return shp_files[0]
            
            # Check subdirectories
            for subdir in extract_path.iterdir():
                if subdir.is_dir():
                    shp_files = list(subdir.glob('*.shp'))
                    if shp_files:
                        return shp_files[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting shapefile ZIP: {e}")
            return None
    
    def _extract_aoi_from_feature(self, feature, index: int) -> Dict:
        """Extract AOI data from a single geospatial feature"""
        aoi_data = {
            'valid': False,
            'name': '',
            'description': '',
            'geometry': None,
            'area_km2': 0.0,
            'errors': []
        }
        
        try:
            # Get geometry
            geom = feature.geometry
            if geom is None or geom.is_empty:
                aoi_data['errors'].append(f"Feature {index}: Empty geometry")
                return aoi_data
            
            # Convert to polygon if needed
            if geom.geom_type == 'MultiPolygon':
                # Take largest polygon
                geom = max(geom.geoms, key=lambda p: p.area)
            elif geom.geom_type not in ['Polygon']:
                aoi_data['errors'].append(f"Feature {index}: Unsupported geometry type {geom.geom_type}")
                return aoi_data
            
            # Validate geometry
            if not geom.is_valid:
                geom = make_valid(geom)
            
            # Extract attributes for naming
            name_candidates = ['name', 'Name', 'NAME', 'id', 'ID', 'FID']
            name = f"Imported AOI {index}"
            
            for col in name_candidates:
                if col in feature.index and pd.notna(feature[col]):
                    name = str(feature[col])
                    break
            
            # Description from other attributes
            desc_parts = []
            for col in feature.index:
                if col not in ['geometry'] + name_candidates and pd.notna(feature[col]):
                    desc_parts.append(f"{col}: {feature[col]}")
            
            description = "; ".join(desc_parts[:3])  # Limit description length
            
            # Calculate area
            # Transform to equal-area projection for accurate area calculation
            geom_proj = transform(
                pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True).transform,
                geom
            )
            area_km2 = geom_proj.area / 1_000_000  # Convert m² to km²
            
            # Convert to GeoJSON
            geojson_geom = mapping(geom)
            
            aoi_data.update({
                'valid': True,
                'name': name,
                'description': description,
                'geometry': geojson_geom,
                'area_km2': area_km2
            })
            
        except Exception as e:
            aoi_data['errors'].append(f"Feature {index}: Processing error - {str(e)}")
        
        return aoi_data
    
    def export_aois_to_shapefile(self, aois: List[Dict], output_format: str = 'shapefile') -> bytes:
        """
        Export AOIs to various geospatial formats
        
        Args:
            aois: List of AOI dictionaries
            output_format: Output format ('shapefile', 'geojson', 'kml')
            
        Returns:
            Bytes content of exported file
        """
        try:
            # Create GeoDataFrame
            features = []
            for aoi in aois:
                geom = shape(aoi['geometry'])
                features.append({
                    'name': aoi['name'],
                    'description': aoi.get('description', ''),
                    'area_km2': aoi.get('area_km2', 0),
                    'satellite_source': aoi.get('satellite_source', ''),
                    'created_at': aoi.get('created_at', ''),
                    'geometry': geom
                })
            
            gdf = gpd.GeoDataFrame(features, crs='EPSG:4326')
            
            # Export to requested format
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                if output_format == 'shapefile':
                    output_file = temp_path / 'aois.shp'
                    gdf.to_file(output_file, driver='ESRI Shapefile')
                    
                    # Create ZIP archive with all shapefile components
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for file_path in temp_path.glob('aois.*'):
                            zip_file.write(file_path, file_path.name)
                    
                    return zip_buffer.getvalue()
                
                elif output_format == 'geojson':
                    output_file = temp_path / 'aois.geojson'
                    gdf.to_file(output_file, driver='GeoJSON')
                    
                    with open(output_file, 'rb') as f:
                        return f.read()
                
                elif output_format == 'kml':
                    output_file = temp_path / 'aois.kml'
                    gdf.to_file(output_file, driver='KML')
                    
                    with open(output_file, 'rb') as f:
                        return f.read()
        
        except Exception as e:
            logger.error(f"Export error: {e}")
            raise


class AOIBatchProcessor:
    """Handle batch operations on multiple AOIs"""
    
    def __init__(self):
        self.supported_operations = [
            'bulk_activate', 'bulk_deactivate', 'bulk_delete',
            'bulk_change_satellite', 'bulk_update_thresholds'
        ]
    
    def process_batch_operation(self, operation: str, aoi_ids: List[str], parameters: Dict = None) -> Dict:
        """
        Process batch operations on multiple AOIs
        
        Args:
            operation: Operation type
            aoi_ids: List of AOI IDs to process
            parameters: Operation-specific parameters
            
        Returns:
            Processing result
        """
        if operation not in self.supported_operations:
            return {
                'success': False,
                'error': f"Unsupported operation: {operation}"
            }
        
        results = {
            'success': True,
            'processed': 0,
            'failed': 0,
            'errors': []
        }
        
        try:
            from .models import AreaOfInterest
            
            aois = AreaOfInterest.objects.filter(id__in=aoi_ids)
            
            for aoi in aois:
                try:
                    if operation == 'bulk_activate':
                        aoi.is_active = True
                        aoi.save()
                    
                    elif operation == 'bulk_deactivate':
                        aoi.is_active = False
                        aoi.save()
                    
                    elif operation == 'bulk_delete':
                        aoi.delete()
                    
                    elif operation == 'bulk_change_satellite':
                        if parameters and 'satellite_source' in parameters:
                            aoi.satellite_source = parameters['satellite_source']
                            aoi.save()
                    
                    elif operation == 'bulk_update_thresholds':
                        if parameters:
                            if 'cloud_cover_threshold' in parameters:
                                aoi.cloud_cover_threshold = parameters['cloud_cover_threshold']
                            if 'change_threshold' in parameters:
                                aoi.change_threshold = parameters['change_threshold']
                            aoi.save()
                    
                    results['processed'] += 1
                    
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"AOI {aoi.id}: {str(e)}")
            
            if results['failed'] > 0:
                results['success'] = False
        
        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
        
        return results


def validate_aoi_coordinates_view(request):
    """Django view for coordinate validation"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            coordinates = data.get('coordinates', [])
            crs = data.get('crs', 'EPSG:4326')
            
            validator = CoordinateValidator()
            result = validator.validate_coordinates(coordinates, crs)
            
            return JsonResponse(result)
        
        except Exception as e:
            return JsonResponse({
                'valid': False,
                'errors': [str(e)]
            })
    
    return JsonResponse({'error': 'Only POST method allowed'})


def import_shapefile_view(request):
    """Django view for shapefile import"""
    if request.method == 'POST' and request.FILES.get('shapefile'):
        try:
            uploaded_file = request.FILES['shapefile']
            
            manager = ShapeFileManager()
            result = manager.import_shapefile(uploaded_file)
            
            return JsonResponse(result)
        
        except Exception as e:
            return JsonResponse({
                'success': False,
                'errors': [str(e)]
            })
    
    return JsonResponse({'error': 'No file uploaded'})


def export_aois_view(request):
    """Django view for AOI export"""
    if request.method == 'GET':
        try:
            format_type = request.GET.get('format', 'shapefile')
            aoi_ids = request.GET.getlist('aoi_ids')
            
            if not aoi_ids:
                return JsonResponse({'error': 'No AOIs selected'})
            
            # Get AOI data
            from .models import AreaOfInterest
            aois = AreaOfInterest.objects.filter(id__in=aoi_ids)
            
            aoi_data = []
            for aoi in aois:
                aoi_data.append({
                    'name': aoi.name,
                    'description': aoi.description,
                    'geometry': aoi.geometry_geojson,
                    'area_km2': aoi.area_km2,
                    'satellite_source': aoi.satellite_source,
                    'created_at': aoi.created_at.isoformat()
                })
            
            # Export
            manager = ShapeFileManager()
            exported_data = manager.export_aois_to_shapefile(aoi_data, format_type)
            
            # Set appropriate content type and filename
            content_types = {
                'shapefile': 'application/zip',
                'geojson': 'application/json',
                'kml': 'application/vnd.google-earth.kml+xml'
            }
            
            filenames = {
                'shapefile': 'aois.zip',
                'geojson': 'aois.geojson',
                'kml': 'aois.kml'
            }
            
            response = HttpResponse(
                exported_data,
                content_type=content_types.get(format_type, 'application/octet-stream')
            )
            response['Content-Disposition'] = f'attachment; filename="{filenames.get(format_type, "aois.zip")}"'
            
            return response
        
        except Exception as e:
            return JsonResponse({'error': str(e)})
    
    return JsonResponse({'error': 'Only GET method allowed'})


def batch_process_aois_view(request):
    """Django view for batch AOI operations"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            operation = data.get('operation')
            aoi_ids = data.get('aoi_ids', [])
            parameters = data.get('parameters', {})
            
            processor = AOIBatchProcessor()
            result = processor.process_batch_operation(operation, aoi_ids, parameters)
            
            return JsonResponse(result)
        
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'error': 'Only POST method allowed'}) 