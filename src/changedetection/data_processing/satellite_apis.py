"""
Satellite data acquisition from various APIs
"""
import os
import requests
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
try:
    from sentinelhub import SHConfig, BBox, CRS, DataCollection, SentinelHubRequest, MimeType
    HAS_SENTINELHUB = True
except ImportError:
    HAS_SENTINELHUB = False
from shapely.geometry import Polygon
from django.conf import settings
from .bhoonidhi_api import BhoonidihiAPI
import logging

logger = logging.getLogger(__name__)

try:
    from landsatxplore.api import API as LandsatAPI
    from landsatxplore.earthexplorer import EarthExplorer
except ImportError:
    logger.warning("landsatxplore not available, using mock implementation")
    LandsatAPI = None
    EarthExplorer = None

try:
    import geopandas as gpd
except ImportError:
    logger.warning("geopandas not available")
    gpd = None

HAS_LANDSAT = LandsatAPI is not None

@dataclass
class SatelliteScene:
    """Data class for satellite scene metadata"""
    scene_id: str
    satellite: str
    acquisition_date: datetime
    cloud_cover: float
    geometry: Polygon
    download_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    file_size_mb: Optional[float] = None


if HAS_SENTINELHUB:
    class SentinelHubAPI:
        """Sentinel Hub API client for Sentinel-2 data"""
        
        def __init__(self):
            self.config = SHConfig()
            self.config.sh_client_id = getattr(settings, 'SENTINEL_HUB_CLIENT_ID', '')
            self.config.sh_client_secret = getattr(settings, 'SENTINEL_HUB_CLIENT_SECRET', '')
            self.config.sh_base_url = 'https://services.sentinel-hub.com'
            
            if not all([self.config.sh_client_id, self.config.sh_client_secret]):
                logger.warning("Sentinel Hub credentials not configured")
        
        def search_scenes(self, bbox: Tuple[float, float, float, float], 
                         start_date: datetime, end_date: datetime,
                         max_cloud_cover: float = 20.0) -> List[SatelliteScene]:
            """
            Search for Sentinel-2 scenes in the given area and time range
            
            Args:
                bbox: Bounding box (minx, miny, maxx, maxy) in WGS84
                start_date: Start date for search
                end_date: End date for search
                max_cloud_cover: Maximum cloud cover percentage
                
            Returns:
                List of SatelliteScene objects
            """
            try:
                # Convert bbox to SentinelHub BBox
                sh_bbox = BBox(bbox=bbox, crs=CRS.WGS84)
                
                # Create search request
                search_request = SentinelHubRequest(
                    evalscript=self._get_evalscript(),
                    input_data=[
                        SentinelHubRequest.input_data(
                            data_collection=DataCollection.SENTINEL2_L2A,
                            time_interval=(start_date.strftime('%Y-%m-%d'), 
                                         end_date.strftime('%Y-%m-%d')),
                            maxcc=max_cloud_cover / 100.0
                        )
                    ],
                    responses=[
                        SentinelHubRequest.output_response('default', MimeType.TIFF)
                    ],
                    bbox=sh_bbox,
                    size=(512, 512),
                    config=self.config
                )
                
                # Get available dates/scenes
                scenes = []
                available_dates = search_request.get_dates()
                
                for date in available_dates:
                    scene = SatelliteScene(
                        scene_id=f"S2_{date.strftime('%Y%m%d')}_{bbox[0]:.3f}_{bbox[1]:.3f}",
                        satellite='sentinel2',
                        acquisition_date=date,
                        cloud_cover=0.0,  # Would need additional API call for exact cloud cover
                        geometry=Polygon([
                            (bbox[0], bbox[1]),
                            (bbox[2], bbox[1]),
                            (bbox[2], bbox[3]),
                            (bbox[0], bbox[3]),
                            (bbox[0], bbox[1])
                        ])
                    )
                    scenes.append(scene)
                
                logger.info(f"Found {len(scenes)} Sentinel-2 scenes")
                return scenes
                
            except Exception as e:
                logger.error(f"Error searching Sentinel-2 scenes: {str(e)}")
                return []
        
        def download_scene(self, scene: SatelliteScene, output_dir: str) -> Optional[str]:
            """Download a Sentinel-2 scene"""
            try:
                bbox = list(scene.geometry.bounds)
                sh_bbox = BBox(bbox=bbox, crs=CRS.WGS84)
                
                request = SentinelHubRequest(
                    evalscript=self._get_evalscript(),
                    input_data=[
                        SentinelHubRequest.input_data(
                            data_collection=DataCollection.SENTINEL2_L2A,
                            time_interval=(scene.acquisition_date.strftime('%Y-%m-%d'), 
                                         (scene.acquisition_date + timedelta(days=1)).strftime('%Y-%m-%d'))
                        )
                    ],
                    responses=[
                        SentinelHubRequest.output_response('default', MimeType.TIFF)
                    ],
                    bbox=sh_bbox,
                    size=(1024, 1024),
                    config=self.config
                )
                
                # Download the image
                image_data = request.get_data()[0]
                
                # Save to file
                output_path = os.path.join(output_dir, f"{scene.scene_id}.tiff")
                os.makedirs(output_dir, exist_ok=True)
                
                with open(output_path, 'wb') as f:
                    f.write(image_data)
                
                logger.info(f"Downloaded Sentinel-2 scene to {output_path}")
                return output_path
                
            except Exception as e:
                logger.error(f"Error downloading Sentinel-2 scene: {str(e)}")
                return None
        
        def _get_evalscript(self) -> str:
            """Get evalscript for RGB image extraction"""
            return """
                //VERSION=3
                function setup() {
                    return {
                        input: ["B02", "B03", "B04"],
                        output: { bands: 3 }
                    };
                }
                
                function evaluatePixel(sample) {
                    return [sample.B04, sample.B03, sample.B02];
                }
            """
else:
    class SentinelHubAPI:
        """Mock Sentinel Hub API when sentinelhub is not available"""
        
        def __init__(self):
            logger.warning("SentinelHub not available - using mock implementation")
        
        def search_scenes(self, *args, **kwargs):
            return []
        
        def download_scene(self, *args, **kwargs):
            return None
        
        def _get_evalscript(self):
            return ""


if HAS_LANDSAT:
    class USGSLandsatAPI:
        """USGS Earth Explorer API client for Landsat data"""
        
        def __init__(self):
            self.username = getattr(settings, 'USGS_USERNAME', '')
            self.password = getattr(settings, 'USGS_PASSWORD', '')
            self.api = None
            self.ee = None
            
            if not all([self.username, self.password]):
                logger.warning("USGS credentials not configured")
        
        def _authenticate(self):
            """Authenticate with USGS services"""
            try:
                if not self.api:
                    self.api = LandsatAPI(self.username, self.password)
                if not self.ee:
                    self.ee = EarthExplorer(self.username, self.password)
                return True
            except Exception as e:
                logger.error(f"USGS authentication failed: {str(e)}")
                return False
        
        def search_scenes(self, bbox: Tuple[float, float, float, float], 
                         start_date: datetime, end_date: datetime,
                         max_cloud_cover: float = 20.0,
                         satellite: str = 'landsat8') -> List[SatelliteScene]:
            """Search for Landsat scenes"""
            if not self._authenticate():
                return []
            
            try:
                # Convert bbox to polygon
                polygon = Polygon([
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[1]),
                    (bbox[2], bbox[3]),
                    (bbox[0], bbox[3]),
                    (bbox[0], bbox[1])
                ])
                
                # Map satellite names to dataset names
                dataset_map = {
                    'landsat8': 'landsat_ot_c2_l2',
                    'landsat9': 'landsat_ot_c2_l2'
                }
                
                dataset = dataset_map.get(satellite, 'landsat_ot_c2_l2')
                
                # Search scenes
                scenes_data = self.api.search(
                    dataset=dataset,
                    latitude=(bbox[1] + bbox[3]) / 2,
                    longitude=(bbox[0] + bbox[2]) / 2,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    max_cloud_cover=max_cloud_cover
                )
                
                scenes = []
                for scene_data in scenes_data:
                    scene = SatelliteScene(
                        scene_id=scene_data['landsat_product_id'],
                        satellite=satellite,
                        acquisition_date=datetime.strptime(scene_data['date'], '%Y-%m-%d'),
                        cloud_cover=float(scene_data.get('cloud_cover', 0)),
                        geometry=polygon,
                        download_url=scene_data.get('download_url')
                    )
                    scenes.append(scene)
                
                logger.info(f"Found {len(scenes)} Landsat scenes")
                return scenes
                
            except Exception as e:
                logger.error(f"Error searching Landsat scenes: {str(e)}")
                return []
        
        def download_scene(self, scene: SatelliteScene, output_dir: str) -> Optional[str]:
            """Download a Landsat scene"""
            if not self._authenticate():
                return None
            
            try:
                os.makedirs(output_dir, exist_ok=True)
                
                # Download using Earth Explorer
                output_path = self.ee.download(
                    scene.scene_id,
                    output_dir=output_dir
                )
                
                logger.info(f"Downloaded Landsat scene to {output_path}")
                return output_path
                
            except Exception as e:
                logger.error(f"Error downloading Landsat scene: {str(e)}")
                return None
        
        def close(self):
            """Close API connections"""
            if self.api:
                self.api.logout()
            if self.ee:
                self.ee.logout()
else:
    class USGSLandsatAPI:
        """Mock USGS Landsat API when landsatxplore is not available"""
        
        def __init__(self):
            logger.warning("LandsatXplore not available - using mock implementation")
        
        def _authenticate(self):
            return False
            
        def search_scenes(self, *args, **kwargs):
            return []
        
        def download_scene(self, *args, **kwargs):
            return None
        
        def close(self):
            pass


class SatelliteDataManager:
    """Main interface for satellite data acquisition"""
    
    def __init__(self):
        self.sentinel_api = SentinelHubAPI()
        self.landsat_api = USGSLandsatAPI()
    
    def search_scenes(self, aoi_geometry: Polygon, start_date: datetime, 
                     end_date: datetime, satellite: str = 'sentinel2',
                     max_cloud_cover: float = 20.0) -> List[SatelliteScene]:
        """
        Search for satellite scenes
        
        Args:
            aoi_geometry: Area of interest as Shapely Polygon
            start_date: Start date for search
            end_date: End date for search
            satellite: Satellite type ('sentinel2', 'landsat8', 'landsat9')
            max_cloud_cover: Maximum cloud cover percentage
            
        Returns:
            List of available scenes
        """
        bbox = aoi_geometry.bounds
        
        if satellite == 'sentinel2':
            return self.sentinel_api.search_scenes(
                bbox, start_date, end_date, max_cloud_cover
            )
        elif satellite in ['landsat8', 'landsat9']:
            return self.landsat_api.search_scenes(
                bbox, start_date, end_date, max_cloud_cover, satellite
            )
        else:
            logger.error(f"Unsupported satellite: {satellite}")
            return []
    
    def download_scene(self, scene: SatelliteScene, output_dir: str) -> Optional[str]:
        """Download a satellite scene"""
        if scene.satellite == 'sentinel2':
            return self.sentinel_api.download_scene(scene, output_dir)
        elif scene.satellite in ['landsat8', 'landsat9']:
            return self.landsat_api.download_scene(scene, output_dir)
        else:
            logger.error(f"Unsupported satellite: {scene.satellite}")
            return None
    
    def get_best_scenes(self, aoi_geometry: Polygon, target_date: datetime,
                       satellite: str = 'sentinel2', days_tolerance: int = 30,
                       max_cloud_cover: float = 20.0) -> List[SatelliteScene]:
        """
        Get the best available scenes around a target date
        
        Args:
            aoi_geometry: Area of interest
            target_date: Target acquisition date
            satellite: Satellite type
            days_tolerance: Search window in days around target date
            max_cloud_cover: Maximum cloud cover
            
        Returns:
            List of best scenes sorted by proximity to target date and cloud cover
        """
        start_date = target_date - timedelta(days=days_tolerance)
        end_date = target_date + timedelta(days=days_tolerance)
        
        scenes = self.search_scenes(
            aoi_geometry, start_date, end_date, satellite, max_cloud_cover
        )
        
        # Sort by date proximity and cloud cover
        def scene_score(scene):
            date_diff = abs((scene.acquisition_date.date() - target_date.date()).days)
            return (date_diff, scene.cloud_cover)
        
        scenes.sort(key=scene_score)
        return scenes
    
    def close(self):
        """Close all API connections"""
        self.landsat_api.close() 