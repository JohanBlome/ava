"""
AVA Device and Codec Identifier Module

This module provides general solutions for identifying devices and codecs,
replacing hardcoded patterns with dynamic, extensible identification logic.
"""

import hashlib
import colorsys
import logging
from typing import Dict, List, Set, Optional, Tuple
import re


class DeviceCodecIdentifier:
    """General device and codec identification and styling system"""
    
    def __init__(self, logger: logging.Logger = None):
        """Initialize the identifier with optional logger"""
        self.logger = logger or logging.getLogger(__name__)
        
        # Dynamic storage for discovered devices and codecs
        self._discovered_devices: Set[str] = set()
        self._discovered_codecs: Set[str] = set()
        self._device_colors: Dict[str, str] = {}
        self._codec_types: Dict[str, str] = {}
        self._codec_line_styles: Dict[str, str] = {}
        
        # Initialize with known patterns but make them extensible
        self._initialize_codec_patterns()
        self._initialize_color_palette()
        
    def _initialize_codec_patterns(self):
        """Initialize codec identification patterns"""
        # Codec type patterns - these can be extended
        self._codec_patterns = {
            'av1': [r'av1', r'ao\.av1', r'c2\.av1\.encoder'],
            'hevc': [r'hevc', r'h265', r'c2\.(dolby|qti)\.hevc\.encoder', r'c2\.qti\.hevc\.encoder\.hdr'],
            'avc': [r'avc', r'h264', r'c2\.avc\.encoder'],
            'vp8': [r'vp8', r'c2\.vp8\.encoder'],
            'vp9': [r'vp9', r'c2\.vp9\.encoder'],
            'mpeg4': [r'mpeg4', r'xvid', r'divx'],
            'theora': [r'theora'],
            'mjpeg': [r'mjpeg', r'motion.*jpeg'],
        }
        
        # Line styles for codec types
        self._default_line_styles = {
            'av1': 'solid',
            'hevc': 'dash',
            'avc': 'dot',
            'vp8': 'dashdot',
            'vp9': 'longdash',
            'mpeg4': 'dashdotdot',
            'theora': 'solid',
            'mjpeg': 'dash',
            'other': 'solid'
        }
        
    def _initialize_color_palette(self):
        """Initialize a diverse color palette for devices and codecs"""
        # Use a scientifically designed color palette that's colorblind-friendly
        self._color_palette = [
            '#1f77b4',  # Blue
            '#ff7f0e',  # Orange  
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
            '#bcbd22',  # Olive
            '#17becf',  # Cyan
            '#aec7e8',  # Light Blue
            '#ffbb78',  # Light Orange
            '#98df8a',  # Light Green
            '#ff9896',  # Light Red
            '#c5b0d5',  # Light Purple
            '#c49c94',  # Light Brown
            '#f7b6d3',  # Light Pink
            '#c7c7c7',  # Light Gray
            '#dbdb8d',  # Light Olive
            '#9edae5',  # Light Cyan
        ]
        self._color_index = 0
        
    def add_codec_pattern(self, codec_type: str, patterns: List[str], line_style: str = None):
        """Add new codec patterns dynamically"""
        if codec_type not in self._codec_patterns:
            self._codec_patterns[codec_type] = []
        self._codec_patterns[codec_type].extend(patterns)
        
        if line_style and codec_type not in self._default_line_styles:
            self._default_line_styles[codec_type] = line_style
            
        self.logger.info(f"Added codec patterns for {codec_type}: {patterns}")
        
    def get_codec_type(self, codec_name: str) -> str:
        """Extract codec type from codec name using pattern matching"""
        if codec_name in self._codec_types:
            return self._codec_types[codec_name]
            
        codec_lower = codec_name.lower()
        
        # Try pattern matching for known codec types
        for codec_type, patterns in self._codec_patterns.items():
            for pattern in patterns:
                if re.search(pattern, codec_lower):
                    self._codec_types[codec_name] = codec_type
                    self._discovered_codecs.add(codec_name)
                    return codec_type
                    
        # If no pattern matches, classify as 'other'
        self._codec_types[codec_name] = 'other'
        self._discovered_codecs.add(codec_name)
        return 'other'
        
    def get_codec_line_style(self, codec_type: str) -> str:
        """Get consistent line style for codec type"""
        return self._default_line_styles.get(codec_type, 'solid')
        
    def is_custom_labeled_codec(self, codec_name: str) -> bool:
        """Check if a codec name is a custom label vs standard codec name"""
        # Standard codec names typically contain technical patterns
        standard_indicators = [
            r'encoder', r'codec', r'c2\.', r'ao\.',  # Android codec indicators
            r'h264', r'h265', r'av1', r'vp8', r'vp9',  # Codec family names
            r'\.dll', r'\.so', r'\.dylib',  # Library extensions
            r'lib.*', r'ffmpeg', r'gstreamer'  # Library prefixes
        ]
        
        # Check if it contains standard indicators
        for indicator in standard_indicators:
            if re.search(indicator, codec_name.lower()):
                return False
                
        # Custom labels are typically short, simple names
        # without technical patterns
        return (
            len(codec_name) < 25 and  # Short names
            not re.search(r'[._-]', codec_name) and  # No separators
            not re.search(r'\d+', codec_name) and  # No version numbers
            codec_name.isalnum() or ' ' in codec_name  # Alphanumeric or with spaces
        )
        
    def get_device_color(self, device_name: str) -> str:
        """Get consistent color for device, generating new colors as needed"""
        if device_name in self._device_colors:
            return self._device_colors[device_name]
            
        # Add to discovered devices
        self._discovered_devices.add(device_name)
        
        # Generate a consistent color
        color = self._generate_consistent_color(device_name)
        self._device_colors[device_name] = color
        
        return color
        
    def _generate_consistent_color(self, name: str) -> str:
        """Generate a consistent color based on name hash"""
        # Use hash of the name to generate a consistent color
        hash_obj = hashlib.md5(name.encode('utf-8'))
        hash_int = int(hash_obj.hexdigest()[:8], 16)
        
        # If we have unused colors in our palette, use them first
        if self._color_index < len(self._color_palette):
            color = self._color_palette[self._color_index]
            self._color_index += 1
            return color
            
        # Generate a color from the hash for additional devices
        hue = (hash_int % 360) / 360.0
        saturation = 0.6 + (hash_int % 40) / 100.0  # Vary saturation
        value = 0.7 + (hash_int % 30) / 100.0  # Vary brightness
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        color = f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
        
        return color
        
    def get_trace_name(self, codec: str, device: str) -> str:
        """Generate consistent trace name for filtering"""
        return f"{codec} ({device})"
        
    def get_available_codecs(self) -> List[str]:
        """Get list of all discovered codecs"""
        return sorted(list(self._discovered_codecs))
        
    def get_available_devices(self) -> List[str]:
        """Get list of all discovered devices"""
        return sorted(list(self._discovered_devices))
        
    def get_device_color_mapping(self) -> Dict[str, str]:
        """Get mapping of device names to colors"""
        return self._device_colors.copy()
        
    def get_codec_type_mapping(self) -> Dict[str, str]:
        """Get mapping of codec names to types"""
        return self._codec_types.copy()
        
    def reset_discoveries(self):
        """Reset all discovered devices and codecs (useful for testing)"""
        self._discovered_devices.clear()
        self._discovered_codecs.clear()
        self._device_colors.clear()
        self._codec_types.clear()
        self._color_index = 0
        
    def export_config(self) -> Dict:
        """Export current configuration for persistence"""
        return {
            'codec_patterns': self._codec_patterns,
            'default_line_styles': self._default_line_styles,
            'color_palette': self._color_palette,
            'discovered_devices': list(self._discovered_devices),
            'discovered_codecs': list(self._discovered_codecs),
            'device_colors': self._device_colors,
            'codec_types': self._codec_types,
        }
        
    def import_config(self, config: Dict):
        """Import configuration from persistence"""
        if 'codec_patterns' in config:
            self._codec_patterns.update(config['codec_patterns'])
        if 'default_line_styles' in config:
            self._default_line_styles.update(config['default_line_styles'])
        if 'color_palette' in config:
            self._color_palette = config['color_palette']
        if 'discovered_devices' in config:
            self._discovered_devices = set(config['discovered_devices'])
        if 'discovered_codecs' in config:
            self._discovered_codecs = set(config['discovered_codecs'])
        if 'device_colors' in config:
            self._device_colors.update(config['device_colors'])
        if 'codec_types' in config:
            self._codec_types.update(config['codec_types'])
            
        self.logger.info("Configuration imported successfully")


class DeviceCodecManager:
    """Manager class that provides a unified interface for device/codec operations"""
    
    def __init__(self, logger: logging.Logger = None):
        """Initialize the manager"""
        self.logger = logger or logging.getLogger(__name__)
        self.identifier = DeviceCodecIdentifier(logger)
        
    def process_test_results(self, test_results) -> Dict:
        """Process test results to discover and categorize all devices and codecs"""
        discovered_info = {
            'devices': set(),
            'codecs': set(),
            'device_codec_pairs': set(),
            'codec_types': {},
            'custom_codecs': set(),
            'standard_codecs': set()
        }
        
        for result in test_results:
            if hasattr(result, 'device_serial') and result.device_serial:
                discovered_info['devices'].add(result.device_serial)
                
            if hasattr(result, 'test_data') and result.test_data:
                # Process CSV files to find codecs
                csv_files = result.test_data.get('stats_csv', [])
                if isinstance(csv_files, list):
                    for csv_file in csv_files:
                        try:
                            import pandas as pd
                            df = pd.read_csv(csv_file)
                            if 'codec' in df.columns:
                                for codec in df['codec'].unique():
                                    discovered_info['codecs'].add(codec)
                                    discovered_info['device_codec_pairs'].add((result.device_serial, codec))
                                    
                                    # Classify codec
                                    codec_type = self.identifier.get_codec_type(codec)
                                    discovered_info['codec_types'][codec] = codec_type
                                    
                                    if self.identifier.is_custom_labeled_codec(codec):
                                        discovered_info['custom_codecs'].add(codec)
                                    else:
                                        discovered_info['standard_codecs'].add(codec)
                        except Exception as e:
                            self.logger.warning(f"Could not process CSV file {csv_file}: {e}")
                            
        return discovered_info
        
    def get_styling_info(self, codec: str, device: str) -> Dict:
        """Get complete styling information for a codec/device combination"""
        codec_type = self.identifier.get_codec_type(codec)
        is_custom = self.identifier.is_custom_labeled_codec(codec)
        
        # For custom codecs, use codec name for color; for standard codecs, use device name
        color_source = codec if is_custom else device
        color = self.identifier.get_device_color(color_source)
        
        return {
            'color': color,
            'codec_type': codec_type,
            'line_style': self.identifier.get_codec_line_style(codec_type),
            'trace_name': self.identifier.get_trace_name(codec, device),
            'is_custom_codec': is_custom,
            'color_source': color_source
        }
