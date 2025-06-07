"""
Recognition compatibility layer to handle NumPy version issues
"""
import pickle
import numpy as np
import sys
import warnings

# Monkey patch for NumPy compatibility
if not hasattr(np, '_core'):
    np._core = np.core

class CompatibleUnpickler(pickle.Unpickler):
    """Custom unpickler that handles NumPy version differences"""
    
    def find_class(self, module, name):
        # Handle numpy._core -> numpy.core mapping
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        
        # Handle sklearn version differences
        if module.startswith('sklearn.'):
            try:
                return super().find_class(module, name)
            except (ImportError, AttributeError):
                # Try without dots for older sklearn versions
                if '.' in name:
                    parts = name.split('.')
                    return super().find_class(module + '.' + '.'.join(parts[:-1]), parts[-1])
        
        return super().find_class(module, name)

def load_compatible_pickle(file_path):
    """Load pickle file with compatibility handling"""
    try:
        # First try normal loading
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        # Try with compatibility layer
        try:
            with open(file_path, 'rb') as f:
                return CompatibleUnpickler(f).load()
        except Exception as e2:
            warnings.warn(f"Failed to load {file_path}: {e2}")
            raise

# Patch pickle.load globally for this module
_original_pickle_load = pickle.load

def patched_pickle_load(file, **kwargs):
    """Patched pickle.load that handles compatibility"""
    if hasattr(file, 'read'):
        # File-like object
        try:
            file.seek(0)
            return _original_pickle_load(file, **kwargs)
        except Exception:
            file.seek(0)
            return CompatibleUnpickler(file).load()
    else:
        # Assume it's already loaded data
        return _original_pickle_load(file, **kwargs)

# Apply the patch
pickle.load = patched_pickle_load