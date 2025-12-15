"""
ComfyUI Progress Integration

This module provides progress callbacks for Core ML nodes to integrate with ComfyUI's progress bar.
"""

import comfy.utils
import comfy.model_management

class ProgressCallback:
    """Wrapper for ComfyUI progress updates"""
    
    def __init__(self, total_steps=None):
        self.total_steps = total_steps
        self.current_step = 0
        
    def update(self, step=None):
        """Update progress by one step or to a specific step"""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        # ComfyUI automatically tracks progress through model calls
        # This is mainly for interruption handling
        try:
            if hasattr(comfy.model_management, 'throw_exception_if_processing_interrupted'):
                comfy.model_management.throw_exception_if_processing_interrupted()
        except:
            pass
    
    def set_total(self, total):
        """Set total steps for progress tracking"""
        self.total_steps = total
    
    def reset(self):
        """Reset progress counter"""
        self.current_step = 0


def create_progress_callback(total_steps=None):
    """Create a progress callback for ComfyUI integration"""
    return ProgressCallback(total_steps)
