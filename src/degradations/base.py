from abc import ABC, abstractmethod

class BaseDegradation(ABC):
    """
    Abstract base class for image degradations. All degradation classes should inherit from this class.
    """
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass
    
    
    @abstractmethod
    def __call__(self, x0, t):
        """
        Apply the degradation to the clean input image for a single timestep.
        Returns the degraded image at timestep t.

        Args:
            x0 (torch.Tensor): The original clean image to be degraded.
            t (torch.Tensor): The timestep at which to return the degraded image.

        Returns:
            torch.Tensor: The degraded image at timestep t.
        """
        pass
        