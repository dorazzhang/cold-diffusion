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
        Apply the degradation to the clean input image. Returns degraded images at each timestep from 0 to t.

        Args:
            x0 (torch.Tensor): The original clean image to be degraded.
            t (torch.Tensor): The final time step to degrade the input image to.

        Returns:
            torch.Tensor: The degraded images at each timestep.
        """
        pass
        