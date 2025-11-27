"""
Service Container for Dependency Injection.
Provides a centralized way to access all services with proper lifecycle management.
"""
from typing import Dict, Any, Optional, Type, Callable
from functools import lru_cache
import streamlit as st
from pathlib import Path


class ServiceContainer:
    """
    Dependency injection container for managing service instances.
    
    Usage:
        container = ServiceContainer()
        container.register('model', load_model)
        model = container.get('model')
    """
    
    _instance: Optional['ServiceContainer'] = None
    
    def __new__(cls) -> 'ServiceContainer':
        """Singleton pattern for service container."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._services = {}
            cls._instance._factories = {}
            cls._instance._singletons = {}
        return cls._instance
    
    def register(self, name: str, factory: Callable, singleton: bool = True) -> None:
        """
        Register a service factory.
        
        Args:
            name: Service name
            factory: Callable that creates the service instance
            singleton: If True, cache the instance (default: True)
        """
        self._factories[name] = factory
        self._singletons[name] = singleton
    
    def get(self, name: str) -> Any:
        """
        Get a service instance.
        
        Args:
            name: Service name
            
        Returns:
            Service instance
            
        Raises:
            KeyError: If service is not registered
        """
        if name not in self._factories:
            raise KeyError(f"Service '{name}' is not registered")
        
        # Return cached singleton if available
        if self._singletons.get(name) and name in self._services:
            return self._services[name]
        
        # Create new instance
        instance = self._factories[name]()
        
        # Cache if singleton
        if self._singletons.get(name):
            self._services[name] = instance
        
        return instance
    
    def has(self, name: str) -> bool:
        """Check if a service is registered."""
        return name in self._factories
    
    def clear(self, name: str = None) -> None:
        """
        Clear cached service instances.
        
        Args:
            name: Optional specific service to clear. If None, clears all.
        """
        if name:
            self._services.pop(name, None)
        else:
            self._services.clear()
    
    def reset(self) -> None:
        """Reset the entire container."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()


# Global container instance
container = ServiceContainer()


def register_services():
    """Register all application services."""
    from app.core.config import CONFIG, get_path, PROJECT_ROOT
    
    # Model services
    def load_model_service():
        from app.services.model_utils import load_model
        return load_model()
    
    def load_scaler_service():
        from app.services.model_utils import load_scaler
        return load_scaler()
    
    def load_encoder_service():
        from app.services.model_utils import load_label_encoder
        return load_label_encoder()
    
    # Data services
    def load_participants_service():
        from app.services.data_access import load_participants
        return load_participants()
    
    def load_baseline_results_service():
        from app.services.data_access import load_baseline_results
        return load_baseline_results()
    
    def load_improvement_results_service():
        from app.services.data_access import load_improvement_results
        return load_improvement_results()
    
    # Register all services
    container.register('model', load_model_service, singleton=True)
    container.register('scaler', load_scaler_service, singleton=True)
    container.register('label_encoder', load_encoder_service, singleton=True)
    container.register('participants', load_participants_service, singleton=True)
    container.register('baseline_results', load_baseline_results_service, singleton=True)
    container.register('improvement_results', load_improvement_results_service, singleton=True)


def get_service(name: str) -> Any:
    """
    Get a service from the container.
    
    Args:
        name: Service name
        
    Returns:
        Service instance
    """
    return container.get(name)


def inject(*service_names: str):
    """
    Decorator for dependency injection.
    
    Usage:
        @inject('model', 'scaler')
        def my_function(model, scaler, other_arg):
            ...
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Inject services as keyword arguments
            for name in service_names:
                if name not in kwargs:
                    kwargs[name] = container.get(name)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Initialize services at import time
try:
    register_services()
except Exception:
    # Services will be registered on first use
    pass
