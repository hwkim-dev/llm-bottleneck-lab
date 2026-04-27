from typing import Dict, Type, Any, Optional

class ModelRegistry:
    _models: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model adapter."""
        def wrapper(model_class: Type):
            cls._models[name] = model_class
            return model_class
        return wrapper

    @classmethod
    def get_model(cls, name: str) -> Optional[Type]:
        """Get a model class by name."""
        return cls._models.get(name)

    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered models."""
        return list(cls._models.keys())

    @classmethod
    def has_model(cls, name: str) -> bool:
        """Check if a model adapter is registered."""
        return name in cls._models

    @classmethod
    def create_model(cls, name: str, config: Any, runtime: Any) -> Any:
        """Create a model adapter instance."""
        model_cls = cls.get_model(name)
        if model_cls is None:
            raise ValueError(f"Unknown model type '{name}'. Registered models: {cls.list_models()}")
        return model_cls(config, runtime)
