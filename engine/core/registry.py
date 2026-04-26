from typing import Dict, Type, Any, Optional

class ModelRegistry:
    _models: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(model_class: Type):
            cls._models[name] = model_class
            return model_class
        return wrapper

    @classmethod
    def get_model(cls, name: str) -> Optional[Type]:
        return cls._models.get(name)

    @classmethod
    def list_models(cls) -> list[str]:
        return list(cls._models.keys())
