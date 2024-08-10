class ModelType:
    def __init__(self, model_type_name: str, model_type_id: int):
        self.model_type_name = model_type_name
        self.model_type_id = model_type_id

    def get_enum_name(self) -> str:
        return self.model_type_name

    def get_enum_value(self) -> int:
        return self.model_type_id


# 定义枚举类
class ModelTypeEnum:
    TACTIC = ModelType("TACTIC", 0)
    TECHNIQUE = ModelType("TECHNIQUE", 1)

