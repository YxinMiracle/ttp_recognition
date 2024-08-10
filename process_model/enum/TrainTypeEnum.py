class TrainType:
    def __init__(self, train_type_name: str, train_type_id: int):
        self.train_type_name = train_type_name
        self.train_type_id = train_type_id

    def get_enum_name(self) -> str:
        return self.train_type_name

    def get_enum_value(self) -> int:
        return self.train_type_id


# 定义枚举类
class TrainTypeEnum:
    VAL_TYPE = TrainType("val", 0)
    TEST_TYPE = TrainType("test", 1)
