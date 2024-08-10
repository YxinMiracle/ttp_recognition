class EvaluateType:
    def __init__(self, evaluate_type_name: str, evaluate_type_id: int):
        self.evaluate_type_name = evaluate_type_name
        self.evaluate_type_id = evaluate_type_id

    def get_enum_name(self) -> str:
        return self.evaluate_type_name

    def get_enum_value(self) -> int:
        return self.evaluate_type_id


# 定义枚举类
class EvaluateTypeEnum:
    Coverage_Error = EvaluateType("Coverage_Error", 0)
    LRAP = EvaluateType("LRAP", 1)
    Label_Ranking_Loss = EvaluateType("Label_Ranking_Loss", 2)
    Hamming_Loss = EvaluateType("Hamming_Loss", 3)
    Precision_Score_Samples = EvaluateType("Precision_Score_Samples", 4)
    Precision_Score_Macro = EvaluateType("Precision_Score_Macro", 5)
    Precision_Score_Micro = EvaluateType("Precision_Score_Micro", 6)
    Recall_Score_Samples = EvaluateType("Recall_Score_Samples", 7)
    Recall_Score_Macro = EvaluateType("Recall_Score_Macro", 8)
    Recall_Score_Micro = EvaluateType("Recall_Score_Micro", 9)
    F1_Score_Samples = EvaluateType("F1_Score_Samples", 10)
    F1_Score_Macro = EvaluateType("F1_Score_Macro", 11)
    F1_Score_Micro = EvaluateType("F1_Score_Micro", 12)
    F05_Score_Samples = EvaluateType("F05_Score_Samples", 13)
    F05_Score_Macro = EvaluateType("F05_Score_Macro", 14)
    F05_Score_Micro = EvaluateType("F05_Score_Micro", 15)
    Accuracy_Score = EvaluateType("Accuracy_Score", 16)
