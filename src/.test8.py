def get_string_representation_of_dict_without_quotation_marks(d: dict) -> str:
    return str(d).replace("'", '')

d = {1:2, 3:4}
print(f"{ get_string_representation_of_dict_without_quotation_marks(d)}")