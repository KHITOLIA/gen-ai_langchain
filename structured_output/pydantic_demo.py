from pydantic import BaseModel, Field, EmailStr
from typing import Optional

class Student(BaseModel):
    name: str
    place: str = 'New Delhi' # setting default value to any attribute
    age : Optional[int] = None
    # email : EmailStr
    cgpa : float = Field(gt = 0, lt = 10, description = "Decimal value representing Student CGPA :")

new_student = {'name' : '32', 'age' : '30','cgpa' : 5}  
# Data Validate cannot pass integer values ,Type coercing : converting data type from str to int

student = Student(**new_student)
new_student = dict(student)
print(new_student)

student_json = student.model_dump_json()
