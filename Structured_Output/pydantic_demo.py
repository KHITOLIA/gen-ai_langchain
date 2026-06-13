from pydantic import BaseModel, EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name: str = 'Tushar'  # default value set
    age: Optional[int] = None  # because 
    email: EmailStr
    cgpa: float = Field(gt = 0, lt = 10, default=2, description="A Decimal value representing the cgpa of the student")

new_student = {'age' : '32', 'email': 'abc@gmail.com'} # type ceorcing

student = Student(**new_student)
student_dict = dict(student)
print(student_dict)

student_json = student.model_dump_json()