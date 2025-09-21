from pydantic import BaseModel


class BusinessUseCase(BaseModel):
    scenario: str


class EngineeringDesignTask(BaseModel):
    task_name: str
    task_description: str

    
class EngineeringDesignTaskList(BaseModel):
    tasks: list