from crewai import (
    Agent,
    Crew,
    Process,
    Task
)
from crewai.project import (
    CrewBase,
    agent,
    crew,
    task
)
from crewai.agents.agent_builder.base_agent import (
    BaseAgent
)
from pydantic import (
    BaseModel
)
from typing import List


@CrewBase
class SoftwareEngineer:
    """Software Wngineer"""

    agents: List[BaseAgent]
    tasks: List[Task]
    agents_config: "config/agents.yaml"
    tasks_config: "config/tasks.yaml"

    def __init__(
        self,
        output_file_name: str,
        *args,
        **kwargs
    ):
        super().__init__(
            *args, **kwargs
        )
        self.output_file_name = output_file_name

    @agent
    def software_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config[
                "software_engineer"
            ],
            allow_code_execution=True
        )
    
    @task
    def generate_code(self) -> Task:
        return Task(
            config=self.tasks_config[
                "generate_code"
            ],
            output_file=self.output_file_name
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )