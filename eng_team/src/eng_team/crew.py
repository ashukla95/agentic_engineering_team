from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from eng_team.module.pydantic.agent_output import (
    BusinessUseCase
)


@CrewBase
class EngTeam():
    """EngTeam crew"""
    
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    agents: List[BaseAgent]
    tasks: List[Task]


    @agent
    def business_personal(self) -> Agent:
        return Agent(
            config=self.agents_config["business_personal"]
        )
    
    @task
    def business_usecase_generation_task(self) -> Task:
        return Task(
            config=self.tasks_config["business_usecase_generation_task"],
            output_pydantic=BusinessUseCase
        )


    @crew
    def crew(self) -> Crew:
        """Creates the EngTeam crew"""

        print(f"task config: {self.tasks_config}")

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
