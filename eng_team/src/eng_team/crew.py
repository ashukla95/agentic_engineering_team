from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from eng_team.module.pydantic.agent_output import (
    BusinessUseCase,
    EngineeringDesignTaskList
)


@CrewBase
class EngTeam():
    """EngTeam crew"""
    
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    agents: List[BaseAgent]
    tasks: List[Task]

    llm = LLM(
        model="gpt-4o-mini",
        temoerature=0
    )


    @agent
    def business_personal(self) -> Agent:
        return Agent(
            config=self.agents_config["business_personal"],
            llm=self.llm,
            max_iter=1
        )

    @agent
    def technical_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["technical_manager"],
            llm=self.llm,
            max_iter=10,
            reasoning=True  # Adding this as we need to reason to create proper tasks.
        )

    # allow_code_execution=True  # this should come from a config class ideally or better declare it in the config file itself  # noqa
    
    @task
    def business_usecase_generation_task(self) -> Task:
        return Task(
            config=self.tasks_config["business_usecase_generation_task"],
            output_pydantic=BusinessUseCase
        )
    
    @task
    def generate_engineering_task_list(self) -> Task:
        return Task(
            config=self.tasks_config["generate_engineering_task_list"],
            output_pydantic=EngineeringDesignTaskList
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
