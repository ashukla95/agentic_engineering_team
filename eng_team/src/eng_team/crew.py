from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class EngTeam():
    """EngTeam crew"""
    
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"


    @agent
    def business_use_case_generator(self) -> Agent:
        return Agent(
            config=self.agents_config["business_personal"]
        )


    @crew
    def crew(self) -> Crew:
        """Creates the EngTeam crew"""

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
